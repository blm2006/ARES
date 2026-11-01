"""Worker: queue -> embedding -> PCA (đã freeze) -> faiss dedup -> pre-annotation
-> review_queue.

Process riêng, không đụng gì tới realtime. Nếu worker chết, realtime vẫn chạy và
queue chỉ dày lên cho tới khi backpressure ngắt việc lấy mẫu — đúng thiết kế.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from datetime import datetime

import numpy as np
from sqlalchemy import select

from common.config import Settings, get_settings
from common.db import Frame as FrameRow
from common.db import ReviewItem, init_db, session_scope, utcnow
from common.queue import QueueBackend, get_queue
from common.schemas import QueueMessage
from common.storage import StorageBackend, get_storage
from worker.coreset import select_coreset
from worker.dedup import Deduplicator, FaissStore, bump_stat
from worker.embedding import build_embedder
from worker.pca import load_or_raise
from worker.preannotate import PreAnnotator, review_priority

log = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        settings: Settings | None = None,
        queue: QueueBackend | None = None,
        storage: StorageBackend | None = None,
        detector=None,
        embedder_kind: str = "hist",
    ):
        self.s = settings or get_settings()
        self.q = queue or get_queue(self.s)
        self.storage = storage or get_storage(self.s)
        self.embedder = build_embedder(embedder_kind, self.s.yolo_weights, self.s.device, self.s.imgsz)
        self.pca = load_or_raise(self.s.pca_path, self.s.pca_version)
        self.store = FaissStore(self.s.faiss_path, self.pca.n_components, self.s.pca_version)
        self.dedup = Deduplicator(self.store, self.s.dedup_theta, self.s.dedup_theta_failure)
        self.preannotator = PreAnnotator(detector, model_version=self.s.yolo_weights)
        self._stop = False
        self.processed = 0
        self.discarded = 0
        self.queued_for_review = 0

    # ------------------------------------------------------------------
    def run(self, max_messages: int | None = None, idle_timeout_s: float | None = None) -> None:
        """idle_timeout_s: dừng khi queue rỗng quá lâu (dùng cho chạy theo lô / test).
        None = chạy mãi, đây là chế độ service bình thường."""
        log.info("worker bắt đầu — index hiện có %d vector", len(self.store))
        if idle_timeout_s is None and max_messages is not None:
            idle_timeout_s = 5.0
        last_save = time.monotonic()
        last_work = time.monotonic()
        while not self._stop:
            batch = self.q.consume(count=self.s.worker_batch_size, block_ms=2000)
            if not batch:
                if idle_timeout_s is not None and time.monotonic() - last_work > idle_timeout_s:
                    break
                time.sleep(0.05)
                continue
            last_work = time.monotonic()
            self.process_batch(batch)
            if time.monotonic() - last_save > 30:
                self.store.save()
                last_save = time.monotonic()
            if max_messages and self.processed >= max_messages:
                break
        self.store.save()
        log.info(
            "worker dừng: %d xử lý, %d discard, %d vào review",
            self.processed, self.discarded, self.queued_for_review,
        )

    # ------------------------------------------------------------------
    def process_batch(self, batch: list[tuple[str, QueueMessage]]) -> None:
        images, msgs, entry_ids = [], [], []
        for entry_id, msg in batch:
            try:
                images.append(self.storage.read_image(msg.path))
                msgs.append(msg)
                entry_ids.append(entry_id)
            except FileNotFoundError:
                log.warning("ảnh không tồn tại, ack bỏ: %s", msg.path)
                self.q.ack(entry_id)

        if not msgs:
            return

        raw = self.embedder.embed(images)
        vecs = self.pca.transform(raw)

        survivors: list[int] = []
        for i, msg in enumerate(msgs):
            day = msg.ts.strftime("%Y-%m-%d")
            r = self.dedup.check(vecs[i], msg.source)

            # Trùng TRONG CÙNG BATCH cũng phải bắt: index chỉ được cập nhật ở
            # _finalize, nên nếu chỉ so với index thì 32 frame giống hệt nhau trong
            # một batch sẽ vào dataset đủ cả 32.
            if not r.duplicate and survivors:
                intra = float(np.max(vecs[survivors] @ vecs[i]))
                if intra > r.theta:
                    r.duplicate = True
                    r.similarity = intra
                    r.neighbor_id = msgs[survivors[int(np.argmax(vecs[survivors] @ vecs[i]))]].frame_id

            bump_stat(day, msg.source.value, r.duplicate)
            with session_scope() as s:
                row = s.get(FrameRow, msg.frame_id)
                if row is not None:
                    row.embedded = True
                    row.pca_version = self.s.pca_version
                    row.dedup_similarity = float(r.similarity)
                    row.dedup_discarded = bool(r.duplicate)
                    row.dedup_neighbor_id = r.neighbor_id
            if r.duplicate:
                self.discarded += 1
                self.storage.delete(msg.path)  # trùng thì không giữ ảnh làm gì
            else:
                survivors.append(i)

        # Core-set: trong số còn lại, giữ tập phủ không gian đặc trưng rộng nhất.
        if self.s.coreset_enabled and len(survivors) > 2:
            sub = vecs[survivors]
            keep_local = set(select_coreset(sub, self.s.coreset_keep_ratio))
            dropped = [survivors[j] for j in range(len(survivors)) if j not in keep_local]
            for j in dropped:
                self.storage.delete(msgs[j].path)
                with session_scope() as s:
                    row = s.get(FrameRow, msgs[j].frame_id)
                    if row is not None:
                        row.dedup_discarded = True
            survivors = [survivors[j] for j in sorted(keep_local)]

        for i in survivors:
            self._finalize(images[i], msgs[i], vecs[i])

        for entry_id in entry_ids:
            self.q.ack(entry_id)
        self.processed += len(msgs)

    # ------------------------------------------------------------------
    def _finalize(self, image: np.ndarray, msg: QueueMessage, vec: np.ndarray) -> None:
        pre = self.preannotator.annotate(image, msg)
        self.dedup.add(vec, msg.frame_id)

        with session_scope() as s:
            row = s.get(FrameRow, msg.frame_id)
            if row is not None:
                row.preannotated = True
                row.preannot_model = pre.model_version
                row.preannot_boxes = pre.as_json()
                row.in_faiss = True

            # frame_id là UNIQUE trong review_queue. Một message có thể được giao lại
            # (worker chết sau khi xử lý nhưng trước khi ack), và khi đó INSERT lần hai
            # ném IntegrityError làm CHẾT CẢ WORKER giữa lô. Bỏ qua là đúng: mẫu đã
            # nằm trong hàng chờ duyệt rồi.
            exists = s.execute(
                select(ReviewItem.id).where(ReviewItem.frame_id == msg.frame_id)
            ).scalar_one_or_none()
            if exists is not None:
                log.debug("frame %s đã có trong review_queue, bỏ qua", msg.frame_id)
                return
            s.add(
                ReviewItem(
                    frame_id=msg.frame_id,
                    priority=review_priority(msg.source),
                    source=msg.source.value,
                    status="pending",
                    n_boxes_pre=len(pre.boxes),
                )
            )
        self.queued_for_review += 1

    def stop(self) -> None:
        self._stop = True


def main() -> None:
    ap = argparse.ArgumentParser(prog="worker")
    ap.add_argument("--embedder", choices=["hist", "yolo"], default="hist")
    ap.add_argument("--max-messages", type=int, default=None)
    ap.add_argument("--no-preannot-model", action="store_true",
                    help="không nạp YOLO, dùng box lúc bắt mẫu làm gợi ý")
    args = ap.parse_args()

    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()

    detector = None
    if not args.no_preannot_model:
        from realtime.detector import PersonDetector

        detector = PersonDetector(settings=s)

    w = Worker(s, detector=detector, embedder_kind=args.embedder)
    signal.signal(signal.SIGINT, lambda *_: w.stop())
    signal.signal(signal.SIGTERM, lambda *_: w.stop())
    w.run(max_messages=args.max_messages)


if __name__ == "__main__":
    main()
