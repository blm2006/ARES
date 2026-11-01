"""Thread 4 (phần thu thập + duyệt): Controlled Active Learning, human-in-the-loop.

Luồng:
  camera (điều khiển hoặc camera phụ) gặp detection có 0.35 <= conf <= 0.55
    -> offer(): kiểm tra rẻ (giãn cách theo camera, ảnh trùng thô, đĩa) -> bỏ vào hàng đợi nhỏ, KHÔNG chặn
    -> worker: ghi ảnh GỐC (không resize/crop) + nhãn YOLO tạm vào storage/pending/, tính vector đặc trưng, thêm dòng
       CANDIDATE (hoặc PENDING nếu tắt lọc PCA)
  LỌC PCA LẦN 1 (ares.pca_filter): CANDIDATE -> PENDING (giữ) | DUPLICATE (ảnh thừa)
  Admin duyệt trên dashboard: Approve / Edit BBox / Reject
    -> APPROVED được chuyển sang storage/approved/ cùng nhãn cuối cùng
  00:00: LỌC PCA LẦN 2 chọn ảnh đại diện -> ares.retrain fine-tune từ checkpoint gốc, trộn replay buffer, cổng chất lượng,
         rồi mới promote.

Không còn trần "50 ảnh/ngày" (al_daily_cap = 0): số lượng được kiểm soát bằng lọc PCA chứ không bằng cách bỏ ảnh mới.

Không có đường nào đưa nhãn do model tự sinh vào tập huấn luyện mà chưa qua người duyệt.
"""

from __future__ import annotations

import logging
import math
import queue
import shutil
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
from sqlalchemy import func, select

from ares.config import AresSettings
from ares.db_manager import ActiveLearningItem, DBManager
from ares.detector import Det
from ares.state import Event

log = logging.getLogger(__name__)

STATUSES = ("CANDIDATE", "PENDING", "APPROVED", "REJECTED", "TRAINED", "DUPLICATE")


# ---------------------------------------------------------------------------
# tiện ích ảnh (an toàn với đường dẫn có dấu tiếng Việt trên Windows)
# ---------------------------------------------------------------------------
def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("không mã hoá được JPEG")
    return buf.tobytes()


def read_image(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    return img


def ahash(image: np.ndarray) -> int:
    """Average hash 64 bit: rẻ, đủ để loại ảnh gần như y hệt nhau (cảnh tĩnh)."""
    import cv2

    g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def yolo_label_text(boxes: list[dict], w: int, h: int) -> str:
    """boxes: [{"class":0,"bbox":[x1,y1,x2,y2],...}] (pixel) -> các dòng 'cls cx cy w h' chuẩn hoá."""
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{int(b.get('class', 0))} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def dets_to_json(dets: list[Det]) -> list[dict]:
    return [{"class": d.cls, "bbox": [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)],
             "conf": round(d.conf, 3)} for d in dets]


@dataclass
class _Job:
    image: np.ndarray
    dets: list[Det]
    when: datetime
    camera_id: str = "main"


class ActiveLearner:
    def __init__(self, cfg: AresSettings, clock, db: DBManager, emit: Callable | None = None, pca=None):
        self.cfg, self.clock, self.db, self.emit = cfg, clock, db, emit
        self.pca = pca  # ares.pca_filter.PcaFilter | None
        self.q: queue.Queue[_Job] = queue.Queue(maxsize=cfg.al_queue_size)
        self._lock = threading.Lock()
        self._day: date | None = None
        self._daily = 0
        self._cap_noted: date | None = None
        self._hourly: dict[int, int] = {}
        self._last_t: dict[str, float] = {}  # theo camera
        self._hashes: dict[str, deque[int]] = {}
        self.counters: dict[str, int] = {"offered": 0, "band_hits": 0, "accepted": 0, "saved": 0, "save_errors": 0}
        self.skipped: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        cfg.pending_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================ thu thập
    def _skip(self, why: str) -> bool:
        self.skipped[why] = self.skipped.get(why, 0) + 1
        return False

    def _roll_day(self, day: date) -> None:
        if self._day == day:
            return
        self._day = day
        self._hourly = {}
        start = datetime.combine(day, datetime.min.time())
        with self.db.session() as s:
            self._daily = s.execute(
                select(func.count()).select_from(ActiveLearningItem).where(
                    ActiveLearningItem.room_id == self.cfg.room_id,
                    ActiveLearningItem.captured_at >= start, ActiveLearningItem.captured_at < start + timedelta(days=1))
            ).scalar_one()

    def in_band(self, dets: list[Det]) -> bool:
        lo, hi, pc = self.cfg.al_conf_low, self.cfg.al_conf_high, self.cfg.person_class_id
        return any(d.cls == pc and lo <= d.conf <= hi for d in dets)

    def offer(self, image: np.ndarray, dets: list[Det], when: datetime | None = None, camera_id: str = "main") -> bool:
        """Gọi từ luồng vision/hub. Rẻ và không chặn. `image` không được bị sửa tại chỗ sau khi gọi.

        Trả True nếu khung hình được nhận vào hàng đợi ghi.
        """
        cfg = self.cfg
        if not cfg.al_enabled:
            return False
        self.counters["offered"] += 1
        if not self.in_band(dets):
            return False
        self.counters["band_hits"] += 1
        when = when or self.clock.now()
        t = when.timestamp()
        with self._lock:
            self._roll_day(when.date())
            if cfg.al_daily_cap > 0 and self._daily >= cfg.al_daily_cap:
                if self._cap_noted != when.date() and self.emit is not None:
                    self._cap_noted = when.date()
                    self.emit(Event(self.clock.time(), "AL_DAILY_CAP", "info",
                                    f"Đã đủ {cfg.al_daily_cap} ảnh học chủ động hôm nay — dừng thu thập đến hết ngày."))
                return self._skip("cap_ngày")
            if cfg.al_hourly_cap > 0 and self._hourly.get(when.hour, 0) >= cfg.al_hourly_cap:
                return self._skip("cap_giờ")
            if t - self._last_t.get(camera_id, -math.inf) < cfg.al_min_gap_s:
                return self._skip("giãn_cách")
            if self.q.full():
                return self._skip("hàng_đợi_đầy")
            try:
                if shutil.disk_usage(cfg.storage_root).free < 500 * 1024 * 1024:
                    return self._skip("hết_đĩa")
            except OSError:
                pass
            h = ahash(image)
            ring = self._hashes.setdefault(camera_id, deque(maxlen=64))
            if any(hamming(h, x) < cfg.al_hash_dist_min for x in ring):
                return self._skip("ảnh_trùng")
            ring.append(h)
            self._daily += 1
            self._hourly[when.hour] = self._hourly.get(when.hour, 0) + 1
            self._last_t[camera_id] = t
            self.counters["accepted"] += 1
        self.q.put_nowait(_Job(image, list(dets), when, camera_id))
        return True

    # ---- luồng ghi
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="al-saver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not (self._stop.is_set() and self.q.empty()):
            try:
                job = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.save(job)
            except Exception:
                self.counters["save_errors"] += 1
                with self._lock:  # hoàn lại hạn mức vì chưa lưu được
                    self._daily = max(0, self._daily - 1)
                log.exception("Active learning: ghi ảnh lỗi")
            finally:
                self.q.task_done()

    def drain(self, timeout: float = 10.0) -> bool:
        """Chờ ghi xong (test, tắt máy)."""
        done = threading.Event()
        threading.Thread(target=lambda: (self.q.join(), done.set()), daemon=True).start()
        return done.wait(timeout)

    def save(self, job: _Job) -> str:
        cfg = self.cfg
        h, w = job.image.shape[:2]
        image_id = f"{cfg.room_id[:12]}_{job.camera_id[:12]}_{job.when:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"[:50]
        jpg = cfg.pending_dir / f"{image_id}.jpg"
        txt = cfg.pending_dir / f"{image_id}.txt"
        jpg.write_bytes(encode_jpeg(job.image, cfg.jpeg_quality))  # ảnh gốc, độ phân giải gốc
        preds = dets_to_json(job.dets)
        txt.write_text(yolo_label_text(preds, w, h), encoding="utf-8")
        rel = jpg.relative_to(cfg.storage_root).as_posix()
        emb = None
        if self.pca is not None:
            try:
                from ares.pca_filter import embed_image, emb_to_bytes

                emb = emb_to_bytes(embed_image(job.image))
            except Exception:
                log.warning("Không tính được vector đặc trưng cho %s (sẽ tính lại khi lọc)", image_id, exc_info=True)
        status = "CANDIDATE" if (self.pca is not None and cfg.pca_enabled) else "PENDING"
        with self.db.session() as s:
            s.add(ActiveLearningItem(
                image_id=image_id, captured_at=job.when.replace(tzinfo=None), img_path=rel, pred_bboxes=preds,
                status=status, room_id=cfg.room_id, width=w, height=h, camera_id=job.camera_id, embedding=emb))
        self.counters["saved"] += 1
        return image_id

    # ================================================================ duyệt ảnh
    def _abs(self, rel: str) -> Path:
        return self.cfg.storage_root / rel

    def _row(self, s, image_id: str) -> ActiveLearningItem:
        it = s.get(ActiveLearningItem, image_id)
        if it is None:
            raise KeyError(image_id)
        return it

    def _relocate(self, it: ActiveLearningItem, dst_dir: Path, label_text: str | None) -> str:
        """Copy ảnh (+ nhãn) sang thư mục đích; trả về đường dẫn tương đối mới. Chưa xoá bản cũ."""
        src = self._abs(it.img_path)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if src.exists():
            if src.resolve() != dst.resolve():  # sửa lại ảnh đã APPROVED: ảnh đã ở đúng chỗ
                shutil.copy2(src, dst)
        elif not dst.exists():
            raise FileNotFoundError(f"Mất file ảnh {src}")
        if label_text is not None:
            dst.with_suffix(".txt").write_text(label_text, encoding="utf-8")
        return dst.relative_to(self.cfg.storage_root).as_posix()

    def _cleanup_old(self, old_rel: str, new_rel: str) -> None:
        if old_rel == new_rel:
            return
        for suffix in (".jpg", ".txt"):
            p = self._abs(old_rel).with_suffix(suffix)
            try:
                p.unlink(missing_ok=True)
            except OSError:
                log.warning("Không xoá được %s", p)

    def approve(self, image_id: str) -> dict:
        """Nhãn tạm đúng hoàn toàn (kể cả việc không thiếu người nào) -> APPROVED."""
        with self.db.session() as s:
            it = self._row(s, image_id)
            if it.status != "PENDING":
                raise ValueError(f"Ảnh đang ở trạng thái {it.status}, chỉ duyệt được ảnh PENDING")
            final = [dict(b) for b in (it.pred_bboxes or [])]
            old = it.img_path
            new = self._relocate(it, self.cfg.approved_dir, yolo_label_text(final, it.width or 1, it.height or 1))
            it.img_path, it.final_bboxes, it.status = new, final, "APPROVED"
            it.edited, it.train_pick = False, None
            it.reviewed_at = self.clock.naive_now()
        self._cleanup_old(old, new)
        return {"image_id": image_id, "status": "APPROVED", "n_boxes": len(final)}

    def edit(self, image_id: str, boxes: list[dict]) -> dict:
        """Người sửa bbox (thêm người bị bỏ sót, xoá dương tính giả, chỉnh lệch) rồi chấp nhận -> APPROVED.
        Danh sách rỗng nghĩa là 'khung hình này không có ai' (mẫu âm tính hợp lệ)."""
        with self.db.session() as s:
            it = self._row(s, image_id)
            if it.status not in ("PENDING", "APPROVED"):
                raise ValueError(f"Ảnh đang ở trạng thái {it.status}, không sửa được")
            w, h = it.width or 1, it.height or 1
            final = self._validate_boxes(boxes, w, h)
            old = it.img_path
            base_dir = self.cfg.approved_dir
            new = self._relocate(it, base_dir, yolo_label_text(final, w, h))
            it.img_path, it.final_bboxes, it.status = new, final, "APPROVED"
            it.edited, it.train_pick = True, None  # người đã sửa: mẫu khó, lọc PCA lần 2 luôn giữ
            it.reviewed_at = self.clock.naive_now()
        self._cleanup_old(old, new)
        return {"image_id": image_id, "status": "APPROVED", "n_boxes": len(final)}

    def restore(self, image_id: str) -> dict:
        """Khôi phục ảnh bị lọc trùng (DUPLICATE) về hàng chờ duyệt."""
        with self.db.session() as s:
            it = self._row(s, image_id)
            if it.status != "DUPLICATE":
                raise ValueError(f"Ảnh đang ở trạng thái {it.status}, chỉ khôi phục được ảnh DUPLICATE")
            if not self._abs(it.img_path).exists():
                raise FileNotFoundError(str(self._abs(it.img_path)))
            it.status, it.dup_of, it.dup_dist = "PENDING", None, None
        return {"image_id": image_id, "status": "PENDING"}

    def reject(self, image_id: str, reason: str = "") -> dict:
        with self.db.session() as s:
            it = self._row(s, image_id)
            if it.status != "PENDING":
                raise ValueError(f"Ảnh đang ở trạng thái {it.status}, chỉ loại được ảnh PENDING")
            old = it.img_path
            new = self._relocate(it, self.cfg.rejected_dir, None)
            it.img_path, it.status, it.reject_reason = new, "REJECTED", (reason or "")[:100]
            it.reviewed_at = self.clock.naive_now()
        self._cleanup_old(old, new)
        return {"image_id": image_id, "status": "REJECTED"}

    @staticmethod
    def _validate_boxes(boxes: list[dict], w: int, h: int) -> list[dict]:
        if len(boxes) > 300:
            raise ValueError("Quá nhiều bbox")
        out = []
        for b in boxes:
            try:
                x1, y1, x2, y2 = (float(v) for v in b["bbox"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("bbox phải dạng {'bbox':[x1,y1,x2,y2]}") from None
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                raise ValueError("bbox chứa giá trị không hợp lệ")
            x1, x2 = sorted((min(max(x1, 0.0), w), min(max(x2, 0.0), w)))
            y1, y2 = sorted((min(max(y1, 0.0), h), min(max(y2, 0.0), h)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue  # hộp suy biến sau khi kẹp vào khung ảnh
            out.append({"class": int(b.get("class", 0)), "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "conf": 1.0})
        return out

    # ---- truy vấn cho dashboard ------------------------------------------------
    @staticmethod
    def _to_dict(it: ActiveLearningItem) -> dict:
        return {
            "image_id": it.image_id, "captured_at": it.captured_at.isoformat(), "status": it.status,
            "pred_bboxes": it.pred_bboxes or [], "final_bboxes": it.final_bboxes,
            "width": it.width, "height": it.height, "reject_reason": it.reject_reason, "trained_in": it.trained_in,
            "camera_id": it.camera_id, "dup_of": it.dup_of, "dup_dist": it.dup_dist, "edited": bool(it.edited),
            "train_pick": it.train_pick,
            "reviewed_at": it.reviewed_at.isoformat() if it.reviewed_at else None,
        }

    def list_items(self, status: str = "PENDING", limit: int = 20, offset: int = 0) -> list[dict]:
        with self.db.session() as s:
            q = select(ActiveLearningItem).where(ActiveLearningItem.room_id == self.cfg.room_id)
            if status != "ALL":
                q = q.where(ActiveLearningItem.status == status)
            rows = s.execute(q.order_by(ActiveLearningItem.captured_at.desc()).offset(offset).limit(limit)).scalars().all()
            return [self._to_dict(r) for r in rows]

    def get_item(self, image_id: str) -> dict:
        with self.db.session() as s:
            return self._to_dict(self._row(s, image_id))

    def image_path(self, image_id: str) -> Path:
        with self.db.session() as s:
            it = self._row(s, image_id)
            p = self._abs(it.img_path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p

    def counts(self) -> dict[str, int]:
        with self.db.session() as s:
            rows = s.execute(select(ActiveLearningItem.status, func.count()).where(
                ActiveLearningItem.room_id == self.cfg.room_id).group_by(ActiveLearningItem.status)).all()
        out = {k: 0 for k in STATUSES}
        out.update({k: int(v) for k, v in rows})
        return out

    def stats(self) -> dict:
        with self._lock:
            self._roll_day(self.clock.now().date())
            today = self._daily
        cap = self.cfg.al_daily_cap
        return {
            "counts": self.counts(),
            "today": {"captured": today, "cap": cap or None,
                      "remaining": max(0, cap - today) if cap else None},
            "band": [self.cfg.al_conf_low, self.cfg.al_conf_high],
            "counters": dict(self.counters),
            "skipped": dict(self.skipped),
            "queue_depth": self.q.qsize(),
        }

    def items_for_training(self) -> list[ActiveLearningItem]:
        """APPROVED (mới) + TRAINED (đã dùng) — huấn luyện luôn từ checkpoint gốc trên toàn bộ dữ liệu người đã duyệt."""
        with self.db.session() as s:
            return list(s.execute(select(ActiveLearningItem).where(
                ActiveLearningItem.room_id == self.cfg.room_id,
                ActiveLearningItem.status.in_(("APPROVED", "TRAINED")))).scalars().all())

    def mark_trained(self, image_ids: list[str], version: str) -> int:
        with self.db.session() as s:
            n = 0
            for iid in image_ids:
                it = s.get(ActiveLearningItem, iid)
                if it is not None and it.status == "APPROVED":
                    it.status, it.trained_in = "TRAINED", version
                    n += 1
            return n

    # ================================================================ dọn dẹp (TTL)
    def purge(self) -> dict:
        cfg = self.cfg
        now = self.clock.naive_now()
        removed = {"rejected": 0, "pending": 0, "duplicate": 0}
        with self.db.session() as s:
            old_rej = s.execute(select(ActiveLearningItem).where(
                ActiveLearningItem.status == "REJECTED",
                ActiveLearningItem.reviewed_at < now - timedelta(days=cfg.rejected_ttl_days))).scalars().all()
            old_pen = s.execute(select(ActiveLearningItem).where(
                ActiveLearningItem.status == "PENDING",
                ActiveLearningItem.captured_at < now - timedelta(days=cfg.pending_ttl_days))).scalars().all()
            old_dup = s.execute(select(ActiveLearningItem).where(
                ActiveLearningItem.status == "DUPLICATE",
                ActiveLearningItem.captured_at < now - timedelta(days=cfg.duplicate_ttl_days))).scalars().all()
            for it, key in ([(i, "rejected") for i in old_rej] + [(i, "pending") for i in old_pen]
                            + [(i, "duplicate") for i in old_dup]):
                for suf in (".jpg", ".txt"):
                    self._abs(it.img_path).with_suffix(suf).unlink(missing_ok=True)
                s.delete(it)
                removed[key] += 1
        return removed
