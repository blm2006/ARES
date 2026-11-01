"""Collector — ghép sampling router + ba nguồn + writer thành một object.

Gọi từ vòng lặp realtime qua `Collector.on_frame(...)`, và nó phải RẺ: mọi việc
nặng (embedding, dedup, pre-annotation) nằm ở worker, không nằm ở đây.
Chi phí mỗi frame ở đây = 1 lần tính điểm bất định + vài lần tra quota trong SQLite.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from collector.backpressure import BackpressureGuard
from collector.failure_mining import FailureMiner
from collector.router import SamplingRouter
from collector.uncertainty import UncertaintySampler
from collector.writer import FrameWriter
from common.config import Settings, get_settings
from common.queue import get_queue
from common.schemas import BBox, Source
from common.storage import get_storage
from realtime.tracker import FailureEvent, TrackHistory

log = logging.getLogger(__name__)


@dataclass
class CollectStats:
    seen: int = 0
    accepted: dict[str, int] = field(default_factory=dict)
    rejected: dict[str, int] = field(default_factory=dict)

    def acc(self, source: str) -> None:
        self.accepted[source] = self.accepted.get(source, 0) + 1

    def rej(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


class Collector:
    def __init__(self, settings: Settings | None = None, queue=None, storage=None):
        self.s = settings or get_settings()
        self.q = queue or get_queue(self.s)
        self.storage = storage or get_storage(self.s)
        self.guard = BackpressureGuard(self.q, self.storage, self.s)
        self.router = SamplingRouter(self.guard, self.s)
        self.writer = FrameWriter(self.storage, self.q, self.s)
        self.uncertainty = UncertaintySampler(
            self.s.uncertainty_top_percentile,
            self.s.uncertainty_window_hours,
            self.s.uncertainty_min_samples,
        )
        self.failure = FailureMiner()
        self.stats = CollectStats()
        self._rng = random.Random(1337)
        self._last_accept_ts: dict[str, float] = {}  # cam_id -> ts frame gần nhất đã đưa duyệt

    # ------------------------------------------------------------------
    def on_frame(
        self,
        image: np.ndarray,
        cam_id: str,
        ts: datetime,
        frame_idx: int,
        boxes: list[BBox],
        failure_events: list[FailureEvent] | None = None,
        history: TrackHistory | None = None,
    ) -> list[str]:
        """Trả về danh sách frame_id đã ghi (thường 0 hoặc 1)."""
        self.stats.seen += 1
        written: list[str] = []

        # Luôn quan sát để nuôi phân phối, kể cả khi không lấy mẫu.
        score = self.uncertainty.observe(boxes, ts.timestamp())

        # Giãn cách thời gian: 2 frame đưa duyệt của cùng camera phải cách nhau
        # >= sample_min_gap_s. Chỉ chặn nguồn 'nhàm' (uncertainty/random); failure-mining
        # (frame model thấy THAY ĐỔI/khó) luôn được qua.
        now_ts = ts.timestamp()
        gap = self.s.sample_min_gap_s
        gap_ok = gap <= 0 or (now_ts - self._last_accept_ts.get(cam_id, -1e18)) >= gap

        if not self.guard.can_write():
            self.stats.rej(f"backpressure:{self.guard.state().level}")
            return written

        # --- Nguồn B: failure mining (ưu tiên cao nhất, quota 40%) ---
        for sample in self.failure.select(failure_events or []):
            d = self.router.route(Source.FAILURE_MINING, cam_id, ts)
            if not d.accept:
                self.stats.rej(d.reason)
                break
            neighbors = sample.neighbor_boxes
            if history is not None and not neighbors:
                neighbors = history.neighbor_suggestions(sample.frame_idx)
            fid = self.writer.write(
                image=image,
                cam_id=cam_id,
                ts=ts,
                frame_idx=frame_idx,
                source=Source.FAILURE_MINING,
                boxes=boxes,
                failure_reason=sample.reason,
                neighbor_boxes=neighbors,
            )
            if fid:
                self.router.accept(Source.FAILURE_MINING, cam_id, ts)
                self.stats.acc(Source.FAILURE_MINING.value)
                self._last_accept_ts[cam_id] = now_ts
                written.append(fid)
            break  # 1 frame -> tối đa 1 bản ghi, dù có nhiều sự cố

        if written:
            return written

        # --- Nguồn A: uncertainty theo phân vị (quota 40%) ---
        if gap_ok and self.uncertainty.should_sample(score):
            d = self.router.route(Source.UNCERTAINTY, cam_id, ts)
            if d.accept:
                fid = self.writer.write(
                    image=image,
                    cam_id=cam_id,
                    ts=ts,
                    frame_idx=frame_idx,
                    source=Source.UNCERTAINTY,
                    boxes=boxes,
                    uncertainty_score=score,
                )
                if fid:
                    self.router.accept(Source.UNCERTAINTY, cam_id, ts)
                    self.stats.acc(Source.UNCERTAINTY.value)
                    self._last_accept_ts[cam_id] = now_ts
                    return [fid]
            else:
                self.stats.rej(d.reason)

        # --- Nguồn C: random baseline (quota 20%) ---
        # Bỏ qua MỌI tiêu chí khác. Đây là nhóm đối chứng: nếu active learning
        # không thắng nhóm này trên eval set thì toàn bộ nguồn A đang là chi phí vô ích.
        if gap_ok and self._rng.random() < self.s.random_sample_rate:
            d = self.router.route(Source.RANDOM, cam_id, ts)
            if d.accept:
                fid = self.writer.write(
                    image=image,
                    cam_id=cam_id,
                    ts=ts,
                    frame_idx=frame_idx,
                    source=Source.RANDOM,
                    boxes=boxes,
                )
                if fid:
                    self.router.accept(Source.RANDOM, cam_id, ts)
                    self.stats.acc(Source.RANDOM.value)
                    self._last_accept_ts[cam_id] = now_ts
                    return [fid]
            else:
                self.stats.rej(d.reason)

        return written

    def snapshot(self) -> dict:
        st = self.guard.state()
        return {
            "seen": self.stats.seen,
            "accepted": self.stats.accepted,
            "rejected": self.stats.rejected,
            "backpressure": st.level.value,
            "queue_depth": st.queue_depth,
            "disk_free_pct": round(st.disk_free_pct, 1),
            "uncertainty": self.uncertainty.stats(),
            "quota_today": self.router.quota.usage(),
        }
