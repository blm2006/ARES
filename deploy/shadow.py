"""Shadow mode: model mới chạy SONG SONG, CHỈ LOG, không ảnh hưởng output.

Điểm tinh tế: shadow không được làm tụt FPS của model production. Nếu GPU không
đủ cho hai model cùng lúc, shadow chạy trên MẪU frame (mỗi N frame một lần) —
sai số thống kê chấp nhận được, còn làm chậm realtime thì không.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from common.config import Settings, get_settings
from common.metrics import CameraTelemetry
from common.schemas import BBox
from realtime.tracker import TrackHistory

log = logging.getLogger(__name__)


@dataclass
class ShadowStats:
    frames: int = 0
    agree_count: int = 0        # số frame hai model cho cùng số người
    prod_people: list[int] = field(default_factory=list)
    shadow_people: list[int] = field(default_factory=list)
    extra_detections: int = 0   # shadow thấy mà production không thấy
    missed_detections: int = 0  # production thấy mà shadow không thấy

    def as_dict(self) -> dict:
        n = max(1, self.frames)
        return {
            "frames": self.frames,
            "count_agreement": round(self.agree_count / n, 4),
            "avg_people_prod": round(sum(self.prod_people) / n, 2),
            "avg_people_shadow": round(sum(self.shadow_people) / n, 2),
            "extra_detections": self.extra_detections,
            "missed_detections": self.missed_detections,
        }


class ShadowRunner:
    """Chạy detector thứ hai trên cùng frame, ghi telemetry với role='shadow'."""

    def __init__(
        self,
        cam_id: str,
        detector,
        model_version: str,
        every_n_frames: int = 3,
        settings: Settings | None = None,
    ):
        self.s = settings or get_settings()
        self.cam_id = cam_id
        self.detector = detector
        self.every_n = max(1, every_n_frames)
        self.telemetry = CameraTelemetry(cam_id, model_version=model_version, role="shadow")
        self.history = TrackHistory(self.s.track_history_len, self.s.short_track_frames)
        self.stats = ShadowStats()
        self._lock = threading.Lock()

    def observe(self, image: np.ndarray, frame_idx: int, prod_boxes: list[BBox]) -> None:
        if frame_idx % self.every_n:
            return
        try:
            res = self.detector.track(image)
        except Exception:
            log.exception("[%s] shadow infer lỗi", self.cam_id)
            return

        events = self.history.add(frame_idx, res.boxes)
        self.telemetry.on_id_switch(sum(1 for e in events if e.reason.value == "id_switch"))
        self.telemetry.on_frame_processed(res.inference_ms, len(res.boxes))
        self.telemetry.maybe_flush()

        with self._lock:
            self.stats.frames += 1
            self.stats.prod_people.append(len(prod_boxes))
            self.stats.shadow_people.append(len(res.boxes))
            if len(prod_boxes) == len(res.boxes):
                self.stats.agree_count += 1
            matched_prod = sum(
                1 for p in prod_boxes if any(p.iou(sb) > 0.5 for sb in res.boxes)
            )
            self.stats.missed_detections += len(prod_boxes) - matched_prod
            matched_shadow = sum(
                1 for sb in res.boxes if any(sb.iou(p) > 0.5 for p in prod_boxes)
            )
            self.stats.extra_detections += len(res.boxes) - matched_shadow

    def report(self) -> dict:
        with self._lock:
            return self.stats.as_dict()
