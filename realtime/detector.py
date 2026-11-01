"""YOLO detector đa lớp theo config (DETECT_CLASS_IDS). FP16 trên GPU.

Lưu ý spec: bài toán là PERSON, không phải face. Nếu sau này cần face, đó là
pipeline thứ hai nối sau person crop — không sửa file này.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from common.config import Settings, get_settings
from common.schemas import BBox

log = logging.getLogger(__name__)



@dataclass
class InferResult:
    boxes: list[BBox]
    inference_ms: float
    raw = None  # giữ tham chiếu kết quả gốc cho embedding/tracker nếu cần


class PersonDetector:
    def __init__(self, weights: str | None = None, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.weights = weights or self.s.yolo_weights
        self.model = None
        self._load()

    def _load(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(self.weights)
        log.info("YOLO nạp từ %s, device=%s half=%s", self.weights, self.s.device, self.s.half)
        # Warm-up: lần infer đầu luôn chậm bất thường, đừng để nó làm bẩn telemetry.
        dummy = np.zeros((self.s.imgsz, self.s.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(
                dummy, device=self.s.device, quantize=(16 if self.s.half else 32), imgsz=self.s.imgsz, verbose=False
            )
        except Exception:
            log.warning("warm-up thất bại trên %s, thử CPU", self.s.device)
            self.s.device = "cpu"
            self.s.half = False

    def infer(self, image: np.ndarray) -> InferResult:
        t0 = time.perf_counter()
        res = self.model.predict(
            image,
            device=self.s.device,
            quantize=(16 if self.s.half else 32),
            imgsz=self.s.imgsz,
            conf=self.s.conf_threshold,
            iou=self.s.iou_threshold,
            classes=self.s.class_id_list(),
            verbose=False,
        )[0]
        ms = (time.perf_counter() - t0) * 1000

        boxes: list[BBox] = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, clss):
                boxes.append(
                    BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), conf=float(c), cls=int(k))
                )
        out = InferResult(boxes=boxes, inference_ms=ms)
        out.raw = res
        return out

    def track(self, image: np.ndarray, persist: bool = True) -> InferResult:
        """Infer + tracking trong một lượt (ultralytics gọi ByteTrack/BoT-SORT bên trong)."""
        tracker_cfg = f"{self.s.tracker_type}.yaml"
        t0 = time.perf_counter()
        res = self.model.track(
            image,
            device=self.s.device,
            quantize=(16 if self.s.half else 32),
            imgsz=self.s.imgsz,
            conf=self.s.conf_threshold,
            iou=self.s.iou_threshold,
            classes=self.s.class_id_list(),
            tracker=tracker_cfg,
            persist=persist,
            verbose=False,
        )[0]
        ms = (time.perf_counter() - t0) * 1000

        boxes: list[BBox] = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            ids = (
                res.boxes.id.cpu().numpy().astype(int)
                if getattr(res.boxes, "id", None) is not None
                else [None] * len(xyxy)
            )
            for (x1, y1, x2, y2), c, k, tid in zip(xyxy, conf, clss, ids):
                boxes.append(
                    BBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        conf=float(c),
                        cls=int(k),
                        track_id=int(tid) if tid is not None else None,
                    )
                )
        out = InferResult(boxes=boxes, inference_ms=ms)
        out.raw = res
        return out
