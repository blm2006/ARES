"""YOLO detector cho ARES (độc lập với realtime/detector.py cũ vì cần ngưỡng conf riêng).

Khác bản cũ:
  * Chạy conf thấp (0.35) để các detection nằm trong dải bất định 0.35-0.55 còn xuất hiện cho Active Learning.
  * Tự nhận phiên bản ultralytics: bản >= 8.4 dùng `quantize=16`, bản cũ dùng `half=True`.
  * Chỉ giữ lớp "person".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ares.config import AresSettings

log = logging.getLogger(__name__)


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int = 0


class Detector(Protocol):
    version: str

    def detect(self, image: np.ndarray) -> tuple[list[Det], float]:
        """Trả (danh sách detection, thời gian suy luận ms)."""
        ...


class YoloDetector:
    def __init__(self, cfg: AresSettings, weights: Path | str, version: str = "base"):
        from ultralytics import YOLO

        self.cfg = cfg
        self.version = version
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.device = self._pick_device(cfg.device)
        self._precision = self._precision_kwargs(cfg.half and self.device.startswith("cuda"))
        log.info("YOLO nạp %s (%s), device=%s, precision=%s", self.weights, version, self.device, self._precision or "fp32")
        self._warmup()

    @staticmethod
    def _pick_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def _precision_kwargs(want_half: bool) -> dict:
        if not want_half:
            return {}
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT

            return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
        except Exception:
            return {"half": True}

    def _predict(self, image: np.ndarray):
        return self.model.predict(
            image, device=self.device, imgsz=self.cfg.imgsz, conf=self.cfg.detect_conf, iou=self.cfg.iou,
            classes=[self.cfg.person_class_id], verbose=False, **self._precision,
        )[0]

    def _warmup(self) -> None:
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        try:
            self._predict(dummy)
        except Exception as e:
            if self.device != "cpu":
                log.warning("Warm-up lỗi trên %s (%s) — chuyển sang CPU", self.device, e)
                self.device, self._precision = "cpu", {}
                self._predict(dummy)
            else:
                raise

    def detect(self, image: np.ndarray) -> tuple[list[Det], float]:
        t0 = time.perf_counter()
        r = self._predict(image)
        ms = (time.perf_counter() - t0) * 1000.0
        out: list[Det] = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                out.append(Det(float(x1), float(y1), float(x2), float(y2), float(c), int(k)))
        return out, ms
