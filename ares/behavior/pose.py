"""YOLO-pose (Ultralytics, pretrained COCO-17 keypoint) chạy theo lô cho nhiều camera."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def resolve_device(dev: str) -> str:
    dev = (dev or "auto").strip()
    if dev != "auto":
        return dev
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


class PoseEstimator:
    def __init__(self, weights: str | Path, device: str = "auto", imgsz: int = 640, conf: float = 0.35, half: bool = True,
                 max_det: int = 1000):
        from ultralytics import YOLO

        self.device = resolve_device(device)
        self.model = YOLO(str(weights))
        if getattr(self.model, "task", "pose") != "pose":
            raise ValueError(f"{weights} không phải model pose (task={self.model.task}). Dùng yolo11n-pose.pt / yolo11s-pose.pt.")
        self.imgsz, self.conf, self.max_det = imgsz, conf, max(1, int(max_det))  # không giới hạn số người trong 1 khung
        self.half = bool(half) and self.device.startswith("cuda")
        self.weights = str(weights)
        self.fallback_reason = ""
        if self.device != "cpu":  # chạy thử 1 ảnh: GPU không tương thích bản PyTorch -> chuyển CPU ngay, không lỗi lặp lại
            try:
                self.infer([np.zeros((64, 64, 3), np.uint8)])
            except RuntimeError as e:
                self._to_cpu(e)

    def _to_cpu(self, e: Exception) -> None:
        self.fallback_reason = str(e).splitlines()[0][:160]
        log.warning("YOLO-pose lỗi trên %s (%s) — chuyển sang CPU. Sửa: chạy NANG_CAP_GPU.bat", self.device,
                    self.fallback_reason)
        self.device, self.half = "cpu", False

    def infer(self, images: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """-> mỗi ảnh: (kps (n,17,3) pixel, boxes (n,4) xyxy, confs (n,))."""
        kw = {"half": True} if self.half else {}  # chỉ truyền khi bật (bản Ultralytics mới đổi tên tham số này)
        try:
            res = self.model.predict(images, imgsz=self.imgsz, conf=self.conf, device=self.device, classes=[0],
                                     max_det=getattr(self, "max_det", 1000), verbose=False, **kw)
        except RuntimeError as e:
            if self.device == "cpu" or "CUDA" not in str(e):
                raise
            self._to_cpu(e)
            res = self.model.predict(images, imgsz=self.imgsz, conf=self.conf, device="cpu", classes=[0],
                                     max_det=getattr(self, "max_det", 1000), verbose=False)
        out = []
        for r in res:
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                out.append((np.zeros((0, 17, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)))
                continue
            kp = r.keypoints.data.cpu().numpy().astype(np.float32)
            if kp.shape[-1] == 2:  # model không trả conf từng khớp
                kp = np.concatenate([kp, np.ones(kp.shape[:-1] + (1,), np.float32)], axis=-1)
            out.append((kp, r.boxes.xyxy.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32)))
        return out
