"""Embedding từ YOLO backbone. CHẠY TRONG WORKER, không bao giờ trong luồng realtime.

Có hai implementation:
  - YoloEmbedder: dùng backbone của model hiện tại (ultralytics .embed()).
  - HistogramEmbedder: descriptor rẻ, không cần torch. Dùng cho test và cho máy
    không có GPU rảnh. Kém hơn nhưng vẫn đủ để dedup gần-trùng.

Vector THÔ phải ổn định theo thời gian, vì PCA đã freeze bám vào phân phối của nó.
Đổi embedder = đổi PCA version = reindex lại toàn bộ. Đừng đổi tuỳ hứng.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

log = logging.getLogger(__name__)


class Embedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """(N, dim) float32."""


class HistogramEmbedder(Embedder):
    """Grid color histogram + gradient. 4x4 lưới × (3×8 bins màu + 8 bins hướng) = 512-D."""

    name = "hist_v1"
    dim = 512

    def __init__(self, grid: int = 4, color_bins: int = 8, grad_bins: int = 8):
        self.grid, self.color_bins, self.grad_bins = grid, color_bins, grad_bins
        self.dim = grid * grid * (3 * color_bins + grad_bins)

    def _one(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)  # [0,1)

        step = 128 // self.grid
        feats: list[np.ndarray] = []
        for i in range(self.grid):
            for j in range(self.grid):
                cell = img[i * step:(i + 1) * step, j * step:(j + 1) * step]
                for c in range(3):
                    h = cv2.calcHist([cell], [c], None, [self.color_bins], [0, 256]).ravel()
                    feats.append(h)
                a = ang[i * step:(i + 1) * step, j * step:(j + 1) * step].ravel()
                m = mag[i * step:(i + 1) * step, j * step:(j + 1) * step].ravel()
                hg, _ = np.histogram(a, bins=self.grad_bins, range=(0, 1), weights=m)
                feats.append(hg.astype(np.float32))
        v = np.concatenate(feats).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._one(im) for im in images]).astype(np.float32)


class YoloEmbedder(Embedder):
    name = "yolo_backbone_v1"

    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 640):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        probe = self.embed([np.zeros((64, 64, 3), dtype=np.uint8)])
        self.dim = int(probe.shape[1])
        log.info("YoloEmbedder dim=%d", self.dim)

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        out = self.model.embed(images, device=self.device, imgsz=self.imgsz, verbose=False)
        vecs = np.stack([v.cpu().numpy().ravel() for v in out]).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


def build_embedder(kind: str, weights: str = "", device: str = "cpu", imgsz: int = 640) -> Embedder:
    if kind == "yolo":
        try:
            return YoloEmbedder(weights, device, imgsz)
        except Exception:
            log.exception("YoloEmbedder lỗi, quay về HistogramEmbedder")
    return HistogramEmbedder()
