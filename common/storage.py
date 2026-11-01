"""Storage abstraction. Ảnh KHÔNG đi qua queue, chỉ path đi qua.

Layout: cam/YYYY-MM-DD/HH/uuid.jpg  (spec 0.1)
Ảnh ghi xuống là FULL FRAME GỐC — không resize / crop / normalize / denoise (spec #3).
"""

from __future__ import annotations

import shutil
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from common.config import Settings, get_settings


def frame_relpath(cam_id: str, ts: datetime, ext: str, frame_uuid: str | None = None) -> str:
    fid = frame_uuid or uuid.uuid4().hex
    return f"{cam_id}/{ts:%Y-%m-%d}/{ts:%H}/{fid}.{ext}"


class StorageBackend(ABC):
    @abstractmethod
    def write_image(self, relpath: str, image: np.ndarray) -> str: ...

    @abstractmethod
    def read_image(self, relpath: str) -> np.ndarray: ...

    @abstractmethod
    def exists(self, relpath: str) -> bool: ...

    @abstractmethod
    def delete(self, relpath: str) -> None: ...

    @abstractmethod
    def free_pct(self) -> float: ...

    @abstractmethod
    def local_path(self, relpath: str) -> Path | None:
        """Đường dẫn local nếu có (dataloader cần), None nếu backend từ xa."""


class LocalStorage(StorageBackend):
    def __init__(self, root: Path, jpeg_quality: int = 95, image_format: str = "jpg"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = jpeg_quality
        self.image_format = image_format

    def _abs(self, relpath: str) -> Path:
        p = (self.root / relpath).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"path traversal: {relpath}")
        return p

    def write_image(self, relpath: str, image: np.ndarray) -> str:
        p = self._abs(relpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self.image_format == "webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, 101]  # 101 = lossless
        else:
            params = [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)]
        ok = cv2.imwrite(str(p), image, params)
        if not ok:
            raise IOError(f"cv2.imwrite failed: {p}")
        return relpath

    def read_image(self, relpath: str) -> np.ndarray:
        img = cv2.imread(str(self._abs(relpath)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(relpath)
        return img

    def exists(self, relpath: str) -> bool:
        return self._abs(relpath).exists()

    def delete(self, relpath: str) -> None:
        p = self._abs(relpath)
        if p.exists():
            p.unlink()

    def free_pct(self) -> float:
        usage = shutil.disk_usage(self.root)
        return usage.free / usage.total * 100.0

    def local_path(self, relpath: str) -> Path | None:
        return self._abs(relpath)


class MinioStorage(StorageBackend):
    """Stub có interface đầy đủ — bật khi STORAGE_BACKEND=minio.

    Cố ý chưa implement: spec nói 'interface hoá để sau đổi sang MinIO được',
    không nói phải chạy MinIO ngay. Không giả vờ hoạt động.
    """

    def __init__(self, endpoint: str, bucket: str):
        raise NotImplementedError(
            "MinioStorage chưa implement. Dùng STORAGE_BACKEND=local, "
            "hoặc implement bằng minio-py rồi bỏ dòng này."
        )


_storage: StorageBackend | None = None


def get_storage(settings: Settings | None = None) -> StorageBackend:
    global _storage
    if _storage is None:
        s = settings or get_settings()
        if s.storage_backend == "minio":
            _storage = MinioStorage(s.minio_endpoint, s.minio_bucket)
        else:
            _storage = LocalStorage(s.images_dir, s.jpeg_quality, s.image_format)
    return _storage


def reset_storage() -> None:
    """Chỉ dùng trong test."""
    global _storage
    _storage = None
