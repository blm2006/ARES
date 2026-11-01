"""PCA — FIT MỘT LẦN RỒI FREEZE (spec #6).

Vì sao không refit định kỳ: vector đã lưu ở version cũ không còn so sánh được với
vector mới, và faiss index thì không biết điều đó. Kết quả là dedup sai toàn bộ
mà không có lỗi nào nổ ra — dạng hỏng tệ nhất.

Muốn đổi PCA: bump version, fit lại, reindex TẤT CẢ. `refit()` cố tình bắt phải
truyền version mới và cố tình không ghi đè file cũ.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class PCAFrozenError(RuntimeError):
    pass


@dataclass
class FrozenPCA:
    version: str
    embedder: str
    input_dim: int
    n_components: int
    mean: np.ndarray
    components: np.ndarray  # (n_components, input_dim)
    fitted_at: str
    n_fit_samples: int

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        if x.shape[1] != self.input_dim:
            raise ValueError(f"input_dim {x.shape[1]} != {self.input_dim} của {self.version}")
        z = (x - self.mean) @ self.components.T
        n = np.linalg.norm(z, axis=1, keepdims=True)
        return (z / np.where(n == 0, 1, n)).astype(np.float32)  # chuẩn hoá -> cosine = inner product

    def save(self, path: Path) -> Path:
        path = Path(path)
        if path.exists():
            raise PCAFrozenError(
                f"{path} đã tồn tại. PCA đã freeze — muốn đổi thì bump PCA_VERSION, "
                "fit ra file mới, rồi reindex toàn bộ dataset."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "version": self.version,
                    "embedder": self.embedder,
                    "input_dim": self.input_dim,
                    "n_components": self.n_components,
                    "fitted_at": self.fitted_at,
                    "n_fit_samples": self.n_fit_samples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("PCA %s đã freeze -> %s", self.version, path)
        return path

    @staticmethod
    def load(path: Path) -> "FrozenPCA":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, FrozenPCA):
            raise TypeError(f"{path} không phải FrozenPCA")
        return obj


def fit_pca(
    vectors: np.ndarray,
    version: str,
    embedder: str,
    n_components: int = 128,
    min_samples: int = 2000,
) -> FrozenPCA:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[0] < min_samples:
        raise ValueError(
            f"Chỉ có {vectors.shape[0]} mẫu, cần >= {min_samples}. "
            "Fit PCA trên tập quá nhỏ rồi freeze nó là tự khoá mình vào một phép "
            "chiếu sai suốt đời dự án."
        )
    n_components = min(n_components, vectors.shape[1], vectors.shape[0])
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    # SVD randomized qua sklearn nếu có, không thì numpy đầy đủ.
    try:
        from sklearn.decomposition import PCA as SkPCA

        p = SkPCA(n_components=n_components, svd_solver="randomized", random_state=0)
        p.fit(centered)
        comps = p.components_.astype(np.float32)
    except Exception:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        comps = vt[:n_components].astype(np.float32)

    return FrozenPCA(
        version=version,
        embedder=embedder,
        input_dim=int(vectors.shape[1]),
        n_components=int(n_components),
        mean=mean.astype(np.float32),
        components=comps,
        fitted_at=datetime.now(timezone.utc).isoformat(),
        n_fit_samples=int(vectors.shape[0]),
    )


def load_or_raise(path: Path, version: str) -> FrozenPCA:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Chạy `python -m scripts.fit_pca` trên tập khởi tạo trước — "
            "worker không được phép tự fit ngầm."
        )
    pca = FrozenPCA.load(path)
    if pca.version != version:
        raise PCAFrozenError(f"File là {pca.version} nhưng config yêu cầu {version}")
    return pca
