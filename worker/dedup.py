"""Dedup bằng faiss trên TOÀN BỘ dataset đang có — không chỉ batch hiện tại.

Ba điểm sửa so với v1:
  1. Index phủ toàn bộ dataset. Index theo batch thì một cảnh lặp lại mỗi ngày
     sẽ vào dataset mỗi ngày một lần.
  2. Cosine > theta -> discard NHƯNG TĂNG COUNTER. Counter tăng nhanh nghĩa là
     dataset đang bão hoà — tín hiệu nên nới tiêu chí hoặc thêm camera, chứ không
     phải cứ chạy tiếp và tự hỏi sao mãi không khá lên.
  3. Nguồn B (failure mining) dùng theta LỎNG HƠN: lỗi thì hay lặp lại, nhưng
     vẫn cần học.

Vector đã chuẩn hoá nên inner product == cosine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.db import DedupStat, session_scope
from common.schemas import Source

log = logging.getLogger(__name__)


@dataclass
class DedupResult:
    duplicate: bool
    similarity: float
    neighbor_id: str | None
    theta: float


class FaissStore:
    """Index + ánh xạ vị trí -> frame_id. Ghi xuống đĩa để restart không mất.

    Không có faiss thì tự động rơi về brute-force numpy: chậm hơn nhưng đúng,
    và dưới ~200k vector 128-D thì vẫn chấp nhận được.
    """

    def __init__(self, path: Path, dim: int, pca_version: str):
        self.path = Path(path)
        self.map_path = self.path.with_suffix(".ids.json")
        self.dim = dim
        self.pca_version = pca_version
        self.ids: list[str] = []
        self._index = None
        self._matrix: np.ndarray | None = None  # fallback
        self._load()

    # ---------- backend ----------
    @property
    def use_faiss(self) -> bool:
        return self._index is not None

    def _new_index(self):
        try:
            import faiss

            return faiss.IndexFlatIP(self.dim)
        except ImportError:
            log.warning("faiss chưa cài — dùng brute-force numpy")
            return None

    def _load(self) -> None:
        meta_ok = self.map_path.exists()
        if meta_ok:
            meta = json.loads(self.map_path.read_text(encoding="utf-8"))
            if meta.get("pca_version") != self.pca_version:
                raise RuntimeError(
                    f"Index dựng bằng {meta.get('pca_version')} nhưng config là "
                    f"{self.pca_version}. Phải reindex, không được trộn."
                )
            self.ids = meta["ids"]

        self._index = self._new_index()
        if self._index is not None and self.path.exists():
            import faiss

            self._index = faiss.read_index(str(self.path))
        elif self._index is None:
            npy = self.path.with_suffix(".npy")
            self._matrix = (
                np.load(npy).astype(np.float32)
                if npy.exists()
                else np.zeros((0, self.dim), dtype=np.float32)
            )

    # ---------- thao tác ----------
    def add(self, vec: np.ndarray, frame_id: str) -> None:
        v = np.atleast_2d(vec).astype(np.float32)
        if self._index is not None:
            self._index.add(v)
        else:
            self._matrix = np.vstack([self._matrix, v]) if self._matrix.size else v
        self.ids.append(frame_id)

    def search(self, vec: np.ndarray, k: int = 1) -> tuple[float, str | None]:
        if not self.ids:
            return 0.0, None
        v = np.atleast_2d(vec).astype(np.float32)
        if self._index is not None:
            sims, idx = self._index.search(v, min(k, len(self.ids)))
            return float(sims[0][0]), self.ids[int(idx[0][0])]
        sims = (self._matrix @ v[0]).ravel()
        i = int(np.argmax(sims))
        return float(sims[i]), self.ids[i]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            import faiss

            faiss.write_index(self._index, str(self.path))
        else:
            np.save(self.path.with_suffix(".npy"), self._matrix)
        self.map_path.write_text(
            json.dumps({"pca_version": self.pca_version, "dim": self.dim, "ids": self.ids}),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self.ids)


class Deduplicator:
    def __init__(self, store: FaissStore, theta: float, theta_failure: float):
        self.store = store
        self.theta = theta
        self.theta_failure = theta_failure

    def theta_for(self, source: Source | str) -> float:
        src = source.value if isinstance(source, Source) else source
        return self.theta_failure if src == Source.FAILURE_MINING.value else self.theta

    def check(self, vec: np.ndarray, source: Source | str) -> DedupResult:
        theta = self.theta_for(source)
        sim, neighbor = self.store.search(vec)
        return DedupResult(duplicate=sim > theta, similarity=sim, neighbor_id=neighbor, theta=theta)

    def add(self, vec: np.ndarray, frame_id: str) -> None:
        self.store.add(vec, frame_id)


def bump_stat(day: str, source: str, hit: bool) -> None:
    with session_scope() as s:
        row = (
            s.query(DedupStat)
            .filter(DedupStat.day == day, DedupStat.source == source)
            .one_or_none()
        )
        if row is None:
            row = DedupStat(day=day, source=source, hits=0, passed=0)
            s.add(row)
        if hit:
            row.hits += 1
        else:
            row.passed += 1


def saturation_report(days: int = 7) -> list[dict]:
    """Tỉ lệ discard theo ngày. Tăng dần = dataset bão hoà."""
    with session_scope() as s:
        rows = s.query(DedupStat).order_by(DedupStat.day.desc()).limit(days * 3).all()
    out = []
    for r in rows:
        total = r.hits + r.passed
        out.append(
            {
                "day": r.day,
                "source": r.source,
                "hits": r.hits,
                "passed": r.passed,
                "discard_rate": round(r.hits / total, 3) if total else 0.0,
            }
        )
    return out
