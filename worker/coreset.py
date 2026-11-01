"""Core-set selection — đây chính là "khung hình đại diện" của v1, đặt ĐÚNG CHỖ.

v1 để tiêu chí này ở 2.1 (sampling router), nhưng nó cần embedding mà embedding
chỉ tính ở 2.3 -> phụ thuộc vòng. Ở đây, sau PCA và sau dedup, nó chạy được.

Thuật toán: greedy k-center (farthest-point). Chọn tập phủ không gian đặc trưng
rộng nhất trong số frame còn sống sót.
"""

from __future__ import annotations

import numpy as np


def greedy_kcenter(vectors: np.ndarray, k: int, seed_vectors: np.ndarray | None = None) -> list[int]:
    """Trả về chỉ số của k vector phủ rộng nhất.

    seed_vectors: các vector ĐÃ có trong dataset — điểm mới nên xa chúng, không chỉ
    xa nhau. Bỏ qua tham số này là lý do thường gặp khiến core-set chọn lại đúng
    vùng đã dày.
    """
    n = len(vectors)
    if k >= n:
        return list(range(n))
    if n == 0 or k <= 0:
        return []

    if seed_vectors is not None and len(seed_vectors):
        dist = 1.0 - (vectors @ seed_vectors.T).max(axis=1)
    else:
        dist = np.full(n, np.inf)

    chosen: list[int] = []
    for _ in range(k):
        i = int(np.argmax(dist))
        chosen.append(i)
        d_new = 1.0 - (vectors @ vectors[i])
        dist = np.minimum(dist, d_new)
        dist[i] = -np.inf
    return chosen


def select_coreset(
    vectors: np.ndarray, keep_ratio: float = 0.7, seed_vectors: np.ndarray | None = None
) -> list[int]:
    k = max(1, int(round(len(vectors) * keep_ratio)))
    return sorted(greedy_kcenter(vectors, k, seed_vectors))
