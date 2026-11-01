"""Nguồn A — uncertainty sampling THEO PHÂN VỊ, không theo ngưỡng cố định.

Vì sao không dùng band 0.3–0.6 cứng (v1): confidence của YOLO không được
calibrate, và sau mỗi lần fine-tune phân phối confidence dịch đi. Ngưỡng cứng sẽ
dần bắt sai nhóm mà không có tín hiệu nào báo. Phân vị trên cửa sổ trượt 24h thì
tự co giãn theo model hiện tại.
"""

from __future__ import annotations

import bisect
import time
from collections import deque
from dataclasses import dataclass, field

from common.schemas import BBox


def frame_uncertainty(boxes: list[BBox]) -> float:
    """Điểm bất định của 1 frame ∈ [0,1].

    - Mỗi box: 1 - |2*conf - 1| -> cực đại tại conf=0.5, bằng 0 tại 0 và 1.
    - Frame lấy MAX thay vì mean: một người mơ hồ giữa đám đông rõ ràng vẫn đáng học,
      lấy mean sẽ bị pha loãng.
    - Frame không có detection nào: 0.0. Trường hợp "bỏ sót người" thuộc về
      nguồn B (failure mining), không phải nguồn này — cố nhét vào đây là sai chỗ.
    """
    if not boxes:
        return 0.0
    return max(1.0 - abs(2.0 * b.conf - 1.0) for b in boxes)


@dataclass
class SlidingPercentile:
    """Phân vị trên cửa sổ trượt thời gian.

    Giữ song song: deque theo thứ tự thời gian (để hết hạn) và list đã sắp xếp
    (để tra phân vị O(log n)). Bộ nhớ ~ số frame trong cửa sổ, nên có cap cứng.
    """

    window_s: float = 24 * 3600
    max_points: int = 200_000
    _dq: deque = field(default_factory=deque)  # (ts, score)
    _sorted: list[float] = field(default_factory=list)

    def add(self, score: float, ts: float | None = None) -> None:
        ts = ts if ts is not None else time.time()
        self._dq.append((ts, score))
        bisect.insort(self._sorted, score)
        self._evict(ts)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._dq and (self._dq[0][0] < cutoff or len(self._dq) > self.max_points):
            _, old = self._dq.popleft()
            i = bisect.bisect_left(self._sorted, old)
            if i < len(self._sorted) and self._sorted[i] == old:
                self._sorted.pop(i)

    def __len__(self) -> int:
        return len(self._sorted)

    def threshold_for_top(self, top_percent: float) -> float | None:
        """Ngưỡng để lấy top `top_percent`% cao nhất. None nếu chưa đủ mẫu."""
        n = len(self._sorted)
        if n == 0:
            return None
        k = max(0, min(n - 1, int(n * (1.0 - top_percent / 100.0))))
        return self._sorted[k]


class UncertaintySampler:
    def __init__(self, top_percent: float = 5.0, window_hours: int = 24, min_samples: int = 500):
        self.top_percent = top_percent
        self.min_samples = min_samples
        self.window = SlidingPercentile(window_s=window_hours * 3600)
        self.last_threshold: float | None = None

    def observe(self, boxes: list[BBox], ts: float | None = None) -> float:
        """Mọi frame đều được quan sát để nuôi phân phối, kể cả frame không lấy mẫu."""
        score = frame_uncertainty(boxes)
        self.window.add(score, ts)
        return score

    def should_sample(self, score: float) -> bool:
        if len(self.window) < self.min_samples:
            # Chưa đủ dữ liệu để nói "top 5%" nghĩa là gì. Im lặng còn hơn đoán bừa.
            return False
        thr = self.window.threshold_for_top(self.top_percent)
        self.last_threshold = thr
        return thr is not None and score >= thr and score > 0.0

    def stats(self) -> dict:
        return {
            "window_size": len(self.window),
            "threshold": self.last_threshold,
            "ready": len(self.window) >= self.min_samples,
        }
