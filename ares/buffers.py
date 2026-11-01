"""Smoothing buffers: trung bình trượt THEO THỜI GIAN.

Hai bộ đệm của spec: cửa sổ 30 giây cho đèn, 10 phút cho điều hòa (chống bật/tắt máy nén
liên tục khi người chỉ ra ngoài một lát).

Dùng sample-and-hold có trọng số thời gian thay vì trung bình cộng các mẫu: khi YOLO chậm
lại/gián đoạn, mẫu không bị tính nhiều hơn chỉ vì đến dày hơn. Khoảng trống dài hơn
`max_gap_s` bị loại khỏi mẫu số và hiện ra ở `coverage` — để bên gọi biết "không có dữ liệu"
khác với "không có người" (camera chết không được hiểu là phòng trống).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class WindowStats:
    mean_total: float | None
    mean_zones: list[float] | None
    peak_total: int
    peak_zones: list[int]
    coverage: float  # 0..1: phần cửa sổ thực sự có dữ liệu
    n_samples: int
    last_t: float | None  # thời điểm mẫu mới nhất

    @property
    def ok(self) -> bool:
        return self.mean_total is not None


class OccupancyBuffer:
    def __init__(self, window_s: float, n_zones: int, max_gap_s: float = 20.0):
        if window_s <= 0:
            raise ValueError("window_s phải > 0")
        self.window_s = float(window_s)
        self.n_zones = n_zones
        self.max_gap_s = float(max_gap_s)
        self._s: deque[tuple[float, tuple[int, ...]]] = deque()  # (t, (total, z1..zn))
        self.last_nonzero_total: float | None = None
        self.last_nonzero_zone: list[float | None] = [None] * n_zones

    def push(self, t: float, total: int, zones: list[int]) -> None:
        if len(zones) != self.n_zones:
            raise ValueError(f"cần {self.n_zones} giá trị vùng, nhận {len(zones)}")
        if self._s and t < self._s[-1][0]:
            return  # mẫu đến muộn/đồng hồ lùi: bỏ, không làm hỏng thứ tự
        self._s.append((t, (int(total), *[int(z) for z in zones])))
        if total > 0:
            self.last_nonzero_total = t
        for i, z in enumerate(zones):
            if z > 0:
                self.last_nonzero_zone[i] = t
        self._prune(t)

    def _prune(self, now: float) -> None:
        t0 = now - self.window_s
        # giữ lại 1 mẫu ngay trước cửa sổ vì nó còn "giữ" giá trị vào trong cửa sổ
        while len(self._s) >= 2 and self._s[1][0] <= t0:
            self._s.popleft()

    def stats(self, now: float) -> WindowStats:
        self._prune(now)
        t0 = now - self.window_s
        n_ch = self.n_zones + 1
        acc = [0.0] * n_ch
        peak = [0] * n_ch
        wsum = 0.0
        n_in = 0
        items = list(self._s)
        for k, (t, v) in enumerate(items):
            t_next = items[k + 1][0] if k + 1 < len(items) else now
            seg_start = max(t, t0)
            seg_end = min(t_next, t + self.max_gap_s, now)
            d = seg_end - seg_start
            if d <= 0:
                continue
            n_in += 1
            wsum += d
            for c in range(n_ch):
                acc[c] += v[c] * d
                if v[c] > peak[c]:
                    peak[c] = v[c]
        last_t = items[-1][0] if items else None
        if wsum <= 0:
            return WindowStats(None, None, 0, [0] * self.n_zones, 0.0, 0, last_t)
        mean = [a / wsum for a in acc]
        return WindowStats(
            mean_total=mean[0],
            mean_zones=mean[1:],
            peak_total=peak[0],
            peak_zones=peak[1:],
            coverage=min(1.0, wsum / self.window_s),
            n_samples=n_in,
            last_t=last_t,
        )

    def seconds_since_nonzero(self, now: float, zone: int | None = None) -> float | None:
        """Số giây kể từ lần cuối thấy người (None nếu chưa từng thấy)."""
        t = self.last_nonzero_total if zone is None else self.last_nonzero_zone[zone]
        return None if t is None else max(0.0, now - t)
