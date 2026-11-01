"""Đồng hồ trừu tượng.

Mọi logic phụ thuộc thời gian (bộ đệm trượt, lịch lớp, pre-cooling, retrain nửa đêm)
đều đọc giờ qua Clock, không gọi datetime.now() trực tiếp. Nhờ vậy:
  - chế độ mô phỏng chạy nhanh được (SimClock, speed=60 => 1 giây thật = 1 phút mô phỏng),
  - test chạy xác định, không phụ thuộc giờ máy (ManualClock).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone, tzinfo


def get_tz(name: str) -> tzinfo:
    """Lấy múi giờ. Windows không có sẵn dữ liệu IANA nếu thiếu gói `tzdata`,
    nên với giờ Việt Nam rơi về UTC+7 cố định thay vì chết."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        if name in ("Asia/Ho_Chi_Minh", "Asia/Saigon", "Asia/Bangkok", "Asia/Hanoi"):
            return timezone(timedelta(hours=7), name="ICT")
        return timezone.utc


class Clock:
    """Giao diện chung. `time()` là epoch giây (theo thời gian của đồng hồ này)."""

    tz: tzinfo
    speed: float = 1.0

    def time(self) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def now(self) -> datetime:
        """Giờ địa phương có tzinfo."""
        return datetime.fromtimestamp(self.time(), tz=self.tz)

    def naive_now(self) -> datetime:
        """Giờ địa phương không tzinfo — dạng lưu trong DB."""
        return self.now().replace(tzinfo=None)

    def real_timeout(self, sim_seconds: float) -> float:
        """Đổi số giây theo đồng hồ này sang số giây thật để chờ (Event.wait)."""
        return max(0.0, sim_seconds / max(self.speed, 1e-9))


class RealClock(Clock):
    def __init__(self, tz: tzinfo | str = "Asia/Ho_Chi_Minh"):
        self.tz = get_tz(tz) if isinstance(tz, str) else tz
        self.speed = 1.0

    def time(self) -> float:
        return time.time()


class SimClock(Clock):
    """Chạy nhanh/chậm so với giờ thật, bắt đầu từ một thời điểm tùy chọn."""

    def __init__(self, tz: tzinfo | str = "Asia/Ho_Chi_Minh", start: datetime | None = None, speed: float = 1.0):
        self.tz = get_tz(tz) if isinstance(tz, str) else tz
        self.speed = float(speed)
        start = start or datetime.now(self.tz)
        if start.tzinfo is None:
            start = start.replace(tzinfo=self.tz)
        self._t0_sim = start.timestamp()
        self._t0_real = time.monotonic()

    def time(self) -> float:
        return self._t0_sim + (time.monotonic() - self._t0_real) * self.speed


class ManualClock(Clock):
    """Đồng hồ do test tự quay. `sleep()` chỉ tua thời gian, không chờ thật."""

    def __init__(self, start: datetime, tz: tzinfo | str = "Asia/Ho_Chi_Minh"):
        self.tz = get_tz(tz) if isinstance(tz, str) else tz
        if start.tzinfo is None:
            start = start.replace(tzinfo=self.tz)
        self._t = start.timestamp()
        self.speed = 1.0

    def time(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds

    def set(self, when: datetime) -> None:
        if when.tzinfo is None:
            when = when.replace(tzinfo=self.tz)
        self._t = when.timestamp()


def vn_weekday(dt: datetime) -> int:
    """Thứ theo quy ước Việt Nam dùng trong DB: 2 = Thứ Hai ... 7 = Thứ Bảy, 8 = Chủ nhật."""
    return dt.isoweekday() + 1
