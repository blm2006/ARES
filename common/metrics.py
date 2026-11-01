"""Telemetry hạ tầng — tách hoàn toàn khỏi dataset pipeline (spec 1.7).

Lý do tồn tại: nếu drop frame làm tracker mất ID, failure mining sẽ coi đó là
lỗi model và sinh hàng nghìn mẫu rác. Phải phân biệt được hai loại lỗi.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta

from common.db import Telemetry, session_scope, utcnow


_GPU_STATE = "init"  # init | ok | off
_GPU_HANDLE = None
_GPU_CACHE: tuple[float, float] = (-1e9, 0.0)  # (monotonic, value)


def gpu_util(min_interval_s: float = 1.0) -> float:
    """% GPU, cache 1 giây.

    Trước đây hàm này gọi nvmlInit() + lấy handle MỖI LẦN, mà snapshot() thì chạy
    mỗi frame trong vòng lặp realtime — tức ~25 lần/giây chỉ để điền một con số
    hiển thị. Giờ: init đúng một lần, hỏng thì tắt hẳn, kết quả cache theo thời gian.
    """
    global _GPU_STATE, _GPU_HANDLE, _GPU_CACHE
    if _GPU_STATE == "off":
        return 0.0
    now = time.monotonic()
    if now - _GPU_CACHE[0] < min_interval_s:
        return _GPU_CACHE[1]
    try:
        import pynvml

        if _GPU_STATE == "init":
            pynvml.nvmlInit()
            _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
            _GPU_STATE = "ok"
        val = float(pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE).gpu)
    except Exception:
        _GPU_STATE = "off"  # không có NVML thì đừng thử lại mỗi giây
        val = 0.0
    _GPU_CACHE = (now, val)
    return val


@dataclass
class RollingCounter:
    window_s: float = 60.0
    _events: deque = field(default_factory=deque)

    def hit(self, n: int = 1) -> None:
        now = time.monotonic()
        for _ in range(n):
            self._events.append(now)
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    def per_min(self) -> float:
        now = time.monotonic()
        self._trim(now)
        return len(self._events) * (60.0 / self.window_s)


@dataclass
class CameraTelemetry:
    """Gom metric cho 1 camera; flush định kỳ xuống DB."""

    cam_id: str
    model_version: str = "unknown"
    role: str = "primary"
    flush_every_s: float = 10.0

    frames_read: int = 0
    frames_dropped: int = 0
    frames_processed: int = 0
    _inference_ms: deque = field(default_factory=lambda: deque(maxlen=300))
    _fps_window: deque = field(default_factory=lambda: deque(maxlen=300))
    id_switches: RollingCounter = field(default_factory=RollingCounter)
    track_lengths: deque = field(default_factory=lambda: deque(maxlen=500))
    people_count: float = 0.0
    queue_depth: int = 0
    _last_flush: float = field(default_factory=time.monotonic)
    _last_frame_t: float = field(default_factory=time.monotonic)

    def on_frame_read(self) -> None:
        self.frames_read += 1

    def on_frame_dropped(self, n: int = 1) -> None:
        self.frames_dropped += n

    def on_frame_processed(self, inference_ms: float, n_people: int) -> None:
        now = time.monotonic()
        dt = now - self._last_frame_t
        self._last_frame_t = now
        if dt > 0:
            self._fps_window.append(1.0 / dt)
        self._inference_ms.append(inference_ms)
        self.frames_processed += 1
        self.people_count = float(n_people)

    def on_id_switch(self, n: int = 1) -> None:
        self.id_switches.hit(n)

    def on_track_finished(self, length: int) -> None:
        self.track_lengths.append(length)

    @property
    def fps(self) -> float:
        return sum(self._fps_window) / len(self._fps_window) if self._fps_window else 0.0

    @property
    def inference_ms(self) -> float:
        return (
            sum(self._inference_ms) / len(self._inference_ms) if self._inference_ms else 0.0
        )

    @property
    def frame_drop_rate(self) -> float:
        total = self.frames_read + self.frames_dropped
        return self.frames_dropped / total if total else 0.0

    @property
    def avg_track_length(self) -> float:
        return sum(self.track_lengths) / len(self.track_lengths) if self.track_lengths else 0.0

    def snapshot(self) -> dict:
        return {
            "cam_id": self.cam_id,
            "model_version": self.model_version,
            "role": self.role,
            "fps": round(self.fps, 2),
            "inference_ms": round(self.inference_ms, 2),
            "frame_drop_rate": round(self.frame_drop_rate, 4),
            "queue_depth": self.queue_depth,
            "gpu_util": gpu_util(),
            "id_switch_per_min": round(self.id_switches.per_min(), 2),
            "avg_track_length": round(self.avg_track_length, 2),
            "people_count": self.people_count,
        }

    def maybe_flush(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_flush < self.flush_every_s:
            return False
        self._last_flush = now
        snap = self.snapshot()
        with session_scope() as s:
            s.add(Telemetry(**snap))
        return True


def recent_telemetry(cam_id: str, minutes: int = 60, role: str = "primary") -> list[Telemetry]:
    since = utcnow() - timedelta(minutes=minutes)
    with session_scope() as s:
        return list(
            s.query(Telemetry)
            .filter(Telemetry.cam_id == cam_id, Telemetry.ts >= since, Telemetry.role == role)
            .order_by(Telemetry.ts)
            .all()
        )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
