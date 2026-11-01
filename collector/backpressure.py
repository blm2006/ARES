"""Backpressure — điều kiện để lời hứa "không ảnh hưởng realtime" là thật.

Ba mức:
  NORMAL      : nhận cả A, B, C
  DEGRADED    : queue depth > N1 -> chỉ nhận B (failure) và C (random)
  STOPPED     : queue depth > N2, hoặc disk free < ngưỡng -> không nhận gì

Mọi ngưỡng lấy từ config. Kiểm tra disk có cache vì statvfs không rẻ.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum

from common.config import Settings, get_settings
from common.queue import QueueBackend
from common.schemas import Source
from common.storage import StorageBackend

log = logging.getLogger(__name__)


class Level(StrEnum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    STOPPED = "stopped"


ALLOWED = {
    Level.NORMAL: {Source.UNCERTAINTY, Source.FAILURE_MINING, Source.RANDOM},
    Level.DEGRADED: {Source.FAILURE_MINING, Source.RANDOM},
    Level.STOPPED: set(),
}


@dataclass
class BackpressureState:
    level: Level
    queue_depth: int
    disk_free_pct: float
    reason: str = ""


class BackpressureGuard:
    def __init__(
        self,
        queue: QueueBackend,
        storage: StorageBackend,
        settings: Settings | None = None,
        check_interval_s: float = 1.0,
    ):
        self.q = queue
        self.storage = storage
        self.s = settings or get_settings()
        self.check_interval_s = check_interval_s
        self._last_check = 0.0
        self._state = BackpressureState(Level.NORMAL, 0, 100.0)
        self._last_logged: Level | None = None

    def state(self, force: bool = False) -> BackpressureState:
        now = time.monotonic()
        if not force and now - self._last_check < self.check_interval_s:
            return self._state
        self._last_check = now

        depth = self.q.depth()
        free = self.storage.free_pct()

        if free < self.s.disk_free_min_pct:
            st = BackpressureState(Level.STOPPED, depth, free, f"disk free {free:.1f}% < {self.s.disk_free_min_pct}%")
        elif depth > self.s.queue_depth_n2:
            st = BackpressureState(Level.STOPPED, depth, free, f"queue depth {depth} > N2={self.s.queue_depth_n2}")
        elif depth > self.s.queue_depth_n1:
            st = BackpressureState(Level.DEGRADED, depth, free, f"queue depth {depth} > N1={self.s.queue_depth_n1}")
        else:
            st = BackpressureState(Level.NORMAL, depth, free)

        if st.level is not self._last_logged:
            level_log = log.warning if st.level is not Level.NORMAL else log.info
            level_log("backpressure -> %s (%s)", st.level, st.reason or "ok")
            self._last_logged = st.level
        self._state = st
        return st

    def allows(self, source: Source) -> bool:
        return source in ALLOWED[self.state().level]

    def can_write(self) -> bool:
        return self.state().level is not Level.STOPPED
