"""Lịch phòng (Room Schedule). Nguồn tĩnh từ schedule.json.

`ScheduleProvider` là giao diện tối thiểu; muốn nối Google Calendar / hệ thống xếp lịch của
trường thì viết một lớp có cùng 3 phương thức này, không phải sửa Control Engine.

Quy ước thứ: 2 = Thứ Hai ... 7 = Thứ Bảy, 8 = Chủ nhật (giống bảng habit_profiles).
"""

from __future__ import annotations

import json
import logging
import time as _time
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Session:
    key: str  # ổn định, dùng để nhớ "buổi này đã lập kế hoạch/kiểm tra chưa"
    name: str
    start: datetime
    end: datetime


class ScheduleProvider(Protocol):
    def current(self, now: datetime) -> Session | None: ...
    def next_within(self, now: datetime, minutes: float) -> Session | None: ...
    def sessions_on(self, day: date) -> list[Session]: ...


@dataclass(frozen=True)
class _Rule:
    name: str
    days: tuple[int, ...]
    start: time
    end: time


def _parse_hm(s: str) -> time:
    h, _, m = s.partition(":")
    return time(int(h), int(m))


class StaticSchedule:
    def __init__(self, rules: list[_Rule], cancelled: set[date], tz: tzinfo, path: Path | None = None):
        self.rules = rules
        self.cancelled = cancelled
        self.tz = tz
        self.path = path
        self._mtime = path.stat().st_mtime if path and path.exists() else 0.0
        self._last_check = _time.monotonic()

    # ---------------------------------------------------------------- nạp
    @classmethod
    def from_dict(cls, raw: dict, tz: tzinfo, path: Path | None = None) -> "StaticSchedule":
        rules = []
        for r in raw.get("sessions", []):
            start, end = _parse_hm(r["start"]), _parse_hm(r["end"])
            if end <= start:
                raise ValueError(f"Lịch '{r.get('name')}': giờ kết thúc phải sau giờ bắt đầu")
            days = tuple(int(d) for d in r.get("days", [2, 3, 4, 5, 6]))
            if any(d < 2 or d > 8 for d in days):
                raise ValueError(f"Lịch '{r.get('name')}': thứ phải trong 2..8 (2 = Thứ Hai, 8 = Chủ nhật)")
            rules.append(_Rule(str(r.get("name", "Lớp")), days, start, end))
        cancelled = {date.fromisoformat(e["date"]) for e in raw.get("exceptions", []) if e.get("cancelled", True)}
        return cls(rules, cancelled, tz, path)

    @classmethod
    def from_file(cls, path: Path | str | None, tz: tzinfo) -> "StaticSchedule":
        if not path or not Path(path).exists():
            log.warning("Không có file lịch (%s) — hệ thống chạy chế độ chỉ-theo-hiện-diện.", path)
            return cls([], set(), tz, Path(path) if path else None)
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")), tz, p)

    def maybe_reload(self, every_s: float = 30.0) -> bool:
        """Sửa schedule.json khi đang chạy thì tự nạp lại, không cần khởi động lại."""
        if not self.path or _time.monotonic() - self._last_check < every_s:
            return False
        self._last_check = _time.monotonic()
        try:
            m = self.path.stat().st_mtime
            if m == self._mtime:
                return False
            new = StaticSchedule.from_dict(json.loads(self.path.read_text(encoding="utf-8")), self.tz, self.path)
        except Exception as e:  # file đang sửa dở: giữ lịch cũ
            log.warning("Không nạp lại được lịch: %s", e)
            return False
        self.rules, self.cancelled, self._mtime = new.rules, new.cancelled, m
        log.info("Đã nạp lại lịch từ %s", self.path)
        return True

    # ---------------------------------------------------------------- truy vấn
    def sessions_on(self, day: date) -> list[Session]:
        if day in self.cancelled:
            return []
        dow = day.isoweekday() + 1
        out = []
        for r in self.rules:
            if dow in r.days:
                s = datetime.combine(day, r.start, tzinfo=self.tz)
                e = datetime.combine(day, r.end, tzinfo=self.tz)
                out.append(Session(f"{day.isoformat()}_{r.start:%H%M}", r.name, s, e))
        return sorted(out, key=lambda x: x.start)

    def current(self, now: datetime) -> Session | None:
        for s in self.sessions_on(now.date()):
            if s.start <= now < s.end:
                return s
        return None

    def next_within(self, now: datetime, minutes: float) -> Session | None:
        horizon = now + timedelta(minutes=minutes)
        for day in (now.date(), now.date() + timedelta(days=1)):
            for s in self.sessions_on(day):
                if now < s.start <= horizon:
                    return s
        return None

    def today(self, now: datetime) -> list[dict]:
        return [
            {"key": s.key, "name": s.name, "start": s.start.isoformat(), "end": s.end.isoformat()}
            for s in self.sessions_on(now.date())
        ]
