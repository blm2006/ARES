"""Sampling Router — ba nguồn song song có quota (spec 2.1).

Ba quy tắc, theo thứ tự áp dụng:
  1. Backpressure: mức DEGRADED loại nguồn A, mức STOPPED loại tất cả.
  2. Quota ngày theo source (A 40% / B 40% / C 20%) — không nguồn nào ăn hết
     băng thông review.
  3. Cap theo GIỜ và theo CAMERA — nếu không, 80% dataset sẽ là giờ cao điểm
     của camera ồn nhất.

Quota lưu trong DB (bảng sample_budget), không giữ trong RAM: restart không được
làm reset hạn mức, nếu không thì cứ crash một cái là lấy mẫu gấp đôi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from common.config import Settings, get_settings
from common.db import SampleBudget, session_scope
from common.schemas import Source

log = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    accept: bool
    source: Source | None = None
    reason: str = ""


def _get_or_create(s, day: str, source: str, cam_id: str, hour: int) -> SampleBudget:
    row = s.execute(
        select(SampleBudget).where(
            SampleBudget.day == day,
            SampleBudget.source == source,
            SampleBudget.cam_id == cam_id,
            SampleBudget.hour == hour,
        )
    ).scalar_one_or_none()
    if row is None:
        row = SampleBudget(day=day, source=source, cam_id=cam_id, hour=hour, used=0)
        s.add(row)
        try:
            s.flush()
        except IntegrityError:
            s.rollback()
            row = s.execute(
                select(SampleBudget).where(
                    SampleBudget.day == day,
                    SampleBudget.source == source,
                    SampleBudget.cam_id == cam_id,
                    SampleBudget.hour == hour,
                )
            ).scalar_one()
    return row


class QuotaManager:
    """Ba tầng hạn mức: tổng/source/ngày, theo giờ, theo camera."""

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    def _caps(self, source: Source) -> tuple[int, int, int]:
        daily = self.s.quota_for(source.value)
        per_hour = max(1, int(self.s.daily_sample_budget * self.s.per_hour_cap_ratio))
        per_cam = max(1, int(daily * self.s.per_camera_cap_ratio))
        return daily, per_hour, per_cam

    def check(self, source: Source, cam_id: str, ts: datetime) -> RouteDecision:
        day = ts.strftime("%Y-%m-%d")
        hour = ts.hour
        daily_cap, hour_cap, cam_cap = self._caps(source)

        with session_scope() as s:
            total = _get_or_create(s, day, source.value, "*", -1)
            if total.used >= daily_cap:
                return RouteDecision(False, source, f"hết quota ngày cho {source} ({daily_cap})")

            per_cam = _get_or_create(s, day, source.value, cam_id, -1)
            if per_cam.used >= cam_cap:
                return RouteDecision(False, source, f"camera {cam_id} đã chiếm {cam_cap} mẫu")

            per_hour = _get_or_create(s, day, source.value, "*", hour)
            if per_hour.used >= hour_cap:
                return RouteDecision(False, source, f"giờ {hour:02d} đã chiếm {hour_cap} mẫu")

        return RouteDecision(True, source)

    def commit(self, source: Source, cam_id: str, ts: datetime, n: int = 1) -> None:
        day = ts.strftime("%Y-%m-%d")
        with session_scope() as s:
            for cam, hour in ((("*"), -1), (cam_id, -1), ("*", ts.hour)):
                row = _get_or_create(s, day, source.value, cam, hour)
                row.used += n

    def usage(self, day: str | None = None) -> dict[str, dict]:
        day = day or datetime.now().strftime("%Y-%m-%d")
        out: dict[str, dict] = {}
        with session_scope() as s:
            rows = s.execute(
                select(SampleBudget).where(
                    SampleBudget.day == day, SampleBudget.cam_id == "*", SampleBudget.hour == -1
                )
            ).scalars()
            for r in rows:
                out[r.source] = {
                    "used": r.used,
                    "cap": self.s.quota_for(r.source),
                }
        return out


class SamplingRouter:
    """Cổng duy nhất quyết định một frame có được ghi xuống hay không."""

    def __init__(self, guard, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.guard = guard
        self.quota = QuotaManager(self.s)
        self.rejected: dict[str, int] = {}

    def route(self, source: Source, cam_id: str, ts: datetime) -> RouteDecision:
        if not self.guard.allows(source):
            st = self.guard.state()
            return self._reject(source, f"backpressure {st.level}: {st.reason}")

        d = self.quota.check(source, cam_id, ts)
        if not d.accept:
            return self._reject(source, d.reason)
        return d

    def accept(self, source: Source, cam_id: str, ts: datetime) -> None:
        self.quota.commit(source, cam_id, ts)

    def _reject(self, source: Source, reason: str) -> RouteDecision:
        key = f"{source}:{reason.split(':')[0]}"
        self.rejected[key] = self.rejected.get(key, 0) + 1
        return RouteDecision(False, source, reason)
