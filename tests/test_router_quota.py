"""Quota A/B/C, cap theo giờ/camera, và ưu tiên của router."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from collector.router import QuotaManager, SamplingRouter
from common.schemas import Source


class FakeGuard:
    """Guard giả: luôn cho phép, để test riêng phần quota."""

    def __init__(self, allowed=None):
        self.allowed = allowed

    def allows(self, source):
        return self.allowed is None or source in self.allowed

    def can_write(self):
        return True

    def state(self):
        class S:
            level = "normal"
            reason = ""
            queue_depth = 0
            disk_free_pct = 99.0

        return S()


def ts(hour=9, day=1):
    return datetime(2026, 1, day, hour, 0, tzinfo=timezone.utc)


def test_daily_quota_split_40_40_20(settings):
    settings.daily_sample_budget = 1000
    q = QuotaManager(settings)
    assert settings.quota_for("uncertainty") == 400
    assert settings.quota_for("failure_mining") == 400
    assert settings.quota_for("random") == 200


def test_source_cannot_exceed_its_quota(settings):
    settings.daily_sample_budget = 50
    settings.per_hour_cap_ratio = 1.0
    settings.per_camera_cap_ratio = 1.0
    r = SamplingRouter(FakeGuard(), settings)

    cap = settings.quota_for("random")  # 20% * 50 = 10
    accepted = 0
    for _ in range(cap * 3):
        d = r.route(Source.RANDOM, "cam01", ts())
        if d.accept:
            r.accept(Source.RANDOM, "cam01", ts())
            accepted += 1
    assert accepted == cap

    # Nguồn khác vẫn còn nguyên hạn mức của mình — đây chính là điểm của quota.
    d = r.route(Source.FAILURE_MINING, "cam01", ts())
    assert d.accept


def test_per_hour_cap(settings):
    settings.daily_sample_budget = 100
    settings.per_hour_cap_ratio = 0.10  # 10 mẫu/giờ
    settings.per_camera_cap_ratio = 1.0
    r = SamplingRouter(FakeGuard(), settings)

    accepted = 0
    for _ in range(40):
        d = r.route(Source.FAILURE_MINING, "cam01", ts(hour=9))
        if d.accept:
            r.accept(Source.FAILURE_MINING, "cam01", ts(hour=9))
            accepted += 1
    assert accepted == 10

    # Sang giờ khác thì lại có hạn mức -> mẫu được rải theo giờ.
    d = r.route(Source.FAILURE_MINING, "cam01", ts(hour=10))
    assert d.accept


def test_per_camera_cap(settings):
    settings.daily_sample_budget = 100
    settings.per_hour_cap_ratio = 1.0
    settings.per_camera_cap_ratio = 0.25  # 1 camera tối đa 25% quota của source
    r = SamplingRouter(FakeGuard(), settings)

    cap_source = settings.quota_for("failure_mining")  # 40
    cap_cam = int(cap_source * 0.25)  # 10
    accepted = 0
    for _ in range(60):
        d = r.route(Source.FAILURE_MINING, "camNOISY", ts())
        if d.accept:
            r.accept(Source.FAILURE_MINING, "camNOISY", ts())
            accepted += 1
    assert accepted == cap_cam

    assert r.route(Source.FAILURE_MINING, "camQUIET", ts()).accept


def test_quota_resets_next_day(settings):
    settings.daily_sample_budget = 20
    settings.per_hour_cap_ratio = 1.0
    settings.per_camera_cap_ratio = 1.0
    r = SamplingRouter(FakeGuard(), settings)

    cap = settings.quota_for("random")
    for _ in range(cap):
        r.accept(Source.RANDOM, "cam01", ts(day=1))
    assert not r.route(Source.RANDOM, "cam01", ts(day=1)).accept
    assert r.route(Source.RANDOM, "cam01", ts(day=2)).accept


def test_quota_persists_across_restart(settings):
    """Quota nằm trong DB: crash rồi chạy lại không được lấy mẫu gấp đôi."""
    settings.daily_sample_budget = 20
    settings.per_hour_cap_ratio = 1.0
    settings.per_camera_cap_ratio = 1.0
    cap = settings.quota_for("random")

    r1 = SamplingRouter(FakeGuard(), settings)
    for _ in range(cap):
        r1.accept(Source.RANDOM, "cam01", ts())

    r2 = SamplingRouter(FakeGuard(), settings)  # "process mới"
    assert not r2.route(Source.RANDOM, "cam01", ts()).accept


def test_backpressure_blocks_uncertainty_first(settings):
    """DEGRADED: nguồn A bị cắt trước, B và C vẫn chạy."""
    guard = FakeGuard(allowed={Source.FAILURE_MINING, Source.RANDOM})
    r = SamplingRouter(guard, settings)
    assert not r.route(Source.UNCERTAINTY, "cam01", ts()).accept
    assert r.route(Source.FAILURE_MINING, "cam01", ts()).accept
    assert r.route(Source.RANDOM, "cam01", ts()).accept
