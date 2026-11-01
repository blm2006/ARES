"""Backpressure: điều kiện để "learning không ảnh hưởng realtime" là thật."""

from __future__ import annotations

from collector.backpressure import BackpressureGuard, Level
from common.schemas import Source


class FakeQueue:
    def __init__(self, depth=0):
        self._d = depth

    def depth(self):
        return self._d

    def close(self):
        pass


class FakeStorage:
    def __init__(self, free=90.0):
        self._f = free

    def free_pct(self):
        return self._f


def guard(depth, free, settings):
    g = BackpressureGuard(FakeQueue(depth), FakeStorage(free), settings, check_interval_s=0)
    return g


def test_normal_allows_all(settings):
    g = guard(0, 90.0, settings)
    assert g.state().level is Level.NORMAL
    for s in Source:
        assert g.allows(s)


def test_degraded_drops_uncertainty_only(settings):
    g = guard(settings.queue_depth_n1 + 1, 90.0, settings)
    assert g.state().level is Level.DEGRADED
    assert not g.allows(Source.UNCERTAINTY)
    assert g.allows(Source.FAILURE_MINING)  # lỗi vẫn đáng thu kể cả khi tắc
    assert g.allows(Source.RANDOM)
    assert g.can_write()


def test_stopped_when_queue_beyond_n2(settings):
    g = guard(settings.queue_depth_n2 + 1, 90.0, settings)
    assert g.state().level is Level.STOPPED
    assert not any(g.allows(s) for s in Source)
    assert not g.can_write()


def test_stopped_when_disk_low(settings):
    g = guard(0, settings.disk_free_min_pct - 1, settings)
    st = g.state()
    assert st.level is Level.STOPPED
    assert "disk" in st.reason
    assert not g.can_write()


def test_disk_takes_priority_over_queue(settings):
    g = guard(0, 1.0, settings)
    assert g.state().level is Level.STOPPED


def test_collector_writes_nothing_when_stopped(settings, img, monkeypatch):
    """Kiểm chứng end-to-end: STOPPED thì không có file nào xuống đĩa."""
    from datetime import datetime, timezone

    from collector.service import Collector
    from common.queue import get_queue
    from common.storage import get_storage

    c = Collector(settings, queue=get_queue(settings), storage=get_storage(settings))
    c.guard = guard(settings.queue_depth_n2 + 1, 90.0, settings)
    c.router.guard = c.guard

    out = c.on_frame(img, "cam01", datetime.now(timezone.utc), 1, [])
    assert out == []
    assert c.writer.written == 0
    assert get_queue(settings).depth() == 0
