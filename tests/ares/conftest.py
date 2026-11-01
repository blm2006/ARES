"""Fixture chung cho test ARES. Chạy offline: không camera, không MQTT broker, không GPU."""

from __future__ import annotations

from datetime import datetime

import pytest

from ares.clock import ManualClock
from ares.config import AresSettings


@pytest.fixture(autouse=True)
def isolated_env():
    """Ghi đè fixture autouse cùng tên ở tests/conftest.py (dựng DB/queue của hệ thống cũ) — test ARES không cần."""
    yield


# Tắt các lớp chống bật/tắt liên tục (trễ, vùng chết, máy nén) — để test đúng công thức gốc của spec.
# Các lớp đó có test riêng ở test_anti_chatter.py với giá trị mặc định.
NO_ANTICHATTER = dict(light_on_delay_s=0.0, lux_deadband=0.0, pwm_max_step=100.0, occ_off_threshold=0.35,
                      ac_on_delay_s=0.0, ac_off_delay_s=0.0, ac_min_on_s=0.0, ac_min_off_s=0.0, fan_hysteresis=0.0, ac_keep_on_before_class_min=0.0,
                      ac_setpoint_max_step=5)


@pytest.fixture
def cfg_spec(tmp_path):
    """Cấu hình 'spec thuần' (không lớp chống bật/tắt) cho test công thức điều khiển."""
    c = AresSettings(_env_file=None, storage_root=tmp_path / "ares", al_hourly_cap=100, al_min_gap_s=0.0, pca_enabled=False,
                     **NO_ANTICHATTER)
    c.ensure_dirs()
    return c


@pytest.fixture
def cfg(tmp_path):
    c = AresSettings(_env_file=None, storage_root=tmp_path / "ares", al_hourly_cap=100, al_min_gap_s=0.0, pca_enabled=False)
    c.ensure_dirs()
    return c


@pytest.fixture
def clock():
    return ManualClock(datetime(2026, 11, 2, 6, 0))  # thứ Hai


@pytest.fixture
def mk_cfg(tmp_path):
    def make(**over):
        c = AresSettings(_env_file=None, storage_root=tmp_path / "ares", **over)
        c.ensure_dirs()
        return c
    return make


@pytest.fixture
def db(cfg, clock):
    from ares.db_manager import DBManager

    d = DBManager(cfg.database_url, cfg.room_id, clock)
    yield d
    d.close()


@pytest.fixture
def events():
    """Danh sách thu các Event mà thành phần phát ra."""
    class Bag(list):
        def __call__(self, e):
            self.append(e)

        def kinds(self):
            return [e.kind for e in self]
    return Bag()


@pytest.fixture
def al(cfg, clock, db, events):
    from ares.active_learner import ActiveLearner

    a = ActiveLearner(cfg, clock, db, emit=events)
    a.start()
    yield a
    a.drain(5)
    a.stop()
