from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from ares.clock import ManualClock, RealClock, SimClock, get_tz, vn_weekday
from ares.config import REPO_ROOT, AresSettings
from ares.nightly import NightlyJobs


class StubRetrainer:
    def __init__(self):
        self.calls = []

    def run(self, trigger, force=False):
        self.calls.append(trigger)
        return {"status": "skipped"}


@pytest.fixture
def nj(cfg, clock, db, events):
    rt = StubRetrainer()
    return NightlyJobs(cfg, clock, db, None, rt, events), rt


def test_not_due_before_midnight_slot_and_runs_once(nj, clock):
    n, rt = nj
    clock.set(datetime(2026, 11, 3, 0, 0, 1))
    assert n.due() is True
    n.tick()
    assert rt.calls == ["midnight"] and n.due() is False
    clock.advance(3600)
    assert n.tick() is None and rt.calls == ["midnight"]
    clock.set(datetime(2026, 11, 4, 0, 5))
    assert n.due() is True  # đêm sau


def test_catchup_window_after_machine_was_off(nj, clock, cfg):
    n, rt = nj
    clock.set(datetime(2026, 11, 3, 3, 0))  # bật máy lúc 3h sáng: bù trong 6h
    assert n.due() is True
    clock.set(datetime(2026, 11, 3, 15, 0))  # 15h chiều: quá hạn bù, không train giữa giờ học
    n.last_run_local = None
    assert n.due() is False


def test_done_marker_survives_restart_via_db(cfg, clock, db):
    from ares.state import Event

    n1 = NightlyJobs(cfg, clock, db, None, StubRetrainer(), db.log_event)
    clock.set(datetime(2026, 11, 3, 0, 1))
    n1.emit = lambda e: db.log_event(e)
    n1.run()
    db.flush()
    n2 = NightlyJobs(cfg, clock, db, None, StubRetrainer(), None)  # tiến trình mới
    assert n2.due() is False


def test_failing_step_does_not_block_the_others(cfg, clock, events):
    class BrokenDB:
        def aggregate_habits(self, now): raise RuntimeError("DB hỏng")
        def purge_old_telemetry(self): return 0
        def last_event_time(self, k): return None
    rt = StubRetrainer()
    n = NightlyJobs(cfg, clock, BrokenDB(), None, rt, events)
    out = n.run()
    assert "error" in out["habits"] and rt.calls == ["midnight"] and "NIGHTLY_STEP_FAILED" in events.kinds() and "NIGHTLY_DONE" in events.kinds()


def test_retrain_disabled_by_config(mk_cfg, clock, db, events):
    cfg = mk_cfg(retrain_enabled=False)
    rt = StubRetrainer()
    NightlyJobs(cfg, clock, db, None, rt, events).run()
    assert rt.calls == []


# ================================================================================== cấu hình
def S(**kw):
    return AresSettings(_env_file=None, **kw)


def test_defaults_match_spec_numbers():
    c = S()
    assert (c.kp, c.dl_max, c.lux_occupied, c.lux_dense, c.lux_idle) == (0.5, 450.0, 400.0, 450.0, 100.0)
    assert (c.lighting_window_s, c.hvac_window_s) == (30.0, 600.0)
    assert (c.t_base, c.t_min, c.t_max, c.deadband) == (26.5, 22.0, 27.0, 0.7)
    assert (c.al_conf_low, c.al_conf_high, c.al_daily_cap) == (0.35, 0.55, 0)  # 0 = không giới hạn; lọc PCA thay cho cap 50/ngày
    assert (c.precool_prob, c.failsafe_min, c.replay_old_ratio) == (0.75, 15.0, 0.7)


def test_yolo_must_emit_detections_inside_uncertain_band():
    with pytest.raises(ValidationError, match="detect_conf"):
        S(detect_conf=0.5)


@pytest.mark.parametrize("kw", [dict(al_conf_low=0.6, al_conf_high=0.5), dict(idle_pwm_min=20, idle_pwm_max=10),
                                dict(t_min=28), dict(precool_ac_temp=27), dict(replay_old_ratio=1.0), dict(retrain_time="25:00")])
def test_invalid_settings_rejected(kw):
    with pytest.raises(ValidationError):
        S(**kw)


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("ARES_KP", "0.4")
    monkeypatch.setenv("ARES_LED_WATTS", "[50,50,50,50]")
    c = AresSettings(_env_file=None)
    assert c.kp == 0.4 and c.led_watts == [50, 50, 50, 50]


def test_simulation_mode_isolates_storage_and_disables_training():
    c = S(simulation_mode=True)
    assert c.storage_root == REPO_ROOT / "var" / "ares_sim" and c.retrain_enabled is False
    assert S().storage_root == REPO_ROOT / "var" / "ares"
    c2 = S(simulation_mode=True, storage_root="/tmp/xyz", retrain_enabled=True)
    assert str(c2.storage_root) == "/tmp/xyz" and c2.retrain_enabled is True  # chủ động đặt thì tôn trọng


def test_database_url_defaults_to_sqlite_in_storage(tmp_path):
    assert S(storage_root=tmp_path).database_url == f"sqlite:///{(tmp_path / 'ares.db').as_posix()}"


# ================================================================================== đồng hồ
def test_manual_and_sim_clock():
    m = ManualClock(datetime(2026, 11, 2, 6, 0))
    t = m.time(); m.advance(90)
    assert m.time() - t == 90 and m.now().minute == 1 and m.naive_now().tzinfo is None
    s = SimClock("Asia/Ho_Chi_Minh", start=datetime(2026, 11, 2, 6, 0), speed=1000.0)
    import time; time.sleep(0.05)
    assert s.time() - datetime(2026, 11, 2, 6, 0, tzinfo=get_tz("Asia/Ho_Chi_Minh")).timestamp() > 20
    assert s.real_timeout(1000) == pytest.approx(1.0)


def test_vn_weekday_convention():
    assert vn_weekday(datetime(2026, 11, 2)) == 2 and vn_weekday(datetime(2026, 11, 8)) == 8


def test_tz_fallback_to_fixed_utc7_when_tzdata_missing(monkeypatch):
    import zoneinfo
    monkeypatch.setattr(zoneinfo, "ZoneInfo", lambda n: (_ for _ in ()).throw(Exception("no tzdata")))
    tz = get_tz("Asia/Ho_Chi_Minh")
    assert tz.utcoffset(None) == timedelta(hours=7)
