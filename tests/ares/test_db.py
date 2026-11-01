from datetime import datetime, time, timedelta

import pytest

from ares.db_manager import DBManager, TelemetryLog
from ares.state import Event


def put(db, ts, occ, **kw):
    db.log_telemetry({"timestamp": ts, "occ_total": occ, "occ_zones": [], **kw})


def test_three_tables_exist(db):
    from sqlalchemy import inspect

    names = set(inspect(db.engine).get_table_names())
    assert {"telemetry_logs", "habit_profiles", "active_learning_queue"} <= names


def test_telemetry_roundtrip_and_bucketing(db, clock):
    t0 = datetime(2026, 11, 2, 8, 0)
    for i in range(10):
        put(db, t0 + timedelta(seconds=60 * i), 10 + i, temp_in=25.0 + i / 10, led_pwm=[40, 50], ac_state={"power": 1, "temp": 25, "fan": 2})
    db.flush()
    rows = db.telemetry_range(t0, t0 + timedelta(minutes=10), bucket_s=300)
    assert len(rows) == 2 and rows[0]["occ_total"] == pytest.approx(12.0) and rows[0]["ac_power"] == 1
    assert rows[0]["led_pwm"] == [40.0, 50.0] and rows[0]["ac_temp"] == 25


def test_habits_probability_headcount_and_null_ignored(db, clock):
    """4 thứ Hai liên tiếp 08:00-08:15: 3 buổi có 20 người, 1 buổi trống => P = 0.75, sĩ số trung vị 20."""
    base = datetime(2026, 10, 12, 8, 0)
    for wk in range(4):
        occ = 0 if wk == 3 else 20
        for m in range(15):
            put(db, base + timedelta(weeks=wk, minutes=m), occ)
    for m in range(15):  # tuần thứ 5: camera chết => NULL, KHÔNG được tính là phòng trống
        put(db, base + timedelta(weeks=4, minutes=m), None)
    db.flush()
    out = db.aggregate_habits(datetime(2026, 11, 20, 0, 0))
    assert out["slots_updated"] >= 1
    info = db.get(2, time(8, 0))
    assert info.occ_probability == pytest.approx(0.75) and info.expected_headcount == 20 and info.n_obs == 4


def test_aggregate_keeps_learned_cooling_rate(db, clock):
    db.set_cooling_rate(2, time(7, 0), 0.087)
    base = datetime(2026, 10, 26, 7, 0)
    for m in range(15):
        put(db, base + timedelta(minutes=m), 25)
    db.flush()
    db.aggregate_habits(datetime(2026, 11, 1))
    info = db.get(2, time(7, 0))
    assert info.avg_cooling_rate == pytest.approx(0.087) and info.occ_probability == 1.0


def test_mean_cooling_rate_and_missing(db):
    assert db.get(2, time(7, 0)) is None and db.mean_cooling_rate() is None
    db.set_cooling_rate(2, time(7, 0), 0.10)
    db.set_cooling_rate(3, time(7, 0), 0.06)
    assert db.mean_cooling_rate() == pytest.approx(0.08)


def test_events_logged_and_filtered(db, clock):
    db.log_event(Event(clock.time(), "A", "info", "một"))
    db.log_event(Event(clock.time() + 1, "B", "warn", "hai", {"x": object()}))  # dữ liệu không JSON hoá được không làm hỏng ghi
    db.flush()
    assert [e["kind"] for e in db.recent_events(10)] == ["B", "A"]
    assert [e["kind"] for e in db.recent_events(10, ["A"])] == ["A"]
    assert db.last_event_time("A") is not None and db.last_event_time("Z") is None


def test_writer_never_blocks_and_counts_drops(cfg, clock):
    d = DBManager(cfg.database_url, cfg.room_id, clock, start_writer=False)
    d._q.maxsize = 3
    for i in range(10):
        d.log_telemetry({"timestamp": datetime(2026, 1, 1) + timedelta(seconds=i), "occ_total": 1})
    assert d.dropped == 7
    d.close()


def test_one_bad_row_does_not_poison_the_batch(db):
    ts = datetime(2026, 11, 2, 9, 0)
    db.log_telemetry({"timestamp": ts, "occ_total": 1})
    db.log_telemetry({"timestamp": ts + timedelta(seconds=1), "cot_khong_ton_tai": 5})  # làm lệnh ghi này lỗi
    db.log_telemetry({"timestamp": ts + timedelta(seconds=2), "occ_total": 3})
    db.flush()
    with db.session() as s:
        got = sorted(r.occ_total for r in s.query(TelemetryLog).all())
    assert got == [1, 3] and db.write_errors == 1


def test_purge_old_telemetry(db, clock):
    put(db, clock.naive_now() - timedelta(days=500), 5)
    put(db, clock.naive_now() - timedelta(days=1), 5)
    db.flush()
    assert db.purge_old_telemetry(days=400) == 1
