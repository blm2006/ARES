from datetime import date, datetime

import pytest

from ares.clock import get_tz
from ares.schedule import StaticSchedule

TZ = get_tz("Asia/Ho_Chi_Minh")
RAW = {
    "sessions": [
        {"name": "A", "days": [2, 4], "start": "07:00", "end": "09:30"},
        {"name": "B", "days": [2], "start": "09:45", "end": "12:15"},
    ],
    "exceptions": [{"date": "2026-11-04", "cancelled": True}],
}


def dt(d, h, m=0):
    return datetime(2026, 11, d, h, m, tzinfo=TZ)


def test_weekday_convention_monday_is_2():
    assert date(2026, 11, 2).isoweekday() == 1  # thứ Hai
    s = StaticSchedule.from_dict(RAW, TZ)
    assert [x.name for x in s.sessions_on(date(2026, 11, 2))] == ["A", "B"]
    assert s.sessions_on(date(2026, 11, 3)) == []  # thứ Ba không có lớp


def test_exception_cancels_whole_day():
    s = StaticSchedule.from_dict(RAW, TZ)
    assert s.sessions_on(date(2026, 11, 4)) == []  # thứ Tư có lớp nhưng bị huỷ


def test_current_and_next_within():
    s = StaticSchedule.from_dict(RAW, TZ)
    assert s.current(dt(2, 8)).name == "A"
    assert s.current(dt(2, 9, 30)) is None  # kết thúc là nửa mở
    assert s.next_within(dt(2, 6, 20), 60).name == "A"
    assert s.next_within(dt(2, 6, 20), 30) is None
    assert s.next_within(dt(2, 7, 0), 180).name == "B"  # lớp A đang diễn ra không tính là "sắp tới"


def test_next_within_crosses_midnight():
    raw = {"sessions": [{"name": "Sớm", "days": [3], "start": "00:30", "end": "01:30"}]}
    s = StaticSchedule.from_dict(raw, TZ)
    assert s.next_within(dt(2, 23, 45), 60).name == "Sớm"


def test_invalid_rules_rejected():
    with pytest.raises(ValueError, match="kết thúc"):
        StaticSchedule.from_dict({"sessions": [{"days": [2], "start": "10:00", "end": "09:00"}]}, TZ)
    with pytest.raises(ValueError, match="2..8"):
        StaticSchedule.from_dict({"sessions": [{"days": [1], "start": "07:00", "end": "08:00"}]}, TZ)


def test_missing_file_means_no_sessions(tmp_path):
    s = StaticSchedule.from_file(tmp_path / "nope.json", TZ)
    assert s.current(dt(2, 8)) is None and s.next_within(dt(2, 6), 600) is None


def test_hot_reload(tmp_path):
    import json, os, time

    f = tmp_path / "s.json"
    f.write_text(json.dumps({"sessions": []}), encoding="utf-8")
    s = StaticSchedule.from_file(f, TZ)
    assert s.current(dt(2, 8)) is None
    f.write_text(json.dumps({"sessions": [{"name": "Mới", "days": [2], "start": "07:00", "end": "09:00"}]}), encoding="utf-8")
    os.utime(f, (time.time() + 5, time.time() + 5))
    assert s.maybe_reload(every_s=0) is True
    assert s.current(dt(2, 8)).name == "Mới"
    f.write_text("{hỏng", encoding="utf-8")
    os.utime(f, (time.time() + 10, time.time() + 10))
    assert s.maybe_reload(every_s=0) is False  # file đang sửa dở: giữ lịch cũ
    assert s.current(dt(2, 8)).name == "Mới"


def test_shipped_schedule_is_valid():
    from ares.config import CONFIG_DIR

    s = StaticSchedule.from_file(CONFIG_DIR / "schedule.json", TZ)
    assert len(s.rules) >= 1
