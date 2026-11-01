from datetime import datetime, time, timedelta

import pytest

from ares.clock import get_tz
from ares.control_engine import HabitInfo, HvacController, HvacInputs
from ares.schedule import StaticSchedule
from ares.state import AcState

TZ = get_tz("Asia/Ho_Chi_Minh")
RAW = {"sessions": [{"name": "Lớp 1", "days": [2], "start": "07:00", "end": "09:30"},
                    {"name": "Lớp 2", "days": [2], "start": "09:45", "end": "12:15"},
                    {"name": "Lớp 3", "days": [2], "start": "13:00", "end": "15:30"}]}


class FakeHabit:
    def __init__(self, p=None, n_obs=5, rate=None):
        self.p, self.n_obs, self.rate = p, n_obs, rate
        self.saved = []

    def get(self, dow, slot):
        return None if self.p is None else HabitInfo(self.p, 30, self.rate, self.n_obs)

    def set_cooling_rate(self, dow, slot, rate):
        self.saved.append((dow, slot, rate))

    def mean_cooling_rate(self):
        return self.rate


def at(h, m=0, s=0):
    return datetime(2026, 11, 2, h, m, s, tzinfo=TZ)


def I(now, t_in=29.0, t_out=31.0, rh=60.0, n=0.0, cov=1.0, zones=(0, 0, 0, 0), fresh=True):
    return HvacInputs(now=now, t_in=t_in, t_out=t_out, rh=rh, n_total=n, coverage=cov, zone_avgs=list(zones), vision_fresh=fresh)


@pytest.fixture
def cfg(cfg_spec):
    return cfg_spec  # công thức gốc của spec, không lớp chống bật/tắt (xem test_anti_chatter.py)


@pytest.fixture
def mk(cfg):
    def make(habit=None, raw=RAW):
        return HvacController(cfg, 40, [10, 10, 10, 10], StaticSchedule.from_dict(raw, TZ), habit)
    return make


def kinds(res):
    return [e.kind for e in res.events]


# ----------------------------------------------------------------------------- pre-cooling
def test_precool_plans_early_enough_and_starts_on_time(mk):
    c = mk()
    r = c.step(I(at(6, 20)))
    assert "PRECOOL_PLANNED" in kinds(r) and r.command is None
    # t_lead = (29-25.5)/0.1 + 0.8*(31-30) = 35.8 phút => bật khoảng 06:24
    assert c.plan.t_lead_min == pytest.approx(35.8, abs=0.1)
    r = c.step(I(at(6, 25)))
    assert r.command is not None and (r.command.power, r.command.temp, r.command.fan) == (1, 25, 3)
    assert r.command.reason == "PRECOOL" and r.command.urgent and "PRECOOL_START" in kinds(r)


def test_lead_longer_than_30_min_is_still_planned_in_time(mk):
    """Lỗi do mô phỏng lộ ra: nhìn trước chỉ 30 phút thì t_lead 45 phút không bao giờ kịp."""
    c = mk()
    r = c.step(I(at(6, 5), t_in=30.0, t_out=30.0))  # (30-25.5)/0.1 = 45 phút
    assert "PRECOOL_PLANNED" in kinds(r)
    assert c.plan.start_at == at(7, 0) - timedelta(minutes=45)


def test_no_precool_when_habit_says_unlikely(mk):
    c = mk(FakeHabit(p=0.5))
    r = c.step(I(at(6, 30)))
    assert "PRECOOL_SKIPPED" in kinds(r) and c.plan is None and r.command is None
    assert c.step(I(at(6, 55))).command is None


def test_precool_threshold_is_strictly_greater_than_075(mk):
    c = mk(FakeHabit(p=0.75))
    assert "PRECOOL_SKIPPED" in kinds(c.step(I(at(6, 30))))
    c = mk(FakeHabit(p=0.76))
    assert "PRECOOL_PLANNED" in kinds(c.step(I(at(6, 30))))


def test_schedule_prior_used_when_not_enough_history(mk, cfg):
    c = mk(FakeHabit(p=0.1, n_obs=1))  # mới 1 lần quan sát: chưa đáng tin => dùng lịch (0.9)
    assert "PRECOOL_PLANNED" in kinds(c.step(I(at(6, 30))))


def test_precool_keeps_fan_high_until_class_starts(mk):
    c = mk()
    c.step(I(at(6, 20)))
    c.step(I(at(6, 30)))  # đã bật
    c.next_eval_at = 0
    r = c.step(I(at(6, 50), n=2.0, zones=(1, 1, 0, 0), t_in=27.0))  # vài người đến sớm
    assert r.command is None  # không được hạ quạt/đổi setpoint khi đang pre-cool
    assert r.status.mode == "PRECOOL"


def test_replans_after_ac_turned_off_during_gap(mk):
    """Lỗi do mô phỏng lộ ra: AC đang bật lúc 12:00 => 'chưa cần pre-cool', rồi nghỉ trưa AC tắt => phải lập lại."""
    c = mk()
    c.ac = AcState(power=1, temp=25, fan=2, t=at(11).timestamp(), known=True)
    r = c.step(I(at(12, 0), t_in=25.0))
    assert "PRECOOL_NOT_NEEDED" in kinds(r) and c.plan is None
    c.ac = AcState(power=0, temp=25, fan=0, t=at(12, 30).timestamp(), known=True)
    r = c.step(I(at(12, 40), t_in=26.8, t_out=33.0))
    assert "PRECOOL_PLANNED" in kinds(r)
    assert c.step(I(at(12, 45), t_in=26.8, t_out=33.0)).command is not None or c.plan.start_at > at(12, 45)


def test_not_needed_event_emitted_once(mk):
    c = mk()
    c.ac = AcState(power=1, temp=25, fan=2, t=0, known=True)
    n = sum("PRECOOL_NOT_NEEDED" in kinds(c.step(I(at(12, m), t_in=25.0))) for m in range(0, 20, 5))
    assert n == 1


# ----------------------------------------------------------------------------- fail-safe
def started_precool(c):
    c.step(I(at(6, 20)))
    r = c.step(I(at(6, 30)))
    assert r.command and r.command.power == 1
    return c


def test_failsafe_turns_off_after_15_min_if_room_empty(mk):
    c = started_precool(mk())
    assert c.step(I(at(7, 14), n=0.0)).command is None
    r = c.step(I(at(7, 15, 30), n=0.0))
    assert r.command and r.command.power == 0 and r.command.reason == "FAILSAFE_EMPTY"
    assert "FAILSAFE_EMPTY_CLASS" in kinds(r) and r.status.mode == "FAILSAFE_OFF"


def test_failsafe_does_not_fire_when_people_present(mk):
    c = started_precool(mk())
    r = c.step(I(at(7, 16), n=12.0, zones=(3, 3, 3, 3)))
    assert r.command is None or r.command.power == 1


def test_failsafe_deferred_when_camera_dead_not_treated_as_empty(mk):
    c = started_precool(mk())
    r = c.step(I(at(7, 16), n=None, cov=0.0, fresh=False))
    assert r.command is None and "FAILSAFE_DEFERRED" in kinds(r)
    r = c.step(I(at(7, 20), n=0.0))  # camera hồi phục và phòng thật sự trống
    assert r.command and r.command.power == 0


def test_cancelled_class_does_not_turn_ac_back_on_by_occupancy(mk):
    c = started_precool(mk())
    c.step(I(at(7, 16), n=0.0))
    r = c.step(I(at(7, 20), n=3.0, zones=(3, 0, 0, 0), t_in=29.0))
    # có vài người lẻ tẻ vào phòng sau khi lớp bị huỷ: vẫn được bật theo hiện diện (không phải theo lớp)
    assert r.command is None or r.command.reason == "OCCUPANCY_ON"


# ----------------------------------------------------------------------------- trong lớp
def active(c, now_h, now_m, n, zones, t_in=25.0, rh=60.0, ac_temp=25, power=1, fan=2):
    c.ac = AcState(power=power, temp=ac_temp, fan=fan, t=at(now_h, now_m).timestamp() - 3600, known=True)
    c.next_eval_at = 0
    return c.step(I(at(now_h, now_m), n=n, zones=zones, t_in=t_in, rh=rh))


def test_setpoint_follows_occupancy_with_deadband(mk):
    c = mk()
    r = active(c, 8, 0, 30, (8, 8, 7, 7), ac_temp=26)  # raw = 26.5-1.875 = 24.625 -> chênh 1.375 > 0.7
    assert r.command and r.command.temp == 25 and "AC_SETPOINT" in kinds(r)
    c2 = mk()
    r = active(c2, 8, 0, 30, (8, 8, 7, 7), ac_temp=25, fan=3)  # chênh 0.375 < 0.7 và quạt đúng mức => KHÔNG bắn IR
    assert r.command is None


def test_deadband_stops_flip_flop_around_rounding_boundary(mk):
    c = mk()
    cmds = 0
    for i, n in enumerate([20.0, 21.0, 20.0, 21.0, 20.0, 21.0]):  # raw dao động quanh 25.25/25.2
        r = active(c, 8, i * 10, n, (5, 5, 5, 5), ac_temp=c.ac.temp if c.ac.known else 25)
        cmds += bool(r.command)
    assert cmds <= 1


def test_no_relax_setpoint_right_after_class_start(mk):
    c = mk()
    r = active(c, 7, 5, 6.0, (2, 2, 1, 1), ac_temp=24, fan=3)  # 10 phút đệm còn nhớ phòng vắng => raw ~26
    assert r.command is None or r.command.temp <= 24
    assert any("Giữ setpoint" in n for n in r.status.notes) or r.command is None


def test_relax_allowed_mid_class_when_room_really_emptied(mk):
    c = mk()
    r = active(c, 8, 30, 6.0, (2, 2, 1, 1), ac_temp=24, fan=3)
    assert r.command and r.command.temp >= 26


def test_ir_commands_throttled_unless_urgent(mk):
    c = mk()
    c.ac = AcState(power=1, temp=26, fan=2, t=at(8, 28).timestamp(), known=True)
    c.next_eval_at = 0
    r = c.step(I(at(8, 30), n=30.0, zones=(8, 8, 7, 7)))
    assert r.command is None and any("giãn cách" in n for n in r.status.notes)


# ----------------------------------------------------------------------------- theo hiện diện
def test_occupancy_based_turn_on_without_schedule(mk):
    c = mk(raw={"sessions": []})
    r = c.step(I(at(14, 0), n=6.0, zones=(2, 2, 1, 1), t_in=29.0))
    assert r.command and r.command.power == 1 and r.command.reason == "OCCUPANCY_ON" and "AC_ON_OCCUPANCY" in kinds(r)


def test_no_turn_on_when_room_already_cool(mk):
    c = mk(raw={"sessions": []})
    assert c.step(I(at(14, 0), n=6.0, zones=(2, 2, 1, 1), t_in=24.0)).command is None


def test_off_when_room_empty_outside_class(mk):
    c = mk()
    c.ac = AcState(power=1, temp=25, fan=2, t=at(9, 0).timestamp(), known=True)
    r = c.step(I(at(12, 40), n=0.0, t_in=25.0))
    assert r.command and r.command.power == 0 and "AC_OFF_EMPTY" in kinds(r)


def test_tolerates_isolated_false_positives_when_judging_empty(mk, cfg):
    c = mk()
    c.ac = AcState(power=1, temp=25, fan=2, t=at(9, 0).timestamp(), known=True)
    r = c.step(I(at(12, 40), n=cfg.empty_mean_max / 2))  # nhiễu YOLO: không được làm AC chạy mãi trong phòng trống
    assert r.command and r.command.power == 0


def test_no_vision_data_means_no_occupancy_decisions(mk):
    c = mk(raw={"sessions": []})
    r = c.step(I(at(14, 0), n=None, cov=0.0, fresh=False, t_in=30.0))
    assert r.command is None and any("Thiếu dữ liệu" in n for n in r.status.notes)


# ----------------------------------------------------------------------------- học R_cool
def test_r_cool_learned_after_15_minutes_of_precool(mk):
    h = FakeHabit(p=0.9, rate=0.10)
    c = mk(h)
    c.step(I(at(6, 20)))
    r = c.step(I(at(6, 30), t_in=29.0))
    assert r.command
    r = c.step(I(at(6, 46), t_in=27.5))  # 1.5 °C / 15 phút = 0.1
    assert r.cooling_rate_update is not None
    dow, slot, new = r.cooling_rate_update
    assert dow == 2 and slot == time(7, 0) and new == pytest.approx(0.8 * 0.10 + 0.2 * 0.10)


def test_r_cool_not_updated_if_ac_never_cooled(mk):
    c = mk(FakeHabit(p=0.9, rate=0.10))
    c.step(I(at(6, 20)))
    c.step(I(at(6, 30), t_in=29.0))
    r = c.step(I(at(6, 46), t_in=29.0))
    assert r.cooling_rate_update is None and "R_COOL_SKIPPED" in kinds(r)
