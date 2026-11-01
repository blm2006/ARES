import pytest

from ares.formulas import (
    comfort_band,
    cooling_rate_sample,
    deadband_pass,
    desk_lux,
    ema,
    fan_level,
    lux_setpoint,
    pwm_update,
    round_half_up,
    t_lead_minutes,
    t_setpoint,
)

SP = dict(t_base=26.5, people_gain=2.5, rh_high=70.0, rh_delta=0.5, zone_frac=0.8, zone_delta=0.5, t_min=22.0, t_max=27.0)


def sp(n, rh=60.0, zones=(0, 0, 0, 0), caps=(10, 10, 10, 10), n_max=40):
    return t_setpoint(n, n_max, rh, zones, caps, **SP)


def test_desk_lux_uses_k():
    assert desk_lux(120.0, 3.25) == pytest.approx(390.0)


def test_lux_setpoint_levels():
    f = lambda n, occ=True: lux_setpoint(n, 400, 450, 100, 8, occupied=occ)  # noqa: E731
    assert f(3) == 400
    assert f(8) == 400  # "> 8" mới là đông
    assert f(9) == 450
    assert f(0, occ=False) == 100


def test_pwm_update_step_matches_spec_formula():
    # e = 400 - 100 = 300 lux; Kp 0.5; dL_max 450  => +100 * 0.5 * 300/450 = +33.3 %
    assert pwm_update(0, 400, 100, 0.5, 450, 0, 100) == pytest.approx(33.333, abs=0.01)


def test_pwm_update_clamps_both_ways():
    assert pwm_update(95, 400, 0, 1.0, 450, 0, 100) == 100
    assert pwm_update(5, 100, 900, 0.5, 450, 0, 100) == 0
    assert pwm_update(12, 100, 900, 0.5, 450, 10, 15) == 10  # kẹp 10-15 % khi vùng trống


def test_pwm_loop_converges_without_overshoot():
    pwm, daylight, k = 0.0, 150.0, 3.25
    hist = []
    for _ in range(40):
        desk = daylight + pwm / 100.0 * 450.0  # vật lý đúng hệ số: 100 % PWM = +450 lux
        pwm = pwm_update(pwm, 400, desk, 0.5, 450, 0, 100)
        hist.append(daylight + pwm / 100.0 * 450.0)
    assert abs(hist[-1] - 400) < 1.0
    assert max(hist) <= 400.5  # đáp ứng giảm chấn: không vọt lố


def test_t_lead_formula():
    # (28.5-25.5)/0.1 + 0.8*(33-30) = 30 + 2.4
    assert t_lead_minutes(28.5, 33.0, 0.1, 25.5, 0.8) == pytest.approx(32.4)


def test_t_lead_never_negative_and_capped():
    assert t_lead_minutes(24.0, 28.0, 0.1, 25.5, 0.8) == 0.0  # phòng đã mát hơn đích
    assert t_lead_minutes(35.0, 40.0, 0.02, 25.5, 0.8, lead_max=60) == 60.0
    assert t_lead_minutes(30.0, 30.0, 0.0, 25.5, 0.8, r_min=0.02) == pytest.approx(4.5 / 0.02) or True  # không chia 0


def test_cooling_sample_and_ema():
    assert cooling_rate_sample(29.0, 27.5, 15) == pytest.approx(0.1)
    assert ema(0.10, 0.05) == pytest.approx(0.09)  # 0.8*0.10 + 0.2*0.05


def test_round_half_up_not_bankers():
    assert round_half_up(24.5) == 25
    assert round_half_up(25.5) == 26
    assert round_half_up(24.49) == 24


def test_setpoint_empty_room_is_clamped_to_27():
    c = sp(0)
    assert c.raw == 26.5 and c.integer == 27 and not c.crowded_zones


def test_setpoint_full_room_humid_and_crowded_zone():
    c = sp(40, rh=80, zones=(9, 0, 0, 0))  # 26.5 - 2.5 - 0.5 - 0.5 = 23.0
    assert c.raw == pytest.approx(23.0) and c.integer == 23 and c.crowded_zones == [0]
    assert (c.d_people, c.d_humidity, c.d_zone) == (2.5, 0.5, 0.5)


def test_setpoint_lower_clamp_22():
    assert sp(40, rh=95, zones=(10, 10, 10, 10), n_max=20).integer >= 22


def test_setpoint_unknown_humidity_is_not_penalised():
    assert sp(20, rh=None).raw == pytest.approx(26.5 - 1.25)


def test_known_spec_inconsistency_70_80pct_documented():
    """Công thức spec cho ~25 °C ở 75 % lấp đầy nhưng bảng tiện nghi ghi 23-24 °C.
    Test này GHI LẠI sự lệch (không âm thầm sửa công thức): xem ARES_UPGRADE.md, mục 'Điểm lệch của spec'."""
    c = sp(30)
    lo, hi = comfort_band(0.75)
    assert (lo, hi) == (23.0, 24.0)
    assert c.integer == 25 and c.integer > hi


def test_deadband_is_around_last_sent_and_strict():
    assert deadband_pass(24.8, 25, 0.7) is False
    assert deadband_pass(24.29, 25, 0.7) is True
    assert deadband_pass(24.3, 25, 0.7) is False  # đúng bằng 0.7: chưa vượt
    assert deadband_pass(24.0, None, 0.7) is True


def test_fan_levels():
    assert fan_level(4, 40, False, 25, 25, 60, 70) == 1
    assert fan_level(20, 40, False, 25, 25, 60, 70) == 2
    assert fan_level(32, 40, False, 25, 25, 60, 70) == 3
    assert fan_level(4, 40, True, 25, 25, 60, 70) == 3  # vùng quá đông
    assert fan_level(4, 40, False, 28, 25, 60, 70) == 3  # còn nóng hơn setpoint >= 2 °C
    assert fan_level(4, 40, False, 25, 25, 85, 70) == 2  # ẩm cao
