import pytest

from ares.config import AresSettings
from ares.control_engine import LightInputs, LightingController

K = [3.25] * 4
T0 = 1_000_000.0


@pytest.fixture
def cfg(cfg_spec):
    return cfg_spec  # công thức gốc của spec, không lớp chống bật/tắt (xem test_anti_chatter.py)


@pytest.fixture
def ctl(cfg):
    return LightingController(cfg, K)


def inp(t, n=(0, 0, 0, 0), lux_c=(0, 0, 0, 0), vision_ok=True, in_session=True, win=None):
    return LightInputs(t=t, n_avg=list(n) if vision_ok else None, vision_ok=vision_ok, in_session=in_session,
                       lux_ceiling=list(lux_c), lux_window=win)


def run(ctl, steps, t=T0, dt=15.0, **kw):
    out = None
    for _ in range(steps):
        out = ctl.step(inp(t, **kw))
        t += dt
    return out, t


def test_onset_jumps_fast_with_kp_1_then_settles_with_kp_half(ctl):
    out = ctl.step(inp(T0, n=(5, 0, 0, 0)))
    assert out.zone_state[0] == "OCCUPIED"
    assert out.pwm[0] == pytest.approx(100 * 1.0 * 400 / 450, abs=0.2)  # bước đầu Kp = 1.0
    nxt = ctl.step(inp(T0 + 15, n=(5, 0, 0, 0), lux_c=(out.pwm[0] / 100 * 450 / 3.25, 0, 0, 0)))
    assert nxt.pwm[0] == pytest.approx(out.pwm[0], abs=0.5)  # đã đúng 400 lux => giữ


def test_empty_zone_in_occupied_room_is_idle_with_10_15_pct(ctl):
    out = ctl.step(inp(T0, n=(5, 0, 0, 0)))
    for i in (1, 2, 3):
        assert out.zone_state[i] == "IDLE" and 10.0 <= out.pwm[i] <= 15.0 and out.setpoint[i] == 100


def test_idle_pwm_stays_clamped_even_with_huge_daylight(ctl):
    out, _ = run(ctl, 20, n=(5, 0, 0, 0), lux_c=(100, 500, 500, 500))
    assert all(10.0 <= out.pwm[i] <= 15.0 for i in (1, 2, 3))


def test_grace_then_dim_then_off(ctl):
    t = T0
    out, t = run(ctl, 10, t=t, n=(5, 5, 5, 5), lux_c=(123,) * 4)  # đang có người
    held = out.pwm[0]
    # vừa vắng: GRACE giữ nguyên mục tiêu cũ trong 3 phút
    out = ctl.step(inp(t, n=(0, 0, 0, 0), lux_c=(123,) * 4))
    assert out.zone_state == ["GRACE"] * 4 and out.setpoint == [400.0] * 4
    # > 3 phút vắng cả phòng => TẮT 0 %
    out, t = run(ctl, 15, t=t + 15, n=(0, 0, 0, 0), lux_c=(0,) * 4)
    assert out.zone_state == ["OFF"] * 4 and out.pwm == [0.0] * 4
    assert held > 0


def test_one_zone_empty_gets_idle_not_off_when_others_occupied(ctl):
    t = T0
    _, t = run(ctl, 3, t=t, n=(5, 5, 0, 0))
    out, t = run(ctl, 20, t=t, n=(5, 0, 0, 0))  # zone 2 vắng > 3 phút nhưng zone 1 vẫn có người
    assert out.zone_state[0] == "OCCUPIED" and out.zone_state[1] == "IDLE" and out.pwm[1] >= 10


def test_dense_zone_gets_450(ctl):
    out = ctl.step(inp(T0, n=(9, 0, 0, 0)))
    assert out.setpoint[0] == 450.0


def test_daylight_harvesting_drives_pwm_to_zero(ctl):
    out, _ = run(ctl, 30, n=(5, 5, 5, 5), lux_c=(200,) * 4)  # 200*3.25 = 650 lux > 400
    assert out.pwm == [0.0] * 4 and out.zone_state == ["OCCUPIED"] * 4


def test_closed_loop_converges_to_400(ctl):
    pwm, t = [0.0] * 4, T0
    daylight = 120.0
    for _ in range(40):
        desk = daylight + pwm[0] / 100 * 450
        out = ctl.step(inp(t, n=(5, 5, 5, 5), lux_c=(desk / 3.25,) * 4))
        pwm, t = out.pwm, t + 15
    assert abs(daylight + pwm[0] / 100 * 450 - 400) < 3


def test_sensor_fault_holds_then_falls_back_to_safe_pwm(ctl, cfg):
    t = T0
    out, t = run(ctl, 4, t=t, n=(5, 5, 5, 5), lux_c=(40,) * 4)
    before = out.pwm[0]
    out = ctl.step(inp(t, n=(5, 5, 5, 5), lux_c=(None,) * 4))
    assert out.pwm[0] == pytest.approx(before) and out.actual == [None] * 4  # chưa đủ lâu: giữ PWM
    out, _ = run(ctl, int(cfg.sensor_fault_s / 15) + 2, t=t + 15, n=(5, 5, 5, 5), lux_c=(None,) * 4)
    assert all(p >= cfg.fallback_pwm_occupied for p in out.pwm)
    assert any("PWM dự phòng" in n for n in out.notes)


def test_camera_loss_during_class_assumes_occupied(ctl):
    out = ctl.step(inp(T0, vision_ok=False, in_session=True, lux_c=(0,) * 4))
    assert out.vision_mode == "assume_occupied" and out.zone_state == ["OCCUPIED"] * 4 and min(out.pwm) > 50


def test_camera_loss_outside_class_holds_state(ctl):
    run(ctl, 5, n=(5, 5, 5, 5), lux_c=(60,) * 4)
    before = list(ctl.pwm)
    out = ctl.step(inp(T0 + 999, vision_ok=False, in_session=False, lux_c=(60,) * 4))
    assert out.vision_mode == "hold" and out.zone_state == ["HOLD"] * 4 and out.pwm == [round(p, 1) for p in before]


def test_false_positive_blip_does_not_turn_on_lights(ctl, cfg):
    # 1 khung trong cửa sổ ~15 khung => trung bình 0.07 < ngưỡng 0.35
    out = ctl.step(inp(T0, n=(1 / 15, 0, 0, 0)))
    assert out.zone_state[0] != "OCCUPIED"
    assert cfg.occ_on_threshold >= 0.3
