"""Chống bật/tắt liên tục, chỉnh cài đặt thiết bị lúc đang chạy, điều tiết FPS khi train, hoán đổi model nạp nền.

Dùng cấu hình MẶC ĐỊNH (có trễ + vùng chết), khác test_lighting/test_hvac (công thức spec thuần).
"""

from __future__ import annotations

import json
import time as _time
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ares.clock import ManualClock, get_tz
from ares.config import AresSettings
from ares.control_engine import HvacController, HvacInputs, LightInputs, LightingController
from ares.fps import FpsGovernor
from ares.schedule import StaticSchedule
from ares.state import AcState
from ares.tuning import Tuning

K = [3.25] * 4
T0 = 1_000_000.0
TZ = get_tz("Asia/Ho_Chi_Minh")
RAW = {"sessions": [{"name": "Lớp 1", "days": [2], "start": "07:00", "end": "09:30"}]}
H = {"X-ARES": "1"}


def L(t, n=(0, 0, 0, 0), lux=(0, 0, 0, 0)):
    return LightInputs(t=t, n_avg=list(n), vision_ok=True, in_session=True, lux_ceiling=list(lux))


def lux_for(pwm_pct, k=3.25, dl=450.0):
    return pwm_pct / 100 * dl / k  # lux trần tương ứng nếu chỉ có đèn


def at(h, m=0, s=0):
    return datetime(2026, 11, 2, h, m, s, tzinfo=TZ)


def I(now, t_in=29.0, n=0.0, zones=(0, 0, 0, 0), rh=60.0):
    return HvacInputs(now=now, t_in=t_in, t_out=31.0, rh=rh, n_total=n, coverage=1.0, zone_avgs=list(zones), vision_fresh=True)


@pytest.fixture
def light(cfg):
    return LightingController(cfg, K)


@pytest.fixture
def hvac(cfg):
    return HvacController(cfg, 40, [10, 10, 10, 10], StaticSchedule.from_dict(RAW, TZ), None)


# =========================================================================== ĐÈN
def test_light_on_delay_then_fast_onset(light, cfg):
    assert cfg.light_on_delay_s == 10
    a = light.step(L(T0, n=(5, 0, 0, 0)))
    assert a.zone_state[0] == "OFF" and a.pwm[0] == 0 and any("trễ bật" in x for x in a.notes)
    b = light.step(L(T0 + 15, n=(5, 0, 0, 0)))
    assert b.zone_state[0] == "OCCUPIED" and b.pwm[0] == pytest.approx(100 * 400 / 450, abs=0.2)  # vẫn lên nhanh (Kp=1)


def test_single_tick_blip_never_turns_light_on(light):
    for i in range(20):
        n = 0.5 if i % 2 == 0 else 0.0  # nhận nhầm lẻ tẻ: vượt ngưỡng 1 nhịp rồi mất
        out = light.step(L(T0 + 15 * i, n=(n, 0, 0, 0)))
        assert out.zone_state[0] == "OFF" and out.pwm[0] == 0


def test_hysteresis_keeps_zone_occupied_between_thresholds(light, cfg, cfg_spec):
    spec = LightingController(cfg_spec, K)
    seq = [5, 5] + [0.4, 0.2] * 10  # dao động quanh ngưỡng bật 0,35
    flips_default = flips_spec = 0
    prev_d = prev_s = None
    for i, n in enumerate(seq):
        d = light.step(L(T0 + 15 * i, n=(n, 0, 0, 0))).zone_state[0]
        s = spec.step(L(T0 + 15 * i, n=(n, 0, 0, 0))).zone_state[0]
        flips_default += prev_d is not None and d != prev_d
        flips_spec += prev_s is not None and s != prev_s
        prev_d, prev_s = d, s
    assert prev_d == "OCCUPIED" and flips_default == 1  # chỉ một lần OFF -> OCCUPIED
    assert flips_spec >= 15  # không có trễ: OCCUPIED <-> GRACE mỗi nhịp
    out = light.step(L(T0 + 15 * len(seq), n=(0.1, 0, 0, 0)))  # < ngưỡng tắt 0,15
    assert out.zone_state[0] == "GRACE"


def test_lux_deadband_holds_pwm(light):
    light.step(L(T0, n=(5, 0, 0, 0)))
    on = light.step(L(T0 + 15, n=(5, 0, 0, 0)))
    p = on.pwm[0]
    for i, desk in enumerate((380, 425, 372, 429)):  # cách 400 lux không quá 30
        out = light.step(L(T0 + 30 + 15 * i, n=(5, 0, 0, 0), lux=(desk / 3.25, 0, 0, 0)))
        assert out.pwm[0] == pytest.approx(p)
    out = light.step(L(T0 + 120, n=(5, 0, 0, 0), lux=(340 / 3.25, 0, 0, 0)))  # thiếu 60 lux -> chỉnh
    assert out.pwm[0] > p


def test_pwm_step_limited_after_onset(light, cfg):
    light.step(L(T0, n=(5, 0, 0, 0)))
    on = light.step(L(T0 + 15, n=(5, 0, 0, 0)))
    sunny = light.step(L(T0 + 30, n=(5, 0, 0, 0), lux=(800 / 3.25, 0, 0, 0)))  # nắng chiếu: dư 400 lux
    assert on.pwm[0] - sunny.pwm[0] == pytest.approx(cfg.pwm_max_step, abs=0.11)


def test_occupied_pwm_range(cfg):
    cfg.pwm_occupied_min, cfg.pwm_occupied_max = 20.0, 60.0
    c = LightingController(cfg, K)
    c.step(L(T0, n=(5, 0, 0, 0)))
    assert c.step(L(T0 + 15, n=(5, 0, 0, 0))).pwm[0] == 60.0
    for i in range(10):  # trời rất sáng: vẫn không xuống dưới 20 %
        out = c.step(L(T0 + 30 + 15 * i, n=(5, 0, 0, 0), lux=(2000 / 3.25, 0, 0, 0)))
    assert out.pwm[0] == 20.0


def test_room_off_delay_separate_from_dim_delay(cfg):
    cfg.empty_dim_after_s, cfg.room_off_after_s = 60.0, 600.0
    c = LightingController(cfg, K)
    c.step(L(T0, n=(5, 0, 0, 0)))
    c.step(L(T0 + 15, n=(5, 0, 0, 0)))
    t_last = T0 + 15
    assert c.step(L(t_last + 30, n=(0, 0, 0, 0))).zone_state[0] == "GRACE"
    mid = c.step(L(t_last + 120, n=(0, 0, 0, 0)))
    assert mid.zone_state[0] == "IDLE" and 10 <= mid.pwm[0] <= 15
    assert c.step(L(t_last + 700, n=(0, 0, 0, 0))).zone_state[0] == "OFF"


# =========================================================================== ĐIỀU HÒA
def run_until(c, start, end_min, fn, step_s=30):
    """Chạy từ `start` mỗi step_s giây; fn(now) -> HvacInputs. Trả list (now, command)."""
    out = []
    t = start.timestamp()
    while t <= start.timestamp() + end_min * 60:
        now = datetime.fromtimestamp(t, TZ)
        r = c.step(fn(now))
        if r.command is not None:
            out.append((now, r.command))
        t += step_s
    return out


def test_ac_on_delay_min_on_and_off_delay(hvac, cfg):
    t_on = at(17, 0).timestamp() + cfg.ac_on_delay_s
    t_empty = at(17, 5)
    cmds = run_until(hvac, at(17, 0), 30, lambda now: I(now, n=20.0 if now < t_empty else 0.0))
    assert [c.power for _, c in cmds][:2] == [1, 0]
    (on_at, on), (off_at, off) = cmds[0], cmds[1]
    assert on.reason == "OCCUPANCY_ON" and on_at.timestamp() == pytest.approx(t_on, abs=30)
    assert off.reason == "EMPTY_ROOM"
    # tắt sau cả trễ tắt (300 s kể từ lúc trống) VÀ thời gian chạy tối thiểu (600 s kể từ lúc bật)
    assert off_at.timestamp() >= max(t_empty.timestamp() + cfg.ac_off_delay_s, on_at.timestamp() + cfg.ac_min_on_s)
    assert off_at.timestamp() <= on_at.timestamp() + cfg.ac_min_on_s + 30


def test_ac_min_off_blocks_quick_restart(hvac, cfg):
    cfg.min_cmd_interval_s = 60.0  # tách riêng ảnh hưởng của thời gian nghỉ máy nén
    hvac.ac = AcState(power=1, temp=25, fan=2, mode="cool", t=at(16, 0).timestamp(), known=True)
    hvac._power_changed_t = at(16, 0).timestamp()
    back = at(17, 9)
    cmds = run_until(hvac, at(17, 0), 25, lambda now: I(now, n=0.0 if now < back else 20.0))
    off_at = next(t for t, c in cmds if c.power == 0)
    on_at = next(t for t, c in cmds if c.power == 1)
    assert on_at.timestamp() - off_at.timestamp() >= cfg.ac_min_off_s


def test_precool_waits_for_compressor_rest_but_still_runs(hvac, cfg):
    hvac.ac = AcState(power=0, temp=25, fan=0, mode="cool", t=at(6, 22).timestamp(), known=True)
    hvac._power_changed_t = at(6, 22).timestamp()  # vừa tắt lúc 06:22
    hvac.step(I(at(6, 20)))
    assert hvac.plan is not None and hvac.plan.start_at < at(6, 27)
    r = hvac.step(I(at(6, 25)))
    assert r.command is None and any("máy nén" in n for n in r.status.notes)
    r = hvac.step(I(at(6, 27, 1)))
    assert r.command is not None and r.command.reason == "PRECOOL" and r.command.power == 1


def test_short_break_keeps_ac_on_instead_of_off_then_precool(cfg):
    raw = {"sessions": [{"name": "Sáng", "days": [2], "start": "09:45", "end": "12:15"},
                        {"name": "Chiều", "days": [2], "start": "13:00", "end": "15:30"}]}
    c = HvacController(cfg, 40, [10] * 4, StaticSchedule.from_dict(raw, TZ), None)
    c.ac = AcState(power=1, temp=25, fan=2, mode="cool", t=at(11, 0).timestamp(), known=True)
    # nghỉ trưa: bộ đệm 10 phút chỉ báo "trống" từ ~12:32, lúc đó còn 28 phút nữa vào lớp chiều
    cmds = run_until(c, at(12, 32), 28, lambda now: I(now, n=0.0))
    assert not [x for _, x in cmds if x.power == 0]  # không tắt rồi pre-cool lại sau vài phút
    c.ac = AcState(power=1, temp=26, fan=1, mode="cool", t=at(11, 0).timestamp(), known=True)  # đang để 26 °C
    c._keepon_done.clear()
    r = c.step(I(at(12, 35), n=0.0))
    assert r.command is not None and (r.command.temp, r.command.fan, r.command.reason) == (25, 3, "PRECOOL_KEEP_ON")
    cfg.ac_keep_on_before_class_min = 0
    c2 = HvacController(cfg, 40, [10] * 4, StaticSchedule.from_dict(raw, TZ), None)
    c2.ac = AcState(power=1, temp=25, fan=2, mode="cool", t=at(11, 0).timestamp(), known=True)
    cmds2 = run_until(c2, at(12, 32), 28, lambda now: I(now, n=0.0))
    assert [x.reason for _, x in cmds2][:2] == ["EMPTY_ROOM", "PRECOOL"]  # tắt bình thường thì sẽ bật lại ngay


def test_setpoint_changes_at_most_max_step(hvac, cfg):
    hvac.ac = AcState(power=1, temp=27, fan=1, mode="cool", t=at(7, 0).timestamp(), known=True)
    r = hvac.step(I(at(8, 0), t_in=25.0, n=40.0, zones=(10, 10, 10, 10)))  # tính ra ~23-24 °C
    assert r.command is not None and r.command.temp == 27 - cfg.ac_setpoint_max_step


def test_fan_hysteresis_and_range(hvac, cfg):
    hvac.ac = AcState(power=1, temp=25, fan=1, mode="cool", t=0.0, known=True)
    assert hvac._fan(8.5, False, 25.0, 25, 60.0) == 1  # vừa qua mốc 20 % (8/40): trong vùng chết, giữ mức 1
    assert hvac._fan(11.0, False, 25.0, 25, 60.0) == 2  # vượt hẳn mốc -> đổi
    cfg.fan_max = 2
    assert hvac._fan(40.0, True, 25.0, 25, 60.0) == 2  # kẹp trong dải quạt


# =========================================================================== CÀI ĐẶT LÚC ĐANG CHẠY
def test_tuning_applies_live_persists_and_rejects_bad(cfg):
    c = LightingController(cfg, K)
    tu = Tuning(cfg)
    tu.update({"light_on_delay_s": 0})
    assert c.step(L(T0, n=(5, 0, 0, 0))).zone_state[0] == "OCCUPIED"  # có hiệu lực ngay, không khởi động lại
    assert json.loads(cfg.tuning_file.read_text(encoding="utf-8")) == {"light_on_delay_s": 0.0}
    with pytest.raises(ValueError):
        tu.update({"occ_off_threshold": 0.9})  # > ngưỡng bật
    with pytest.raises(ValueError):
        tu.update({"t_min": 26, "t_max": 25})
    with pytest.raises(ValueError):
        tu.update({"mqtt_host": 1})  # không nằm trong danh sách được chỉnh
    assert cfg.occ_off_threshold == 0.15 and cfg.t_min == 22  # lô lỗi không áp dụng nửa vời
    cfg2 = AresSettings(_env_file=None, storage_root=cfg.storage_root)
    t2 = Tuning(cfg2)
    t2.load()
    assert cfg2.light_on_delay_s == 0.0
    t2.reset()
    assert cfg2.light_on_delay_s == 10.0 and json.loads(cfg2.tuning_file.read_text(encoding="utf-8")) == {}


def test_tuning_corrupt_file_is_ignored(cfg):
    cfg.tuning_file.write_text("{không phải json", encoding="utf-8")
    assert Tuning(cfg).load() == {} and cfg.light_on_delay_s == 10.0


def test_tuning_api(tmp_path):
    from ares.dashboard.app import create_app
    from ares.runtime import AresRuntime

    cfg = AresSettings(_env_file=None, storage_root=tmp_path / "a", simulation_mode=True, pca_enabled=False)
    cfg.ensure_dirs()
    rt = AresRuntime(cfg, clock=ManualClock(datetime(2026, 11, 2, 7, 5)), with_dashboard=False)
    try:
        c = TestClient(create_app(rt))
        d = c.get("/api/tuning").json()
        keys = {i["key"] for g in d["groups"] for i in g["items"]}
        assert {"light_on_delay_s", "lux_deadband", "ac_min_off_s", "deadband", "vision_fps_training"} <= keys
        assert c.post("/api/tuning", json={"values": {"ac_min_off_s": 240}}).status_code == 403  # thiếu X-ARES
        r = c.post("/api/tuning", json={"values": {"ac_min_off_s": 240}}, headers=H)
        assert r.status_code == 200 and rt.cfg.ac_min_off_s == 240 and r.json()["n_overrides"] == 1
        assert c.post("/api/tuning", json={"values": {"fan_min": 3, "fan_max": 1}}, headers=H).status_code == 409
        assert "TUNING_CHANGED" in [e["kind"] for e in rt.state.recent_events(10)]
        assert c.post("/api/tuning/reset", headers=H).json()["n_overrides"] == 0 and rt.cfg.ac_min_off_s == 300
        rt._housekeep()
        assert c.get("/api/status").json()["extra"]["fps"]["mode"] == "normal"
    finally:
        rt.al.stop(); rt.db.close()


# =========================================================================== FPS KHI TRAIN
def test_fps_governor_throttles_while_training():
    cfg = AresSettings(_env_file=None)
    flag = {"on": False}
    evs = []
    g = FpsGovernor(cfg, lambda: flag["on"], clock=ManualClock(datetime(2026, 11, 2, 0, 0)), emit=evs.append)
    assert g.vision_period() == pytest.approx(1 / 15) and g.hub_period() == pytest.approx(1 / 5)
    flag["on"] = True
    assert g.vision_period() == pytest.approx(1.0) and g.status()["mode"] == "training"
    assert g.vision_period() == pytest.approx(1.0)
    flag["on"] = False
    assert g.vision_period() == pytest.approx(1 / 15)
    assert [e.kind for e in evs] == ["FPS_THROTTLE", "FPS_RESTORED"]  # mỗi lần đổi chỉ báo một lần


def test_runtime_reports_training_fps(tmp_path):
    from ares.runtime import AresRuntime

    cfg = AresSettings(_env_file=None, storage_root=tmp_path / "a", simulation_mode=True, pca_enabled=False)
    cfg.ensure_dirs()
    rt = AresRuntime(cfg, clock=ManualClock(datetime(2026, 11, 2, 7, 5)), with_dashboard=False)
    try:
        base = rt.fps.vision_period()
        rt.retrainer.running = True
        assert rt.fps.vision_period() >= max(base, 1.0) and rt.tracker.period_fn() == rt.fps.vision_period()
        rt._housekeep()
        assert rt.state.extra["fps"]["mode"] == "training"
    finally:
        rt.retrainer.running = False
        rt.al.stop(); rt.db.close()


# =========================================================================== HOÁN ĐỔI MODEL NẠP NỀN
def test_async_model_swap_keeps_detecting_with_old_model(cfg, clock, events, tmp_path):
    from ares.buffers import OccupancyBuffer
    from ares.config import default_room
    from ares.detector import Det
    from ares.model_registry import ModelRegistry
    from ares.state import SharedState
    from ares.vision_tracker import VisionTracker
    from ares.zones import ZoneMap

    class D:
        def __init__(self, v):
            self.version = v

        def detect(self, img):
            return [Det(10, 10, 50, 90, 0.9)], 1.0

    class Src:
        exhausted = False

        def read(self):
            return np.zeros((100, 100, 3), np.uint8)

        def close(self):
            pass

    base = tmp_path / "b.pt"; base.write_bytes(b"b")
    reg = ModelRegistry(tmp_path / "m", base)
    slow = []

    def factory(p, v):
        _time.sleep(0.3)  # nạp model thật mất vài giây
        slow.append(v)
        return D(v)

    tr = VisionTracker(cfg, clock, ZoneMap(default_room()), Src(), D("base"), OccupancyBuffer(30, 4),
                       OccupancyBuffer(600, 4), SharedState(4, "P101", False), registry=reg,
                       detector_factory=factory, emit=events, async_model_load=True)
    w = tmp_path / "v.pt"; w.write_bytes(b"x")
    mv = reg.add(w, "base", "test", 1, 1, 0, {}, {}, status="candidate")
    reg.promote(mv.version)
    t0 = _time.time()
    assert tr.tick() and tr.detector.version == "base"  # nhịp này KHÔNG bị chặn bởi việc nạp model
    assert _time.time() - t0 < 0.2
    _time.sleep(0.5)
    tr.tick()
    assert tr.detector.version == mv.version and "MODEL_SWAPPED" in events.kinds() and slow == [mv.version]
