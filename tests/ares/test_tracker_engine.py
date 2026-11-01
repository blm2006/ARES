from datetime import datetime, timedelta

import numpy as np
import pytest

from ares.buffers import OccupancyBuffer
from ares.clock import get_tz
from ares.config import default_room
from ares.control_engine import ControlEngine
from ares.detector import Det
from ares.model_registry import BASE, ModelRegistry
from ares.mqtt_client import RecordingBus
from ares.schedule import StaticSchedule
from ares.state import SensorReading, SharedState
from ares.vision_tracker import VisionTracker
from ares.zones import ZoneMap

TZ = get_tz("Asia/Ho_Chi_Minh")
W, H = 400, 200


class Src:
    exhausted = False

    def __init__(self):
        self.frames = []
        self.closed = False

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def close(self):
        self.closed = True


class Detector:
    def __init__(self, dets=None, version="base"):
        self.dets, self.version = dets or [], version

    def detect(self, img):
        return list(self.dets), 12.5


def img():
    return np.full((H, W, 3), 100, np.uint8)


def person(x, y_bottom, conf=0.9):
    return Det(x - 10, y_bottom - 80, x + 10, y_bottom, conf)


@pytest.fixture
def parts(cfg, clock, events):
    zm = ZoneMap(default_room())
    st = SharedState(4, "P101", False)
    lb, hb = OccupancyBuffer(30, 4), OccupancyBuffer(600, 4)
    src, det = Src(), Detector()
    tr = VisionTracker(cfg, clock, zm, src, det, lb, hb, st, emit=events)
    return tr, src, det, st, lb, hb


def test_tick_counts_by_foot_point_and_feeds_both_buffers(parts, clock):
    tr, src, det, st, lb, hb = parts
    det.dets = [person(50, 90), person(300, 90), person(50, 190), person(300, 190), person(310, 195)]
    src.frames = [img()]
    assert tr.tick()
    v = st.get_vision()
    assert v.n_total == 5 and v.n_zones == [1, 1, 1, 2] and v.inference_ms == 12.5
    clock.advance(1)
    assert lb.stats(clock.time()).mean_total == 5 and hb.stats(clock.time()).mean_total == 5
    assert st.get_frame()[0].startswith(b"\xff\xd8")  # ảnh overlay JPEG cho dashboard


def test_detections_below_count_threshold_are_not_counted(parts, cfg):
    tr, src, det, st, *_ = parts
    cfg.count_conf_min = 0.5
    det.dets = [person(50, 90, 0.45), person(300, 90, 0.8)]
    src.frames = [img()]
    tr.tick()
    assert st.get_vision().n_total == 1


def test_non_person_class_ignored(parts):
    tr, src, det, st, *_ = parts
    det.dets = [Det(10, 10, 50, 90, 0.9, cls=2)]
    src.frames = [img()]
    tr.tick()
    assert st.get_vision().n_total == 0


def test_offers_frame_to_active_learning_without_blocking(parts, cfg, clock, db):
    from ares.active_learner import ActiveLearner

    tr, src, det, st, *_ = parts
    tr.al = ActiveLearner(cfg, clock, db)  # chưa start: chỉ vào hàng đợi
    det.dets = [person(50, 90, 0.45)]
    src.frames = [img()]
    tr.tick()
    assert tr.al.q.qsize() == 1


def test_camera_loss_emits_stale_once_then_restored(parts, cfg, clock, events):
    tr, src, *_ = parts
    n = int(cfg.frame_stale_s / cfg.vision_interval_s) + 2
    for _ in range(n + 5):
        assert tr.tick() is False
    assert events.kinds().count("VISION_STALE") == 1
    src.frames = [img()]
    assert tr.tick() is True
    assert "VISION_RESTORED" in events.kinds()


def test_camera_loss_leaves_old_buffer_to_expire_not_to_read_as_empty(parts, clock):
    tr, src, det, st, lb, hb = parts
    det.dets = [person(50, 90)]
    src.frames = [img()]
    tr.tick()
    clock.advance(120)  # camera chết 2 phút
    assert not lb.stats(clock.time()).ok  # "không có dữ liệu", chứ không phải "0 người"


def test_brightness_per_zone(parts):
    tr, src, det, st, *_ = parts
    im = img()
    im[:, :200] = 220  # nửa trái sáng
    src.frames = [im]
    tr.tick()
    b = st.get_vision().brightness
    assert b[0] > 200 and b[1] < 120 and b[2] > 200 and b[3] < 120


def test_hot_swap_to_promoted_model(cfg, clock, events, tmp_path, parts):
    tr, src, det, st, *_ = parts
    base = tmp_path / "b.pt"; base.write_bytes(b"b")
    reg = ModelRegistry(tmp_path / "m", base)
    built = []
    tr.registry, tr.detector_factory = reg, lambda p, v: built.append((p, v)) or Detector(version=v)
    src.frames = [img(), img(), img()]
    tr.tick()
    assert tr.detector.version == "base" and built == []
    w = tmp_path / "w.pt"; w.write_bytes(b"w")
    reg.add(w, BASE, "midnight", 1, 1, 0, {}, {}); reg.promote("v0001")
    tr.tick()
    assert tr.detector.version == "v0001" and "MODEL_SWAPPED" in events.kinds() and st.model_version == "v0001"
    tr.tick()
    assert len(built) == 1  # không nạp lại mỗi nhịp


def test_failed_model_load_keeps_old_model_and_does_not_retry_forever(cfg, clock, events, tmp_path, parts):
    tr, src, det, st, *_ = parts
    base = tmp_path / "b.pt"; base.write_bytes(b"b")
    reg = ModelRegistry(tmp_path / "m", base)
    w = tmp_path / "w.pt"; w.write_bytes(b"w")
    reg.add(w, BASE, "x", 1, 1, 0, {}, {}); reg.promote("v0001")
    calls = []

    def factory(p, v):
        calls.append(v)
        raise RuntimeError("file .pt hỏng")

    tr.registry, tr.detector_factory = reg, factory
    src.frames = [img()] * 5
    for _ in range(4):
        tr.tick()
    assert tr.detector.version == "base" and calls == ["v0001"] and events.kinds().count("MODEL_LOAD_FAILED") == 1


# ============================================================================= ControlEngine
@pytest.fixture
def eng(cfg, clock, events):
    zm = ZoneMap(default_room())
    st = SharedState(4, "P101", False)
    lb, hb = OccupancyBuffer(30, 4), OccupancyBuffer(600, 4)
    sch = StaticSchedule.from_dict({"sessions": [{"name": "A", "days": [2], "start": "07:00", "end": "09:30"}]}, TZ)
    bus = RecordingBus()
    e = ControlEngine(cfg, clock, st, zm, lb, hb, sch, None, bus, None)
    e.emit = events  # thu sự kiện
    return e, st, lb, hb, bus


def feed(clock, lb, hb, n, zones, seconds=30, step=2):
    for _ in range(seconds // step):
        clock.advance(step)
        lb.push(clock.time(), n, zones)
        hb.push(clock.time(), n, zones)


def sensors(clock, lux=100.0, t_in=28.0):
    return SensorReading(t=clock.time(), temp_in=t_in, temp_out=31.0, humidity=60.0, lux_window=5000.0, lux_zones=[lux] * 4)


def test_lighting_tick_publishes_pwm_and_updates_state(eng, clock):
    eng[0].cfg.light_on_delay_s = 0.0  # công thức gốc, không trễ bật
    e, st, lb, hb, bus = eng
    feed(clock, lb, hb, 8, [2, 2, 2, 2])
    st.set_sensors(sensors(clock, lux=30.0))
    out = e.lighting_tick()
    assert bus.led and bus.led[-1][0] == out.pwm and max(out.pwm) > 50 and st.get_lighting().zone_state == ["OCCUPIED"] * 4


def test_no_republish_if_unchanged_until_heartbeat(eng, clock, cfg):
    e, st, lb, hb, bus = eng
    feed(clock, lb, hb, 8, [2, 2, 2, 2])
    st.set_sensors(sensors(clock, lux=400 / 3.25))  # đã đúng 400 lux
    for _ in range(3):
        e.lighting_tick(); clock.advance(15); lb.push(clock.time(), 8, [2, 2, 2, 2]); st.set_sensors(sensors(clock, lux=400 / 3.25))
    n = len(bus.led)
    clock.advance(cfg.led_heartbeat_s + 1)
    lb.push(clock.time(), 8, [2, 2, 2, 2]); st.set_sensors(sensors(clock, lux=400 / 3.25))
    e.lighting_tick()
    assert len(bus.led) == n + 1  # nhịp tim: gửi lại để ESP32 vừa khởi động lại vẫn đồng bộ


def test_stale_sensor_packets_are_not_trusted(eng, clock, cfg):
    e, st, lb, hb, bus = eng
    feed(clock, lb, hb, 8, [2, 2, 2, 2])
    st.set_sensors(sensors(clock, lux=130.0))
    clock.advance(cfg.sensor_stale_s + 5)
    lb.push(clock.time(), 8, [2, 2, 2, 2])
    out = e.lighting_tick()
    assert out.actual == [None] * 4 and any("Mất gói cảm biến" in n for n in out.notes)


def test_hvac_tick_precool_publishes_ir_command(eng, clock):
    e, st, lb, hb, bus = eng
    clock.set(datetime(2026, 11, 2, 6, 40))
    st.set_sensors(sensors(clock, t_in=29.0))
    feed(clock, lb, hb, 0, [0, 0, 0, 0], seconds=60)
    e.hvac_tick()  # lập kế hoạch
    clock.advance(5 * 60)
    st.set_sensors(sensors(clock, t_in=29.0))
    lb.push(clock.time(), 0, [0, 0, 0, 0]); hb.push(clock.time(), 0, [0, 0, 0, 0])
    e.hvac_tick()
    assert bus.ac and bus.ac[0]["power"] == 1 and bus.ac[0]["reason"] == "PRECOOL"
    assert st.get_ac().power == 1 and st.snapshot()["hvac"]["plan"]["class"] == "A"


def test_hvac_without_sensor_packets_does_not_crash_or_command(eng, clock):
    e, st, lb, hb, bus = eng
    feed(clock, lb, hb, 8, [2, 2, 2, 2], seconds=60)
    res = e.hvac_tick()
    assert res.command is None and any("Mất gói cảm biến" in n for n in res.status.notes)


def test_learned_cooling_rate_is_saved_to_habit_store(eng, clock, cfg, events):
    class Habit:
        saved = []
        def get(self, d, s): return None
        def set_cooling_rate(self, d, s, r): self.saved.append((d, s, r))
        def mean_cooling_rate(self): return None
    e, st, lb, hb, bus = eng
    e.habit = Habit(); e.hvac.habit = e.habit
    clock.set(datetime(2026, 11, 2, 6, 10)); st.set_sensors(sensors(clock, t_in=29.0)); feed(clock, lb, hb, 0, [0] * 4, 30)
    e.hvac_tick()
    for _ in range(40):
        clock.advance(60); lb.push(clock.time(), 0, [0] * 4); hb.push(clock.time(), 0, [0] * 4)
        st.set_sensors(sensors(clock, t_in=29.0 - 0.08 * max(0, (clock.time() - datetime(2026, 11, 2, 6, 24, tzinfo=TZ).timestamp()) / 60)))
        e.hvac_tick()
    assert Habit.saved and "R_COOL_UPDATED" in events.kinds()
