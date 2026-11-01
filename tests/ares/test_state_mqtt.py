import json
import math

import pytest

from ares.mqtt_client import MqttBus, RecordingBus, parse_sensor_payload, topics
from ares.state import Event, SensorReading, SharedState

GOOD = {"temp_in": 27.4, "temp_out": 31.2, "humidity": 68, "lux_window": 9000, "lux_zones": [110, 120, 95, 40]}


def test_sensor_payload_parsed():
    r = parse_sensor_payload(json.dumps(GOOD), 100.0, 4)
    assert r.t == 100.0 and r.temp_in == 27.4 and r.lux_zones == [110, 120, 95, 40]


@pytest.mark.parametrize("raw", [b"", b"{hong", b"[1,2]", b"null", "chuỗi bậy"])
def test_garbage_packets_return_none(raw):
    assert parse_sensor_payload(raw, 1.0, 4) is None


def test_bad_values_become_none_not_exceptions():
    bad = {"temp_in": "nan", "temp_out": 999, "humidity": -5, "lux_window": float("inf"), "lux_zones": [1, None, "x"]}
    r = SensorReading.from_payload(1.0, bad, 4)
    assert r.temp_in is None and r.temp_out is None and r.humidity is None and r.lux_window is None
    assert r.lux_zones == [1.0, None, None, None]  # thiếu vùng được đệm None, đủ n_zones


def test_json_nan_literal_from_esp32_is_handled():
    r = parse_sensor_payload('{"temp_in": NaN, "lux_zones": [100, NaN, 50, 20]}', 1.0, 4)  # json.loads chấp nhận NaN
    assert r.temp_in is None and r.lux_zones[1] is None and not any(isinstance(v, float) and math.isnan(v) for v in r.lux_zones if v is not None)


def test_alias_lux_ceiling_accepted():
    assert SensorReading.from_payload(1, {"lux_ceiling": [10, 20]}, 2).lux_zones == [10, 20]


class FakeClient:
    def __init__(self):
        self.published, self.subs = [], []

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, json.loads(payload) if payload.startswith("{") else payload, qos, retain))

    def subscribe(self, lst):
        self.subs.append(lst)


class Msg:
    def __init__(self, topic, payload):
        self.topic, self.payload = topic, payload if isinstance(payload, bytes) else payload.encode()


@pytest.fixture
def bus(cfg, clock):
    got = []
    b = MqttBus(cfg, clock, 4, on_sensor=got.append)
    b._client = FakeClient()
    b.got = got
    return b


def test_topic_contract(cfg):
    t = topics("ares", "P101")
    assert t == {"sensors": "ares/P101/sensors", "led": "ares/P101/cmd/led", "ac": "ares/P101/cmd/ac",
                 "status": "ares/P101/status", "ares_status": "ares/P101/status/ares"}


def test_sensor_message_reaches_callback_with_receive_time(bus, clock):
    bus._on_message(None, None, Msg("ares/P101/sensors", json.dumps(GOOD)))
    assert len(bus.got) == 1 and bus.got[0].t == clock.time() and bus.rx_packets == 1


def test_bad_sensor_packet_counted_not_raised(bus):
    bus._on_message(None, None, Msg("ares/P101/sensors", b"\xff\xfe rac"))
    assert bus.got == [] and bus.bad_packets == 1


def test_device_status_tracked(bus):
    bus._on_message(None, None, Msg("ares/P101/status/esp32_pwm", b"offline"))
    assert bus.device_status == {"esp32_pwm": "offline"}


def test_commands_are_retained_and_carry_timestamp(bus):
    bus.publish_led([40.04, 0, 100, 12.5], 2000)
    bus.publish_ac({"power": 1, "temp": 25, "fan": 3, "mode": "cool", "reason": "PRECOOL"})
    (t1, p1, q1, r1), (t2, p2, q2, r2) = bus._client.published
    assert t1 == "ares/P101/cmd/led" and p1["pwm"] == [40.0, 0.0, 100.0, 12.5] and p1["fade_ms"] == 2000 and r1 and "ts" in p1
    assert t2 == "ares/P101/cmd/ac" and p2["temp"] == 25 and r2 and q2 >= 1


def test_publish_before_start_is_noop(cfg, clock):
    b = MqttBus(cfg, clock, 4, on_sensor=lambda r: None)
    b.publish_led([1, 2, 3, 4], 1000)  # chưa start(): không được nổ


def test_connect_subscribes_and_announces_online(bus):
    seen = []
    bus.on_connection = seen.append
    bus._on_connect(bus._client, None, None, type("RC", (), {"is_failure": False})())
    assert bus.connected and seen == [True]
    assert any(p[0] == "ares/P101/status/ares" and p[1] == "online" and p[3] for p in bus._client.published)
    assert bus._client.subs and any("sensors" in t for t, _ in bus._client.subs[0])
    bus._on_disconnect(bus._client, None, None, "x")
    assert not bus.connected and seen == [True, False]


def test_recording_bus():
    b = RecordingBus()
    b.publish_led([1.0], 500)
    b.publish_ac({"power": 0})
    assert b.led == [([1.0], 500)] and b.ac == [{"power": 0}]


def test_shared_state_snapshot_is_a_copy_and_thread_safe():
    import threading

    s = SharedState(4, "P101", False)
    s.set_sensors(SensorReading(t=1.0, temp_in=25.0, lux_zones=[1, 2, 3, 4]))
    snap = s.snapshot()
    snap["sensors"]["temp_in"] = 99
    assert s.get_sensors().temp_in == 25.0
    errs = []

    def hammer():
        try:
            for i in range(300):
                s.add_event(Event(i, "K", "info", "m"))
                s.beat("x", i)
                s.snapshot()
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=hammer) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs and len(s.recent_events(1000)) == 300  # deque maxlen 300
