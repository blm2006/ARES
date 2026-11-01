"""MQTT: cầu nối giữa server ARES và hai ESP32.

  ESP32 #1 (cảm biến)  --publish-->  ares/<room>/sensors
  ARES                 --publish-->  ares/<room>/cmd/led   {"pwm":[..4 vùng..], "fade_ms":2000}
  ARES                 --publish-->  ares/<room>/cmd/ac    {"power":1,"temp":25,"fan":2,"mode":"cool"}
  ESP32 #1/#2          --publish-->  ares/<room>/status/<thiết_bị>  "online" | "offline"  (retain + LWT)
  ARES                 --publish-->  ares/<room>/status/ares        "online" | "offline"  (retain + LWT)

Lệnh gửi với retain=True để ESP32 khởi động lại vẫn nhận lại trạng thái mới nhất.
Không bao giờ chặn: kết nối chạy ở luồng riêng của paho; mất broker thì kết nối lại ngầm.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Protocol

from ares.config import AresSettings
from ares.state import SensorReading

log = logging.getLogger(__name__)


class Bus(Protocol):
    connected: bool

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def publish_led(self, pwm: list[float], fade_ms: int) -> None: ...
    def publish_ac(self, payload: dict) -> None: ...


def topics(prefix: str, room: str) -> dict[str, str]:
    base = f"{prefix}/{room}"
    return {
        "sensors": f"{base}/sensors",
        "led": f"{base}/cmd/led",
        "ac": f"{base}/cmd/ac",
        "status": f"{base}/status",
        "ares_status": f"{base}/status/ares",
    }


def parse_sensor_payload(raw: bytes | str, t: float, n_zones: int) -> SensorReading | None:
    """JSON từ ESP32 -> SensorReading. Gói hỏng trả None; giá trị ngoài dải đã được đổi thành None bên trong."""
    try:
        p = json.loads(raw)
        if not isinstance(p, dict):
            return None
    except (ValueError, TypeError):
        return None
    return SensorReading.from_payload(t, p, n_zones)


class MqttBus:
    def __init__(self, cfg: AresSettings, clock, n_zones: int,
                 on_sensor: Callable[[SensorReading], None],
                 on_connection: Callable[[bool], None] | None = None):
        self.cfg, self.clock, self.n_zones = cfg, clock, n_zones
        self.on_sensor = on_sensor
        self.on_connection = on_connection
        self.t = topics(cfg.mqtt_prefix, cfg.room_id)
        self.connected = False
        self.device_status: dict[str, str] = {}
        self.bad_packets = 0
        self.rx_packets = 0
        self._client = None
        self._last_bad_log = 0.0

    # ---------------------------------------------------------------- vòng đời
    def start(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Thiếu thư viện paho-mqtt: pip install \"paho-mqtt>=2.0\"") from e
        if not hasattr(mqtt, "CallbackAPIVersion"):  # pragma: no cover
            raise RuntimeError("paho-mqtt quá cũ (cần >= 2.0): pip install -U paho-mqtt")
        cfg = self.cfg
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{cfg.mqtt_client_id}-{cfg.room_id}")
        if cfg.mqtt_username:
            c.username_pw_set(cfg.mqtt_username, cfg.mqtt_password or None)
        c.will_set(self.t["ares_status"], "offline", qos=1, retain=True)
        c.reconnect_delay_set(min_delay=1, max_delay=30)
        c.max_queued_messages_set(50)
        c.on_connect = self._on_connect
        c.on_disconnect = self._on_disconnect
        c.on_message = self._on_message
        self._client = c
        log.info("MQTT: kết nối %s:%s (nền, tự thử lại nếu broker chưa chạy)", cfg.mqtt_host, cfg.mqtt_port)
        c.connect_async(cfg.mqtt_host, cfg.mqtt_port, keepalive=30)
        c.loop_start()

    def stop(self) -> None:
        c = self._client
        if c is None:
            return
        try:
            if self.connected:
                c.publish(self.t["ares_status"], "offline", qos=1, retain=True).wait_for_publish(timeout=2)
        except Exception:
            pass
        c.loop_stop()
        try:
            c.disconnect()
        except Exception:
            pass
        self.connected = False

    # ---------------------------------------------------------------- callback paho (VERSION2)
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):  # noqa: ANN001
        ok = not getattr(reason_code, "is_failure", False)
        if not ok:
            log.warning("MQTT: broker từ chối kết nối: %s", reason_code)
            return
        self.connected = True
        client.subscribe([(self.t["sensors"], self.cfg.mqtt_qos), (f"{self.t['status']}/#", 1)])
        client.publish(self.t["ares_status"], "online", qos=1, retain=True)
        log.info("MQTT: đã kết nối, đăng ký %s", self.t["sensors"])
        if self.on_connection:
            self.on_connection(True)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):  # noqa: ANN001
        self.connected = False
        log.warning("MQTT: mất kết nối (%s) — sẽ tự kết nối lại", reason_code)
        if self.on_connection:
            self.on_connection(False)

    def _on_message(self, client, userdata, msg):  # noqa: ANN001
        self.rx_packets += 1
        topic = msg.topic
        if topic == self.t["sensors"]:
            r = parse_sensor_payload(msg.payload, self.clock.time(), self.n_zones)
            if r is None:
                self.bad_packets += 1
                now = time.monotonic()
                if now - self._last_bad_log > 30:  # đừng ngập log khi ESP32 gửi rác liên tục
                    self._last_bad_log = now
                    log.warning("MQTT: gói cảm biến không đọc được (đã gặp %d gói hỏng)", self.bad_packets)
                return
            self.on_sensor(r)
        elif topic.startswith(self.t["status"] + "/"):
            dev = topic.rsplit("/", 1)[-1]
            try:
                self.device_status[dev] = msg.payload.decode("utf-8", "replace")[:20]
            except Exception:
                pass

    # ---------------------------------------------------------------- phát lệnh
    def _pub(self, topic: str, payload: dict) -> None:
        if self._client is None:
            return
        payload = {**payload, "ts": self.clock.now().isoformat(timespec="seconds")}
        self._client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=self.cfg.mqtt_qos, retain=True)

    def publish_led(self, pwm: list[float], fade_ms: int) -> None:
        self._pub(self.t["led"], {"pwm": [round(float(p), 1) for p in pwm], "fade_ms": int(fade_ms)})

    def publish_ac(self, payload: dict) -> None:
        self._pub(self.t["ac"], payload)


class RecordingBus:
    """Bus giả cho test: ghi lại mọi lệnh đã phát."""

    def __init__(self):
        self.connected = True
        self.led: list[tuple[list[float], int]] = []
        self.ac: list[dict] = []

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def publish_led(self, pwm: list[float], fade_ms: int) -> None:
        self.led.append((list(pwm), fade_ms))

    def publish_ac(self, payload: dict) -> None:
        self.ac.append(dict(payload))
