"""Serial: cầu nối giữa ARES và Arduino cắm USB (thay cho MQTT + ESP32).

  Arduino --dòng JSON mỗi ~1 s-->  {"lux_zones":[412],"lux_window":5200,"temp_in":27.1}   (cùng định dạng gói ESP32)
  ARES    --dòng lệnh-------------> P,45.0,50.0\\n   (PWM % từng vùng, đúng thứ tự zones.json)

Vòng điều khiển (daylight harvesting, trễ bật/tắt, chống nhấp nháy...) vẫn nằm trong ARES; Arduino chỉ đọc cảm biến
và xuất PWM, kèm fail-safe riêng khi mất lệnh. Điều hòa IR chưa có phần cứng: publish_ac bị bỏ qua.

Cùng giao diện với MqttBus nên runtime không phải sửa gì khác. Chạy ở luồng riêng, tự nối lại khi rút/cắm cáp.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from ares.config import AresSettings
from ares.mqtt_client import parse_sensor_payload
from ares.state import SensorReading

log = logging.getLogger(__name__)

ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x10C4, 0x0403}  # Arduino, clone CH340, CP210x, FTDI


def find_port(preferred: str = "") -> str | None:
    """Cổng chỉ định, hoặc cổng đầu tiên trông giống Arduino."""
    if preferred:
        return preferred
    from serial.tools import list_ports

    for p in list_ports.comports():
        if p.vid in ARDUINO_VIDS or "arduino" in (p.description or "").lower():
            return p.device
    return None


class SerialBus:
    def __init__(self, cfg: AresSettings, clock, n_zones: int,
                 on_sensor: Callable[[SensorReading], None],
                 on_connection: Callable[[bool], None] | None = None):
        self.cfg, self.clock, self.n_zones = cfg, clock, n_zones
        self.on_sensor = on_sensor
        self.on_connection = on_connection
        self.connected = False
        self.device_status: dict[str, str] = {}
        self.bad_packets = 0
        self.rx_packets = 0
        self._ser = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_bad_log = 0.0
        self._ac_warned = False

    # ---------------------------------------------------------------- vòng đời
    def start(self) -> None:
        try:
            import serial  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Thiếu thư viện pyserial: pip install pyserial") from e
        self._thread = threading.Thread(target=self._run, name="ares-serial", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._close()

    def _set_connected(self, ok: bool) -> None:
        if ok != self.connected:
            self.connected = ok
            self.device_status["arduino"] = "online" if ok else "offline"
            if self.on_connection:
                self.on_connection(ok)

    def _close(self) -> None:
        with self._lock:
            ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        self._set_connected(False)

    # ---------------------------------------------------------------- luồng đọc
    def _run(self) -> None:
        import serial

        while not self._stop.is_set():
            port = None
            try:
                port = find_port(self.cfg.serial_port)
                if port is None:
                    log.warning("Serial: không thấy Arduino. Cắm cáp USB hoặc đặt ARES_SERIAL_PORT=COMx (thử lại sau 5 s)")
                    self._stop.wait(5)
                    continue
                ser = serial.Serial(port, self.cfg.serial_baud, timeout=0.5, write_timeout=1.0)
                self._stop.wait(2.0)  # Arduino tự reset khi mở cổng; đợi bootloader xong
                ser.reset_input_buffer()
                with self._lock:
                    self._ser = ser
                log.info("Serial: đã mở %s @ %d", port, self.cfg.serial_baud)
                self._set_connected(True)
                self._read_loop(ser)
            except Exception as e:  # noqa: BLE001 - rút cáp, cổng bận... đều thử lại
                log.warning("Serial: lỗi cổng %s (%s: %s) — thử lại sau 3 s", port, type(e).__name__, e)
            finally:
                self._close()
            self._stop.wait(3)

    def _read_loop(self, ser) -> None:
        while not self._stop.is_set():
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("{"):
                continue  # dòng debug / chữ khởi động của sketch
            self.rx_packets += 1
            r = parse_sensor_payload(line, self.clock.time(), self.n_zones)
            if r is None:
                self.bad_packets += 1
                now = time.monotonic()
                if now - self._last_bad_log > 30:
                    self._last_bad_log = now
                    log.warning("Serial: gói cảm biến không đọc được (đã gặp %d gói hỏng)", self.bad_packets)
                continue
            self.on_sensor(r)

    # ---------------------------------------------------------------- phát lệnh
    def _write(self, text: str) -> None:
        with self._lock:
            ser = self._ser
            if ser is None:
                return
            try:
                ser.write(text.encode("ascii"))
            except Exception as e:  # noqa: BLE001
                log.warning("Serial: ghi lệnh lỗi (%s) — luồng đọc sẽ nối lại", e)

    def publish_led(self, pwm: list[float], fade_ms: int) -> None:
        self._write("P," + ",".join(f"{max(0.0, min(100.0, float(p))):.1f}" for p in pwm) + "\n")

    def publish_ac(self, payload: dict) -> None:
        if not self._ac_warned:
            self._ac_warned = True
            log.info("Serial: chưa có phần cứng điều hòa (IR) trên Arduino — bỏ qua lệnh điều hòa.")
