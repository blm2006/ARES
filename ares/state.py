"""Kiểu dữ liệu dùng chung giữa các luồng + trạng thái chia sẻ (thread-safe).

Quy tắc: luồng nào cũng chỉ đọc/ghi trạng thái qua SharedState (có khóa), trả về BẢN SAO.
Không luồng nào chờ luồng khác ngoài khóa ngắn này.
"""

from __future__ import annotations

import copy
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Cảm biến
# ---------------------------------------------------------------------------
def _num(v: Any) -> float | None:
    """Số hữu hạn hoặc None (chặn NaN/inf/chuỗi rác từ ESP32)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _in_range(v: float | None, lo: float, hi: float) -> float | None:
    return v if v is not None and lo <= v <= hi else None


@dataclass
class SensorReading:
    """Một gói từ ESP32 #1. Giá trị không hợp lệ được đổi thành None, không bao giờ làm hỏng bộ điều khiển."""

    t: float  # thời điểm NHẬN gói (theo Clock)
    temp_in: float | None = None
    temp_out: float | None = None
    humidity: float | None = None
    lux_window: float | None = None
    lux_zones: list[float | None] = field(default_factory=list)

    @classmethod
    def from_payload(cls, t: float, p: dict, n_zones: int) -> "SensorReading":
        lz_raw = p.get("lux_zones") or p.get("lux_ceiling") or []
        lz: list[float | None] = []
        for i in range(n_zones):
            v = _num(lz_raw[i]) if i < len(lz_raw) else None
            lz.append(_in_range(v, 0.0, 120_000.0))
        return cls(
            t=t,
            temp_in=_in_range(_num(p.get("temp_in")), -10.0, 60.0),
            temp_out=_in_range(_num(p.get("temp_out")), -10.0, 70.0),
            humidity=_in_range(_num(p.get("humidity")), 0.0, 100.0),
            lux_window=_in_range(_num(p.get("lux_window")), 0.0, 120_000.0),
            lux_zones=lz,
        )


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
@dataclass
class VisionSnapshot:
    t: float
    n_total: int = 0
    n_zones: list[int] = field(default_factory=list)
    n_unzoned: int = 0
    n_detections: int = 0
    inference_ms: float = 0.0
    fps: float = 0.0
    brightness: list[float | None] = field(default_factory=list)  # độ sáng V trung bình mỗi vùng (0..255)
    model_version: str = "base"


# ---------------------------------------------------------------------------
# Đầu ra điều khiển
# ---------------------------------------------------------------------------
@dataclass
class LightingState:
    t: float = 0.0
    pwm: list[float] = field(default_factory=list)  # %
    zone_state: list[str] = field(default_factory=list)  # OCCUPIED | GRACE | IDLE | OFF | HOLD | FAULT
    setpoint: list[float] = field(default_factory=list)  # lux mặt bàn mục tiêu
    actual: list[float | None] = field(default_factory=list)  # lux mặt bàn ước tính
    notes: list[str] = field(default_factory=list)
    daylight_lux: float | None = None  # cảm biến cửa sổ (ánh sáng tự nhiên, không lẫn LED)
    vision_mode: str = "vision"  # vision | assume_occupied | hold


@dataclass
class AcState:
    power: int = 0
    temp: int = 0
    fan: int = 0
    mode: str = "cool"
    t: float = 0.0  # lần gửi lệnh gần nhất
    known: bool = False  # IR là một chiều: đây là trạng thái ĐÃ LỆNH, không phải đo được

    def as_json(self) -> dict:
        return {"power": self.power, "temp": self.temp, "fan": self.fan}


@dataclass
class HvacStatus:
    t: float = 0.0
    mode: str = "IDLE"  # IDLE | PRECOOL | ACTIVE | FAILSAFE_OFF | ...
    setpoint_raw: float | None = None
    plan: dict | None = None
    next_eval_t: float | None = None
    r_cool: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Event:
    t: float
    kind: str
    severity: str  # info | warn | error
    message: str
    data: dict = field(default_factory=dict)


@dataclass
class Health:
    last_tick: float = 0.0
    ticks: int = 0
    errors: int = 0
    last_error: str = ""
    alive: bool = False


class SharedState:
    def __init__(self, n_zones: int, room_id: str, simulation: bool):
        self._lock = threading.RLock()
        self.n_zones = n_zones
        self.room_id = room_id
        self.simulation = simulation
        self.sensors: SensorReading | None = None
        self.vision: VisionSnapshot | None = None
        self.vision_smoothed: dict = {}
        self.hvac_smoothed: dict = {}
        self.lighting = LightingState(pwm=[0.0] * n_zones, zone_state=["OFF"] * n_zones,
                                      setpoint=[0.0] * n_zones, actual=[None] * n_zones)
        self.ac = AcState()
        self.hvac = HvacStatus()
        self.events: deque[Event] = deque(maxlen=300)
        self.health: dict[str, Health] = {}
        self.frame_jpeg: bytes | None = None
        self.frame_t: float = 0.0
        self.mqtt_connected: bool = False
        self.model_version: str = "base"
        self.al_stats: dict = {}
        self.extra: dict[str, Any] = {}

    # ---- ghi -------------------------------------------------------------
    def set_sensors(self, r: SensorReading) -> None:
        with self._lock:
            self.sensors = r

    def set_vision(self, v: VisionSnapshot, light: dict, hvac: dict) -> None:
        with self._lock:
            self.vision = v
            self.vision_smoothed = light
            self.hvac_smoothed = hvac

    def set_lighting(self, s: LightingState) -> None:
        with self._lock:
            self.lighting = s

    def set_ac(self, a: AcState) -> None:
        with self._lock:
            self.ac = a

    def set_hvac(self, h: HvacStatus) -> None:
        with self._lock:
            self.hvac = h

    def set_frame(self, jpeg: bytes, t: float) -> None:
        with self._lock:
            self.frame_jpeg = jpeg
            self.frame_t = t

    def add_event(self, e: Event) -> None:
        with self._lock:
            self.events.append(e)

    def beat(self, name: str, t: float, error: str | None = None) -> None:
        with self._lock:
            h = self.health.setdefault(name, Health())
            h.alive = True
            h.last_tick = t
            h.ticks += 1
            if error:
                h.errors += 1
                h.last_error = error

    # ---- đọc (bản sao) -----------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "room_id": self.room_id,
                "simulation": self.simulation,
                "sensors": asdict(self.sensors) if self.sensors else None,
                "vision": asdict(self.vision) if self.vision else None,
                "vision_smoothed": copy.deepcopy(self.vision_smoothed),
                "hvac_smoothed": copy.deepcopy(self.hvac_smoothed),
                "lighting": asdict(self.lighting),
                "ac": asdict(self.ac),
                "hvac": asdict(self.hvac),
                "mqtt_connected": self.mqtt_connected,
                "model_version": self.model_version,
                "al": copy.deepcopy(self.al_stats),
                "health": {k: asdict(v) for k, v in self.health.items()},
                "extra": copy.deepcopy(self.extra),
            }

    def get_sensors(self) -> SensorReading | None:
        with self._lock:
            return copy.copy(self.sensors) if self.sensors else None

    def get_ac(self) -> AcState:
        with self._lock:
            return copy.copy(self.ac)

    def get_lighting(self) -> LightingState:
        with self._lock:
            return copy.deepcopy(self.lighting)

    def get_vision(self) -> VisionSnapshot | None:
        with self._lock:
            return copy.deepcopy(self.vision) if self.vision else None

    def get_frame(self) -> tuple[bytes | None, float]:
        with self._lock:
            return self.frame_jpeg, self.frame_t

    def recent_events(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return [asdict(e) for e in list(self.events)[-limit:]][::-1]
