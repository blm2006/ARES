"""SimulationMode: thay ESP32 + camera bằng một "thế giới" giả có vật lý đơn giản.

Mục đích: chạy và kiểm thử toàn bộ Control Engine, Active Learning và Dashboard khi chưa có phần cứng.
Các thành phần giả có CÙNG giao diện với bản thật nên mã điều khiển không biết mình đang chạy giả:

    SimWorld        <->  ESP32 #1 (cảm biến) và ESP32 #2 (PWM + IR): nhiệt, ẩm, ánh sáng, điện năng
    SimBus          <->  MqttBus
    SimDetector     <->  YoloDetector   (cùng đường đi: bbox -> điểm giữa-đáy -> vùng -> bộ đệm)
    SimFrameSource  <->  CvFrameSource  (ảnh tổng hợp, sáng/tối theo độ rọi thật của thế giới)
    OccupancyScenario    người đến/đi theo lịch lớp

CẢNH BÁO: đây là mô hình để kiểm tra logic, KHÔNG phải số đo thực địa. Hệ số vật lý (UA, nhiệt dung,
công suất máy lạnh...) là giả định hợp lý cho một phòng ~60 m2, hãy chỉnh theo phòng thật nếu muốn
dùng con số điện năng để báo cáo.
"""

from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from ares.config import AresSettings
from ares.detector import Det
from ares.state import SensorReading
from ares.zones import ZoneMap, point_in_polygon


# ===========================================================================
# Người trong phòng
# ===========================================================================
@dataclass
class Person:
    x: float  # vị trí chân (chuẩn hoá 0..1)
    y: float
    arrive: float  # epoch giây
    leave: float
    zone: int


class OccupancyScenario:
    """Mỗi buổi học sinh ra một nhóm người với giờ đến/đi ngẫu nhiên (xác định theo seed)."""

    def __init__(self, cfg: AresSettings, zonemap: ZoneMap, schedule, seed: int = 42,
                 attendance: tuple[float, float] = (0.45, 0.9), cancelled_keys: set[str] | None = None,
                 visitor_prob_per_day: float = 0.0):
        self.cfg, self.zonemap, self.schedule = cfg, zonemap, schedule
        self.seed = seed
        self.attendance = attendance
        self.cancelled_keys = cancelled_keys or set()
        self.visitor_prob = visitor_prob_per_day
        self._plans: dict[str, list[Person]] = {}
        self._visitors: dict[str, list[Person]] = {}
        self.zone_weights = [0.34, 0.26, 0.22, 0.18] + [0.1] * 8

    def _rng(self, key: str) -> random.Random:
        return random.Random(f"{self.seed}:{key}")

    def _seat(self, rng: random.Random, zi: int) -> tuple[float, float]:
        poly = self.zonemap.polys[zi]
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        for _ in range(200):
            x = rng.uniform(min(xs), max(xs))
            y = rng.uniform(min(ys), max(ys))
            if point_in_polygon(x, y, poly):
                # chừa mép khung để bbox không tràn ra ngoài ảnh
                return min(0.97, max(0.03, x)), min(0.97, max(0.12, y))
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _plan(self, s) -> list[Person]:
        if s.key in self._plans:
            return self._plans[s.key]
        rng = self._rng(s.key)
        people: list[Person] = []
        if s.key not in self.cancelled_keys:
            n = int(round(rng.uniform(*self.attendance) * self.zonemap.room.n_max))
            w = self.zone_weights[: self.zonemap.n]
            for _ in range(n):
                zi = rng.choices(range(self.zonemap.n), weights=w)[0]
                x, y = self._seat(rng, zi)
                arrive = s.start.timestamp() - 600 + min(1500.0, rng.expovariate(1 / 420.0))
                stay_early = rng.random() < 0.06
                leave = s.end.timestamp() + rng.uniform(0, 420) - (rng.uniform(300, 1500) if stay_early else 0)
                people.append(Person(x, y, arrive, max(leave, arrive + 300), zi))
        self._plans[s.key] = people
        return people

    def _visitor_plan(self, day) -> list[Person]:
        key = day.isoformat()
        if key in self._visitors:
            return self._visitors[key]
        rng = self._rng("visitor:" + key)
        out: list[Person] = []
        if rng.random() < self.visitor_prob:
            start = datetime.combine(day, datetime.min.time(), tzinfo=self.cfg.tz) + timedelta(hours=rng.uniform(5.5, 21))
            zi = rng.randrange(self.zonemap.n)
            x, y = self._seat(rng, zi)
            out.append(Person(x, y, start.timestamp(), start.timestamp() + rng.uniform(300, 1800), zi))
        self._visitors[key] = out
        return out

    def people_at(self, now: datetime) -> list[Person]:
        t = now.timestamp()
        out: list[Person] = []
        for day in (now.date() - timedelta(days=1), now.date()):
            for s in self.schedule.sessions_on(day):
                out += [p for p in self._plan(s) if p.arrive <= t < p.leave]
            out += [p for p in self._visitor_plan(day) if p.arrive <= t < p.leave]
        return out


# ===========================================================================
# Thế giới vật lý
# ===========================================================================
@dataclass
class SimParams:
    ua_w_per_k: float = 150.0  # truyền nhiệt vỏ nhà
    c_j_per_k: float = 1.0e6  # nhiệt dung hiệu dụng (không khí + nội thất + tường)
    ac_capacity_w: float = 10000.0  # công suất lạnh (~2 máy 3 HP cho phòng ~60 m2, 40 người)
    ac_band_k: float = 1.0  # dải điều chế tỉ lệ quanh setpoint
    cop: float = 3.0
    fan_w: float = 80.0
    person_w: float = 100.0
    solar_gain_w: float = 500.0
    t_out_mean: float = 29.0  # ngày nóng Sài Gòn
    t_out_amp: float = 5.0
    lux_peak: float = 28000.0  # độ rọi ngoài cửa sổ lúc nắng đẹp
    daylight_depth: tuple[float, ...] = (0.030, 0.030, 0.012, 0.012)  # tỉ lệ ánh sáng ngoài vào mặt bàn từng vùng
    led_delta_max: float = 450.0  # lux mặt bàn do LED ở 100 % (= ΔL_max trong spec)
    led_gain_err: tuple[float, ...] = (1.04, 0.93, 1.07, 0.97)  # LED thật khác danh định
    k_err: tuple[float, ...] = (1.05, 0.95, 1.03, 0.97)  # k thật lệch k đã hiệu chuẩn
    sensor_noise_t: float = 0.08
    sensor_noise_lux: float = 0.02
    sensor_period_s: float = 10.0


@dataclass
class SimMetrics:
    led_wh: float = 0.0
    ac_wh: float = 0.0
    occupied_min: float = 0.0
    hot_min: float = 0.0  # có người mà T_in > 27 °C
    warm_min: float = 0.0  # có người mà T_in > 26.5 °C
    dark_zone_min: float = 0.0  # phút-vùng có người mà lux mặt bàn < 300
    ac_on_min: float = 0.0


class SimWorld:
    def __init__(self, cfg: AresSettings, clock, zonemap: ZoneMap, scenario: OccupancyScenario,
                 params: SimParams | None = None, seed: int = 42, t_in0: float | None = None):
        self.cfg, self.clock, self.zonemap, self.scenario = cfg, clock, zonemap, scenario
        self.p = params or SimParams()
        self.rng = random.Random(seed)  # nhiễu cảm biến
        self._cloud_rng = random.Random(seed + 1000)  # mây: tách riêng để hai thế giới (ARES vs baseline) có cùng thời tiết
        self.n = zonemap.n
        self.pwm = [0.0] * self.n
        self.ac = {"power": 0, "temp": 26, "fan": 1}
        self.t_in = t_in0 if t_in0 is not None else None  # None: lấy theo nhiệt độ ngoài trời ở lần chạy đầu
        self.rh_in = 68.0
        self.metrics = SimMetrics()
        self._t_last: float | None = None
        self._t_pub = -1e18
        self._cloud = 0.8
        self.on_sensor = None  # callable(SensorReading)
        self.ac_log: list[tuple[float, dict]] = []
        self.led_log: list[tuple[float, list[float]]] = []
        self.last_desk: list[float] = [0.0] * self.n
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ môi trường ngoài
    def _hour(self, now: datetime) -> float:
        return now.hour + now.minute / 60 + now.second / 3600

    def t_out(self, now: datetime) -> float:
        # đỉnh khoảng 14-15h
        return self.p.t_out_mean + self.p.t_out_amp * math.sin(2 * math.pi * (self._hour(now) - 9.0) / 24.0)

    def rh_out(self, now: datetime) -> float:
        return 72.0 + 14.0 * math.cos(2 * math.pi * (self._hour(now) - 4.0) / 24.0)

    def sun(self, now: datetime) -> float:
        h = self._hour(now)
        return max(0.0, math.sin(math.pi * (h - 6.0) / 12.0)) if 6.0 <= h <= 18.0 else 0.0

    def lux_window(self, now: datetime) -> float:
        return self.p.lux_peak * self.sun(now) * self._cloud

    def daylight_desk(self, i: int, now: datetime) -> float:
        d = self.p.daylight_depth
        return self.lux_window(now) * d[i % len(d)]

    def led_desk(self, i: int) -> float:
        g = self.p.led_gain_err
        return self.pwm[i] / 100.0 * self.p.led_delta_max * g[i % len(g)]

    def desk_lux(self, i: int, now: datetime) -> float:
        return self.daylight_desk(i, now) + self.led_desk(i)

    # ------------------------------------------------------------------ actuator (ESP32 #2)
    def apply_led(self, pwm: list[float]) -> None:
        with self._lock:
            self._apply_led(pwm)

    def _apply_led(self, pwm: list[float]) -> None:
        self.pwm = [min(100.0, max(0.0, float(p))) for p in pwm[: self.n]] + self.pwm[len(pwm):]
        self.led_log.append((self.clock.time(), list(self.pwm)))

    def apply_ac(self, payload: dict) -> None:
        with self._lock:
            self._apply_ac(payload)

    def _apply_ac(self, payload: dict) -> None:
        self.ac = {"power": int(payload.get("power", 0)), "temp": int(payload.get("temp", self.ac["temp"])),
                   "fan": int(payload.get("fan", self.ac["fan"]))}
        self.ac_log.append((self.clock.time(), dict(payload)))

    # ------------------------------------------------------------------ tiến thời gian
    def advance(self, t_now: float) -> None:
        with self._lock:
            self._advance(t_now)

    def _advance(self, t_now: float) -> None:
        if self._t_last is None:
            self._t_last = t_now
            if self.t_in is None:  # phòng đóng cửa qua đêm: ấm hơn ngoài trời một chút
                self.t_in = self.t_out(datetime.fromtimestamp(t_now, tz=self.cfg.tz)) + 1.5
        while self._t_last < t_now:
            dt = min(15.0, t_now - self._t_last)
            self._t_last += dt
            self._substep(self._t_last, dt)
            if self._t_last - self._t_pub >= self.p.sensor_period_s:
                self._t_pub = self._t_last
                if self.on_sensor is not None:
                    self.on_sensor(self.read_sensors(self._t_last))

    def _substep(self, t: float, dt: float) -> None:
        p = self.p
        now = datetime.fromtimestamp(t, tz=self.cfg.tz)
        people = self.scenario.people_at(now)
        n = len(people)
        self._cloud = min(1.0, max(0.25, self._cloud + self._cloud_rng.gauss(0, 0.01)))
        led_w = sum(pw / 100.0 * w for pw, w in zip(self.pwm, self._led_watts()))
        q_gain = p.ua_w_per_k * (self.t_out(now) - self.t_in) + p.person_w * n + p.solar_gain_w * self.sun(now) * self._cloud \
            + led_w * 0.9 + (150.0 if n else 50.0)
        q_ac = 0.0
        elec_ac = 0.0
        if self.ac["power"]:
            fan_f = {1: 0.7, 2: 0.85, 3: 1.0}.get(self.ac["fan"], 0.85)
            frac = min(1.0, max(0.0, (self.t_in - self.ac["temp"]) / p.ac_band_k))
            q_ac = p.ac_capacity_w * fan_f * frac
            elec_ac = q_ac / p.cop + p.fan_w
            self.metrics.ac_on_min += dt / 60
        self.t_in += (q_gain - q_ac) / p.c_j_per_k * dt
        rh_target = 55.0 if self.ac["power"] else 0.85 * self.rh_out(now) + 5.0
        self.rh_in += (rh_target + 0.08 * n - self.rh_in) * (dt / 1800.0)
        self.metrics.led_wh += led_w * dt / 3600.0
        self.metrics.ac_wh += elec_ac * dt / 3600.0
        self.last_desk = [self.desk_lux(i, now) for i in range(self.n)]
        if n:
            self.metrics.occupied_min += dt / 60
            if self.t_in > 27.0:
                self.metrics.hot_min += dt / 60
            if self.t_in > 26.5:
                self.metrics.warm_min += dt / 60
            occ_zones = {pe.zone for pe in people}
            self.metrics.dark_zone_min += sum(dt / 60 for z in occ_zones if self.last_desk[z] < 300.0)

    def _led_watts(self) -> list[float]:
        w = list(self.cfg.led_watts)
        return (w + [w[-1] if w else 60.0] * self.n)[: self.n]

    # ------------------------------------------------------------------ cảm biến (ESP32 #1)
    def read_sensors(self, t: float | None = None) -> SensorReading:
        t = t or self.clock.time()
        now = datetime.fromtimestamp(t, tz=self.cfg.tz)
        p = self.p
        g = self.rng.gauss
        ks = self.zonemap.k_factors()
        lux_c = []
        for i in range(self.n):
            k_true = ks[i] * p.k_err[i % len(p.k_err)]
            lux_c.append(max(0.0, self.desk_lux(i, now) / k_true * (1 + g(0, p.sensor_noise_lux))))
        return SensorReading(
            t=t, temp_in=self.t_in + g(0, p.sensor_noise_t), temp_out=self.t_out(now) + g(0, 0.1),
            humidity=min(100.0, max(0.0, self.rh_in + g(0, 0.3))),
            lux_window=max(0.0, self.lux_window(now) * (1 + g(0, 0.02))), lux_zones=lux_c,
        )


class SimBus:
    """Giao diện Bus nhưng nối thẳng vào SimWorld (không có broker)."""

    def __init__(self, world: SimWorld):
        self.world = world
        self.connected = True

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def publish_led(self, pwm: list[float], fade_ms: int) -> None:
        self.world.apply_led(pwm)

    def publish_ac(self, payload: dict) -> None:
        self.world.apply_ac(payload)


# ===========================================================================
# Camera + YOLO giả
# ===========================================================================
class SimDetector:
    """Trả detection từ người trong kịch bản, có miss/dương tính giả và conf thấp ở hàng xa.

    Dùng cùng khung ảnh với SimFrameSource để bbox khớp hình.
    """

    version = "sim"

    def __init__(self, cfg: AresSettings, clock, scenario: OccupancyScenario, seed: int = 7, fp_rate: float = 0.12):
        self.cfg, self.clock, self.scenario = cfg, clock, scenario
        self.rng = random.Random(seed)
        self.fp_rate = fp_rate

    @staticmethod
    def box_for(x: float, y: float, w: int, h: int) -> tuple[float, float, float, float]:
        bh = (0.16 + 0.26 * y) * h
        bw = 0.36 * bh
        return x * w - bw / 2, y * h - bh, x * w + bw / 2, y * h

    def detect(self, image: np.ndarray) -> tuple[list[Det], float]:
        h, w = image.shape[:2]
        now = self.clock.now()
        out: list[Det] = []
        for pe in self.scenario.people_at(now):
            if self.rng.random() > 0.80 + 0.17 * pe.y:  # hàng xa dễ bị bỏ sót hơn
                continue
            conf = min(0.99, max(0.36, 0.42 + 0.48 * pe.y + self.rng.gauss(0, 0.08)))
            x1, y1, x2, y2 = self.box_for(pe.x + self.rng.gauss(0, 0.004), pe.y + self.rng.gauss(0, 0.004), w, h)
            out.append(Det(x1, y1, x2, y2, conf, 0))
        if self.rng.random() < self.fp_rate:  # ví dụ cái ghế/áo khoác bị nhận nhầm là người
            x1, y1, x2, y2 = self.box_for(self.rng.uniform(0.1, 0.9), self.rng.uniform(0.3, 0.9), w, h)
            out.append(Det(x1, y1, x2, y2, self.rng.uniform(0.36, 0.54), 0))
        return out, round(self.rng.uniform(18, 32), 1)


class SimFrameSource:
    exhausted = False

    def __init__(self, cfg: AresSettings, clock, scenario: OccupancyScenario, world: SimWorld | None = None,
                 width: int = 640, height: int = 360):
        self.cfg, self.clock, self.scenario, self.world = cfg, clock, scenario, world
        self.w, self.h = width, height

    def read(self) -> np.ndarray | None:
        import cv2

        now = self.clock.now()
        lux = float(np.mean(self.world.last_desk)) if self.world is not None else 400.0
        base = int(35 + 170 * min(1.0, lux / 700.0))
        img = np.full((self.h, self.w, 3), base, dtype=np.uint8)
        # sàn có độ dốc nhẹ + dãy bàn
        for r in range(3):
            y = int((0.22 + 0.27 * r) * self.h)
            cv2.rectangle(img, (int(0.06 * self.w), y), (int(0.94 * self.w), y + int(0.05 * self.h)),
                          (max(0, base - 25), max(0, base - 20), max(0, base - 10)), -1)
        for pe in self.scenario.people_at(now):
            x1, y1, x2, y2 = (int(v) for v in SimDetector.box_for(pe.x, pe.y, self.w, self.h))
            bh = y2 - y1
            col = (60 + (pe.zone * 40) % 120, 90 + (pe.zone * 25) % 100, 170 - (pe.zone * 20) % 80)
            cv2.rectangle(img, (x1, y1 + bh // 4), (x2, y2), col, -1)
            cv2.circle(img, ((x1 + x2) // 2, y1 + bh // 8), max(3, bh // 8), (150, 170, 200), -1)
        return img

    def close(self) -> None: ...
