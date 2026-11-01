"""Control Engine: điều khiển đèn (Thread 2) và điều hòa (Thread 3).

Hai bộ điều khiển (`LightingController`, `HvacController`) là HÀM THUẦN theo nghĩa "không I/O":
nhận đầu vào + thời gian, trả lệnh. Không MQTT, không DB, không sleep => test được bằng
đồng hồ giả. `ControlEngine` bên dưới mới là lớp keo nối chúng với cảm biến, MQTT và DB.

Các chỗ spec không nói rõ và cách chọn (đều chỉnh được trong config):
  * "N_zone > 0" dùng trung bình 30 giây >= occ_on_threshold (0.35), không phải >0 tuyệt đối,
    để một khung hình dương tính giả không bật đèn cả vùng.
  * Trống 30 giây đến 3 phút = GRACE: giữ nguyên mục tiêu cũ (người có thể quay lại).
  * Vùng trống > 3 phút: 100 lux nền nhưng PWM kẹp 10-15 %. Cả phòng trống > 3 phút: 0 %.
  * Bước đầu khi vùng vừa có người dùng Kp=1.0 (lên đèn nhanh), sau đó Kp=0.5 như spec.
  * Điều hòa là điều khiển HỞ qua IR: không đọc lại được trạng thái máy, `AcState` là trạng thái
    ĐÃ LỆNH.

CHỐNG BẬT/TẮT LIÊN TỤC (mọi con số chỉnh được trên dashboard, tab "Thiết bị"):
  Đèn     * hiện diện có TRỄ (hysteresis): bật khi trung bình 30 s >= occ_on_threshold, chỉ hết khi < occ_off_threshold
          * trễ BẬT light_on_delay_s, trễ HẠ empty_dim_after_s, trễ TẮT CẢ PHÒNG room_off_after_s
          * VÙNG CHẾT lux_deadband: lux đo đã đủ gần lux đích thì giữ PWM; mỗi nhịp đổi tối đa pwm_max_step %
          * dải điều chỉnh khi có người [pwm_occupied_min, pwm_occupied_max]; lệnh MQTT chỉ gửi khi đổi >= led_publish_min_delta
  Điều hòa * deadband ±0,7 °C quanh setpoint đang đặt; giãn cách tối thiểu giữa 2 lệnh IR min_cmd_interval_s
          * máy nén: đã bật thì chạy >= ac_min_on_s mới tắt vì phòng trống; đã tắt thì nghỉ >= ac_min_off_s mới bật lại
            (kể cả pre-cool — pre-cool chỉ lùi lại vài phút, không bị huỷ)
          * trễ BẬT theo hiện diện ac_on_delay_s, trễ TẮT khi phòng trống ac_off_delay_s; không tắt nếu lớp kế tiếp
            bắt đầu trong ac_keep_on_before_class_min phút (giờ giải lao ngắn)
          * setpoint đổi tối đa ac_setpoint_max_step °C/lệnh, trong [t_min, t_max]; quạt trong [fan_min, fan_max] và có
            vùng chết fan_hysteresis quanh các mốc đổi mức
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Protocol

from ares.config import AresSettings
from ares.formulas import (
    clamp,
    cooling_rate_sample,
    deadband_pass,
    desk_lux,
    ema,
    fan_level,
    lux_setpoint,
    pwm_update,
    t_lead_minutes,
    t_setpoint,
)
from ares.state import AcState, Event, HvacStatus, LightingState

log = logging.getLogger(__name__)

NEG_INF = -math.inf


# ===========================================================================
# THREAD 2 — ĐÈN
# ===========================================================================
@dataclass
class LightInputs:
    t: float
    n_avg: list[float] | None  # trung bình 30 s mỗi vùng; None nếu không có dữ liệu vision
    vision_ok: bool
    in_session: bool
    lux_ceiling: list[float | None]  # None = cảm biến mất/lỗi
    lux_window: float | None = None


class LightingController:
    def __init__(self, cfg: AresSettings, k: list[float]):
        self.cfg = cfg
        self.k = list(k)
        self.n = len(k)
        self.pwm = [0.0] * self.n
        self.state = ["OFF"] * self.n
        self.last_occ = [NEG_INF] * self.n
        self.prev_sp = [cfg.lux_occupied] * self.n
        self.invalid_since: list[float | None] = [None] * self.n
        self.occ = [False] * self.n  # trạng thái hiện diện ĐÃ QUA trễ/hysteresis
        self.on_since: list[float | None] = [None] * self.n  # lần đầu vượt ngưỡng bật (đang đếm trễ bật)

    def _occupancy(self, i: int, avg: float, t: float) -> bool:
        """Hiện diện của vùng i có trễ bật + hysteresis, để đèn không nhấp nháy khi số đếm dao động quanh ngưỡng."""
        cfg = self.cfg
        if self.occ[i]:
            if avg < cfg.occ_off_threshold:
                self.occ[i] = False
        elif avg >= cfg.occ_on_threshold:
            if self.on_since[i] is None:
                self.on_since[i] = t
            if t - self.on_since[i] >= cfg.light_on_delay_s:
                self.occ[i] = True
        if self.occ[i] or avg < cfg.occ_on_threshold:
            self.on_since[i] = None
        return self.occ[i]

    def step(self, inp: LightInputs) -> LightingState:
        cfg, n, t = self.cfg, self.n, inp.t
        notes: list[str] = []

        # ---- hiện diện từng vùng ---------------------------------------------------
        if inp.vision_ok and inp.n_avg is not None:
            occ = [self._occupancy(i, inp.n_avg[i], t) for i in range(n)]
            n_avg = inp.n_avg
            vmode = "vision"
            waiting = [i + 1 for i in range(n) if self.on_since[i] is not None]
            if waiting:
                notes.append(f"Khu {', '.join(map(str, waiting))}: thấy người, đang chờ trễ bật {cfg.light_on_delay_s:g} s.")
        elif inp.in_session:
            # Camera mất giữa giờ học: thà thừa điện còn hơn để lớp học tối đen.
            occ = [True] * n
            self.occ = [True] * n
            self.on_since = [None] * n
            n_avg = [0.0] * n
            vmode = "assume_occupied"
            notes.append("Mất dữ liệu camera trong giờ học: coi mọi vùng đang có người.")
        else:
            # Ngoài giờ và mất camera: giữ nguyên, không đoán.
            notes.append("Mất dữ liệu camera ngoài giờ học: giữ nguyên đèn.")
            return LightingState(
                t=t, pwm=[round(p, 1) for p in self.pwm], zone_state=["HOLD"] * n,
                setpoint=list(self.prev_sp), actual=[self._actual(i, inp) for i in range(n)], notes=notes,
                daylight_lux=inp.lux_window, vision_mode="hold",
            )

        for i in range(n):
            if occ[i]:
                self.last_occ[i] = t
        empty_for = [t - self.last_occ[i] for i in range(n)]
        room_empty_for = min(empty_for)

        new_pwm = list(self.pwm)
        new_state = list(self.state)
        sps = list(self.prev_sp)
        actuals: list[float | None] = []

        for i in range(n):
            prev_state = self.state[i]
            actual = self._actual(i, inp)
            actuals.append(actual)
            pwm_min, pwm_max = 0.0, 100.0

            if occ[i]:
                st = "OCCUPIED"
                pwm_min, pwm_max = cfg.pwm_occupied_min, cfg.pwm_occupied_max
                sp = lux_setpoint(n_avg[i], cfg.lux_occupied, cfg.lux_dense, cfg.lux_idle,
                                  cfg.dense_people, occupied=True)
                self.prev_sp[i] = sp
            elif empty_for[i] <= cfg.empty_dim_after_s:
                st = "GRACE"
                sp = self.prev_sp[i]
                # Người đã rời khỏi: được giảm đèn nếu trời sáng hơn, nhưng KHÔNG được thêm đèn
                # (che cảm biến lúc này không được mở đèn lên, nếu không đèn sẽ bật rồi lại tắt khi hết trễ).
                pwm_max = self.pwm[i]
            elif room_empty_for > cfg.room_off_after_s:
                st = "OFF"
                sp = 0.0
            else:
                st = "IDLE"
                sp = cfg.lux_idle
                pwm_min, pwm_max = cfg.idle_pwm_min, cfg.idle_pwm_max
            sps[i] = sp
            new_state[i] = st

            if st == "OFF":
                new_pwm[i] = 0.0
                self.invalid_since[i] = None
                continue

            if actual is None:  # cảm biến lux của vùng này không dùng được
                if self.invalid_since[i] is None:
                    self.invalid_since[i] = t
                dur = t - self.invalid_since[i]
                if dur >= cfg.sensor_fault_s:
                    if st == "OCCUPIED":
                        new_pwm[i] = max(self.pwm[i], cfg.fallback_pwm_occupied)
                    elif st == "IDLE":
                        new_pwm[i] = cfg.idle_pwm_min
                    notes.append(f"Khu {i + 1}: cảm biến lux lỗi {dur / 60:.0f} phút -> PWM dự phòng.")
                else:
                    new_pwm[i] = clamp(self.pwm[i], pwm_min, pwm_max)
                continue

            self.invalid_since[i] = None
            onset = st == "OCCUPIED" and prev_state in ("OFF", "IDLE", "HOLD")
            kp = cfg.kp_onset if onset else cfg.kp
            if not onset and abs(sp - actual) <= cfg.lux_deadband:
                new_pwm[i] = clamp(self.pwm[i], pwm_min, pwm_max)  # trong vùng chết: giữ nguyên
                continue
            raw = pwm_update(self.pwm[i], sp, actual, kp, cfg.dl_max, 0.0, 100.0)
            if not onset:  # bước đầu khi vừa có người được lên thẳng (đèn sáng nhanh), sau đó đổi từ từ
                raw = clamp(raw, self.pwm[i] - cfg.pwm_max_step, self.pwm[i] + cfg.pwm_max_step)
            new_pwm[i] = clamp(raw, pwm_min, pwm_max)

        self.pwm = new_pwm
        self.state = new_state
        return LightingState(
            t=t, pwm=[round(p, 1) for p in new_pwm], zone_state=list(new_state),
            setpoint=sps, actual=[None if a is None else round(a, 1) for a in actuals], notes=notes,
            daylight_lux=inp.lux_window, vision_mode=vmode,
        )

    def _actual(self, i: int, inp: LightInputs) -> float | None:
        lc = inp.lux_ceiling[i] if i < len(inp.lux_ceiling) else None
        return None if lc is None else desk_lux(lc, self.k[i])


# ===========================================================================
# THREAD 3 — ĐIỀU HÒA
# ===========================================================================
@dataclass
class HabitInfo:
    occ_probability: float
    expected_headcount: int
    avg_cooling_rate: float | None
    n_obs: int


class HabitProvider(Protocol):
    def get(self, dow: int, slot: time) -> HabitInfo | None: ...
    def set_cooling_rate(self, dow: int, slot: time, rate: float) -> None: ...
    def mean_cooling_rate(self) -> float | None: ...


@dataclass
class HvacInputs:
    now: datetime
    t_in: float | None
    t_out: float | None
    rh: float | None
    n_total: float | None  # trung bình 10 phút; None nếu không có dữ liệu
    coverage: float  # phần cửa sổ 10 phút thực sự có dữ liệu
    zone_avgs: list[float] | None
    vision_fresh: bool  # camera còn sống (không phải số liệu cũ)


@dataclass
class AcCommand:
    power: int
    temp: int
    fan: int
    mode: str
    reason: str
    urgent: bool = False  # bỏ qua giãn cách tối thiểu giữa các lệnh

    def payload(self) -> dict:
        return {"power": self.power, "temp": self.temp, "fan": self.fan, "mode": self.mode, "reason": self.reason}


@dataclass
class Arrival:
    key: str
    start: datetime
    end: datetime | None
    name: str
    source: str  # schedule | habit
    p: float


@dataclass
class PreCoolPlan:
    arrival: Arrival
    t_lead_min: float
    start_at: datetime
    r_cool: float
    t_in_plan: float | None
    started: bool = False
    started_at: datetime | None = None
    t_in_start: float | None = None
    eval_at: datetime | None = None
    evaluated: bool = False
    valid_for_rate: bool = True

    def to_dict(self) -> dict:
        return {
            "key": self.arrival.key,
            "class": self.arrival.name,
            "class_start": self.arrival.start.isoformat(),
            "source": self.arrival.source,
            "p_occ": round(self.arrival.p, 3),
            "t_lead_min": round(self.t_lead_min, 1),
            "start_at": self.start_at.isoformat(),
            "r_cool": round(self.r_cool, 4),
            "started": self.started,
            "evaluated": self.evaluated,
        }


@dataclass
class HvacResult:
    command: AcCommand | None
    events: list[Event]
    status: HvacStatus
    cooling_rate_update: tuple[int, time, float] | None = None


class HvacController:
    def __init__(self, cfg: AresSettings, n_max: int, zone_caps: list[int], schedule, habit: HabitProvider | None):
        self.cfg = cfg
        self.n_max = n_max
        self.zone_caps = zone_caps
        self.schedule = schedule
        self.habit = habit
        self.ac = AcState(mode=cfg.ac_mode)
        self.plan: PreCoolPlan | None = None
        self.decided: set[str] = set()
        self.failsafe_done: set[str] = set()
        self.cancelled: set[str] = set()
        self.next_eval_at: float = 0.0
        self._session_key: str | None = None
        self._warned: set[str] = set()
        self._noted: set[str] = set()
        self._last_raw: float | None = None
        self.r_cool_used: float | None = None
        self._power_changed_t: float = NEG_INF  # lần cuối đổi BẬT<->TẮT (bảo vệ máy nén)
        self._on_cond_since: float | None = None  # "có người + nóng" bắt đầu giữ liên tục từ lúc này
        self._empty_since: float | None = None  # "phòng trống, ngoài giờ học" bắt đầu giữ liên tục từ lúc này
        self._keepon_done: set[str] = set()  # buổi đã được "giữ máy + hạ về nhiệt độ pre-cool"

    # ---------------------------------------------------------------- tiện ích
    def _slot(self, dt: datetime) -> time:
        m = (dt.minute // self.cfg.habit_slot_min) * self.cfg.habit_slot_min
        return time(dt.hour, m)

    def _ev(self, t: float, kind: str, sev: str, msg: str, **data) -> Event:
        return Event(t=t, kind=kind, severity=sev, message=msg, data=data)

    def _send(self, now: datetime, power: int, temp: int, fan: int, reason: str, urgent: bool) -> AcCommand:
        cfg = self.cfg
        if power == 1:
            fan = int(clamp(fan, cfg.fan_min, cfg.fan_max))
            temp = int(clamp(temp, cfg.t_min, cfg.t_max))
        cmd = AcCommand(power=power, temp=temp, fan=fan, mode=self.cfg.ac_mode, reason=reason, urgent=urgent)
        if power != self.ac.power or not self.ac.known:
            self._power_changed_t = now.timestamp()
        self.ac = AcState(power=power, temp=temp, fan=fan, mode=self.cfg.ac_mode, t=now.timestamp(), known=True)
        return cmd

    def _throttled(self, now: datetime) -> bool:
        return self.ac.known and (now.timestamp() - self.ac.t) < self.cfg.min_cmd_interval_s

    def _off_wait_s(self, now: datetime) -> float:
        """Còn phải chờ bao nhiêu giây nữa mới được BẬT lại (máy vừa tắt chưa đủ ac_min_off_s)."""
        if self.ac.power == 1:
            return 0.0
        return max(0.0, self.cfg.ac_min_off_s - (now.timestamp() - self._power_changed_t))

    def _on_wait_s(self, now: datetime) -> float:
        """Còn phải chờ bao nhiêu giây nữa mới được TẮT (máy vừa bật chưa đủ ac_min_on_s)."""
        if self.ac.power == 0:
            return 0.0
        return max(0.0, self.cfg.ac_min_on_s - (now.timestamp() - self._power_changed_t))

    def _fan(self, n_total: float, crowded: bool, t_in: float | None, sp: float, rh: float | None) -> int:
        """Mức quạt có vùng chết: chỉ đổi khi tỉ lệ lấp đầy vượt hẳn mốc (± fan_hysteresis), rồi kẹp [fan_min, fan_max]."""
        cfg = self.cfg
        new = fan_level(n_total, self.n_max, crowded, t_in, sp, rh, cfg.rh_high)
        prev = self.ac.fan if (self.ac.known and self.ac.power == 1) else None
        if prev is not None and cfg.fan_hysteresis > 0:
            m = cfg.fan_hysteresis * self.n_max
            lo = fan_level(max(0.0, n_total - m), self.n_max, crowded, t_in, sp, rh, cfg.rh_high)
            hi = fan_level(n_total + m, self.n_max, crowded, t_in, sp, rh, cfg.rh_high)
            if lo <= prev <= hi:
                new = prev
        return int(clamp(new, cfg.fan_min, cfg.fan_max))

    def _r_cool(self, start: datetime) -> float:
        info = self.habit.get(start.isoweekday() + 1, self._slot(start)) if self.habit else None
        if info and info.avg_cooling_rate and info.avg_cooling_rate > 0:
            return info.avg_cooling_rate
        mean = self.habit.mean_cooling_rate() if self.habit else None
        return mean if mean and mean > 0 else self.cfg.r_cool_default

    def _occ_probability(self, start: datetime) -> float:
        info = self.habit.get(start.isoweekday() + 1, self._slot(start)) if self.habit else None
        if info is not None and info.n_obs >= self.cfg.habit_min_obs:
            return info.occ_probability
        return self.cfg.schedule_prior_prob

    def _next_arrival(self, now: datetime) -> Arrival | None:
        cfg = self.cfg
        # Nhìn trước tối thiểu bằng t_lead tối đa: nếu chỉ nhìn 30 phút mà phòng cần 45 phút để mát thì kế hoạch lập quá muộn.
        look_ahead = max(cfg.precool_horizon_min, cfg.lead_max_min)
        s = self.schedule.next_within(now, look_ahead) if self.schedule else None
        if s is not None:
            return Arrival(s.key, s.start, s.end, s.name, "schedule", self._occ_probability(s.start))
        # Không có lịch: tìm điểm "bắt đầu có người" từ thói quen (xác suất vượt ngưỡng sau một slot thấp).
        if self.habit is None:
            return None
        step = timedelta(minutes=cfg.habit_slot_min)
        base = now.replace(minute=(now.minute // cfg.habit_slot_min) * cfg.habit_slot_min, second=0, microsecond=0)
        slot_start = base + step
        horizon = now + timedelta(minutes=look_ahead)
        while slot_start <= horizon:
            cur = self.habit.get(slot_start.isoweekday() + 1, self._slot(slot_start))
            prev_dt = slot_start - step
            prev = self.habit.get(prev_dt.isoweekday() + 1, self._slot(prev_dt))
            if (cur and cur.n_obs >= cfg.habit_min_obs and cur.occ_probability > cfg.precool_prob
                    and (prev is None or prev.occ_probability <= cfg.precool_prob)):
                return Arrival(f"habit_{slot_start:%Y-%m-%d_%H%M}", slot_start, None, "Thói quen", "habit",
                               cur.occ_probability)
            slot_start += step
        return None

    # ---------------------------------------------------------------- bước chính
    def step(self, inp: HvacInputs) -> HvacResult:
        cfg, now = self.cfg, inp.now
        t = now.timestamp()
        ev: list[Event] = []
        cmd: AcCommand | None = None
        rate_upd: tuple[int, time, float] | None = None
        notes: list[str] = []

        if hasattr(self.schedule, "maybe_reload"):
            self.schedule.maybe_reload()

        data_ok = (inp.n_total is not None and inp.coverage >= cfg.min_coverage and inp.vision_fresh)
        empty = data_ok and inp.n_total < cfg.empty_mean_max
        occupied = data_ok and inp.n_total >= cfg.empty_mean_max
        if not data_ok:
            notes.append("Thiếu dữ liệu đếm người (camera/bộ đệm 10 phút): không ra quyết định theo hiện diện.")

        sess = self.schedule.current(now) if self.schedule else None
        in_class = sess is not None and sess.key not in self.cancelled

        # Buổi mới bắt đầu: nếu đã pre-cool thì chờ phòng đầy dần rồi mới điều chỉnh setpoint.
        if sess is not None and sess.key != self._session_key:
            self._session_key = sess.key
            pre = self.plan is not None and self.plan.arrival.key == sess.key and self.plan.started
            self.next_eval_at = t + (cfg.hvac_interval_min * 60 if pre else 0.0)
        elif sess is None:
            self._session_key = None

        # ---- 1) học R_cool sau 15 phút pre-cooling ---------------------------------
        p = self.plan
        if p and p.started and not p.evaluated and p.eval_at and now >= p.eval_at:
            p.evaluated = True
            if p.valid_for_rate and p.t_in_start is not None and inp.t_in is not None and self.ac.power == 1:
                sample = cooling_rate_sample(p.t_in_start, inp.t_in, cfg.r_cool_eval_min)
                if sample >= 0.01:
                    sample = min(sample, 1.0)
                    new = ema(p.r_cool, sample, cfg.r_cool_alpha)
                    rate_upd = (p.arrival.start.isoweekday() + 1, self._slot(p.arrival.start), new)
                    ev.append(self._ev(t, "R_COOL_UPDATED", "info",
                                       f"R_cool: {p.r_cool:.3f} -> {new:.3f} °C/phút (mẫu hôm nay {sample:.3f}).",
                                       old=p.r_cool, today=sample, new=new))
                else:
                    ev.append(self._ev(t, "R_COOL_SKIPPED", "warn",
                                       f"Mẫu R_cool {sample:.3f} °C/phút quá thấp (máy chưa làm mát được?) — không cập nhật.",
                                       sample=sample))
            else:
                ev.append(self._ev(t, "R_COOL_SKIPPED", "info", "Bỏ qua cập nhật R_cool (thiếu số đo hoặc AC không do kế hoạch bật)."))

        # ---- 2) lập / cập nhật kế hoạch pre-cooling ---------------------------------
        arr = self._next_arrival(now)
        if arr is not None and arr.key not in self.cancelled:
            if self.plan is None or self.plan.arrival.key != arr.key:
                if arr.key not in self.decided:
                    if arr.p <= cfg.precool_prob:
                        self.decided.add(arr.key)
                        ev.append(self._ev(t, "PRECOOL_SKIPPED", "info",
                                           f"Không pre-cool '{arr.name}' {arr.start:%H:%M}: xác suất có người {arr.p:.2f} <= {cfg.precool_prob}.",
                                           p=arr.p))
                    elif self.ac.power == 1:
                        # KHÔNG đưa vào `decided`: nếu máy bị tắt (phòng trống giữa giờ) trước giờ vào lớp thì còn phải lập kế hoạch.
                        if arr.key not in self._noted:
                            self._noted.add(arr.key)
                            ev.append(self._ev(t, "PRECOOL_NOT_NEEDED", "info",
                                               f"Điều hòa đang bật trước '{arr.name}' {arr.start:%H:%M} — chưa cần pre-cool."))
                    else:
                        self.decided.add(arr.key)
                        self.plan = self._make_plan(arr, inp, now)
                        pl = self.plan
                        ev.append(self._ev(t, "PRECOOL_PLANNED", "info",
                                           f"Pre-cool '{arr.name}' {arr.start:%H:%M}: t_lead = {pl.t_lead_min:.0f} phút "
                                           f"-> bật lúc {pl.start_at:%H:%M} (P có người {arr.p:.2f}).",
                                           **pl.to_dict()))
            elif self.plan is not None and not self.plan.started:
                # Chưa tới giờ bật: tính lại với số đo mới nhất (T_in đổi theo nắng).
                self._refresh_plan(self.plan, inp, now)

        # ---- 3) tới giờ bật pre-cooling ---------------------------------------------
        p = self.plan
        if cmd is None and p and not p.started and now >= p.start_at and self.ac.power == 0 and self._off_wait_s(now) > 0:
            notes.append(f"Tới giờ pre-cool nhưng máy vừa tắt — chờ thêm {self._off_wait_s(now):.0f} s (bảo vệ máy nén).")
        elif cmd is None and p and not p.started and now >= p.start_at:
            p.started = True
            p.started_at = now
            p.t_in_start = inp.t_in
            p.eval_at = now + timedelta(minutes=cfg.r_cool_eval_min)
            if self.ac.power == 1:
                p.valid_for_rate = False
                ev.append(self._ev(t, "PRECOOL_NOT_NEEDED", "info", "Điều hòa đã bật sẵn khi tới giờ pre-cool."))
            else:
                cmd = self._send(now, 1, cfg.precool_ac_temp, 3, "PRECOOL", urgent=True)
                ev.append(self._ev(t, "PRECOOL_START", "info",
                                   f"Bật điều hòa pre-cooling ({cfg.precool_ac_temp}°C, quạt mạnh); T_in = "
                                   f"{'?' if inp.t_in is None else f'{inp.t_in:.1f}'}°C.", t_in=inp.t_in))

        # ---- 4) fail-safe: 15 phút sau giờ vào lớp mà phòng trống -----------------
        cands = []
        if sess is not None:
            cands.append((sess.key, sess.start, sess.end, sess.name))
        if self.plan is not None and self.plan.started and (sess is None or sess.key != self.plan.arrival.key):
            a = self.plan.arrival
            cands.append((a.key, a.start, a.end, a.name))
        for key, start, end, name in cands:
            if key in self.failsafe_done:
                continue
            if now >= start + timedelta(minutes=cfg.failsafe_min) and (end is None or now < end):
                if not data_ok:
                    if "failsafe_nodata" not in self._warned:
                        self._warned.add("failsafe_nodata")
                        ev.append(self._ev(t, "FAILSAFE_DEFERRED", "warn",
                                           "Tới mốc kiểm tra fail-safe nhưng thiếu dữ liệu camera — hoãn, không tắt máy vì mất dữ liệu."))
                    continue
                self._warned.discard("failsafe_nodata")
                self.failsafe_done.add(key)
                if empty:
                    self.cancelled.add(key)
                    in_class = False
                    if self.ac.power == 1 and cmd is None:
                        cmd = self._send(now, 0, self.ac.temp or cfg.precool_ac_temp, 0, "FAILSAFE_EMPTY", urgent=True)
                    ev.append(self._ev(t, "FAILSAFE_EMPTY_CLASS", "warn",
                                       f"EXCEPTION: '{name}' bắt đầu {start:%H:%M} nhưng sau {cfg.failsafe_min:.0f} phút "
                                       f"phòng vẫn trống -> tắt điều hòa.", key=key))

        # ---- 5) hết giờ + phòng trống => tắt ---------------------------------------
        protected = (self.plan is not None and self.plan.started
                     and now < self.plan.arrival.start + timedelta(minutes=cfg.failsafe_min))
        off_cond = self.ac.power == 1 and not in_class and empty and not protected
        class_soon = (arr is not None and arr.key not in self.cancelled and arr.p > cfg.precool_prob
                      and 0 <= (arr.start - now).total_seconds() <= cfg.ac_keep_on_before_class_min * 60)
        if off_cond and class_soon:
            notes.append(f"Phòng trống nhưng '{arr.name}' bắt đầu lúc {arr.start:%H:%M} — giữ điều hòa chạy, "
                         f"không tắt rồi pre-cool lại ngay.")
            off_cond = False
            # Giữ máy chạy thay cho pre-cool => phải làm đúng việc của pre-cool: đưa về nhiệt độ/quạt pre-cool (một lần)
            if (cmd is None and arr.key not in self._keepon_done and self.ac.temp > cfg.precool_ac_temp
                    and not self._throttled(now)):
                self._keepon_done.add(arr.key)
                cmd = self._send(now, 1, cfg.precool_ac_temp, 3, "PRECOOL_KEEP_ON", urgent=False)
                ev.append(self._ev(t, "PRECOOL_KEEP_ON", "info",
                                   f"Giữ điều hòa chạy chờ '{arr.name}' {arr.start:%H:%M}: hạ về {cfg.precool_ac_temp}°C, quạt mạnh "
                                   f"(thay cho tắt rồi pre-cool lại)."))
        if off_cond:
            if self._empty_since is None:
                self._empty_since = t
        else:
            self._empty_since = None
        if cmd is None and off_cond:
            held = t - self._empty_since
            if held < cfg.ac_off_delay_s:
                notes.append(f"Phòng trống: chờ trễ tắt {cfg.ac_off_delay_s - held:.0f} s nữa.")
            elif self._on_wait_s(now) > 0:
                notes.append(f"Phòng trống nhưng máy mới bật — chạy thêm {self._on_wait_s(now):.0f} s mới tắt (bảo vệ máy nén).")
            elif not self._throttled(now):
                cmd = self._send(now, 0, self.ac.temp, 0, "EMPTY_ROOM", urgent=False)
                self._empty_since = None
                ev.append(self._ev(t, "AC_OFF_EMPTY", "info",
                                   f"Phòng trống suốt cửa sổ 10 phút (+{cfg.ac_off_delay_s:g} s trễ tắt) và không trong giờ học -> tắt điều hòa."))

        # ---- 6) bật theo hiện diện (không có lịch / lớp không được pre-cool) ---------
        calc = None
        if data_ok and inp.zone_avgs is not None:
            calc = t_setpoint(inp.n_total, self.n_max, inp.rh, inp.zone_avgs, self.zone_caps,
                              t_base=cfg.t_base, people_gain=cfg.people_gain, rh_high=cfg.rh_high,
                              rh_delta=cfg.rh_delta, zone_frac=cfg.zone_crowd_frac, zone_delta=cfg.zone_crowd_delta,
                              t_min=cfg.t_min, t_max=cfg.t_max)
            self._last_raw = calc.raw
        on_cond = (self.ac.power == 0 and calc is not None and inp.t_in is not None
                   and inp.n_total >= cfg.occ_on_people and inp.t_in > calc.integer + cfg.turn_on_margin)
        if on_cond:
            if self._on_cond_since is None:
                self._on_cond_since = t
        else:
            self._on_cond_since = None
        on_ready = on_cond and (t - self._on_cond_since) >= cfg.ac_on_delay_s
        if cmd is None and on_cond and not on_ready:
            notes.append(f"Có người và nóng: chờ trễ bật {cfg.ac_on_delay_s - (t - self._on_cond_since):.0f} s nữa.")
        elif cmd is None and on_ready and self._off_wait_s(now) > 0:
            notes.append(f"Cần bật điều hòa nhưng máy vừa tắt — chờ thêm {self._off_wait_s(now):.0f} s (bảo vệ máy nén).")
        if cmd is None and on_ready and self._off_wait_s(now) <= 0 and not self._throttled(now):
            fan = self._fan(inp.n_total, bool(calc.crowded_zones), inp.t_in, calc.integer, inp.rh)
            self._on_cond_since = None
            cmd = self._send(now, 1, calc.integer, fan, "OCCUPANCY_ON", urgent=False)
            self.next_eval_at = t + cfg.hvac_interval_min * 60
            ev.append(self._ev(t, "AC_ON_OCCUPANCY", "info",
                               f"Có {inp.n_total:.1f} người, T_in {inp.t_in:.1f}°C > {calc.integer}+{cfg.turn_on_margin:g} -> bật điều hòa {calc.integer}°C.",
                               setpoint=calc.integer, fan=fan))

        # ---- 7) chế độ trong lớp: cập nhật setpoint mỗi 10-15 phút ------------------
        precooling = bool(self.plan and self.plan.started and now < self.plan.arrival.start)
        if (cmd is None and self.ac.power == 1 and calc is not None and (in_class or occupied)
                and not precooling and t >= self.next_eval_at):
            fan = self._fan(inp.n_total, bool(calc.crowded_zones), inp.t_in, calc.integer, inp.rh)
            temp_changed = deadband_pass(calc.raw, self.ac.temp, cfg.deadband)
            fan_changed = fan != self.ac.fan
            if temp_changed and calc.integer > self.ac.temp and self._around_class_start(now, sess):
                # Giờ giải lao giữa 2 buổi / vài phút đầu buổi: bộ đệm 10 phút còn nhớ phòng vãn người nên muốn NỚI setpoint,
                # nhưng lớp đang đông lại => giữ nguyên, nếu nới thì phòng ấm lên đúng lúc sinh viên vào.
                temp_changed = False
                notes.append(f"Giữ setpoint {self.ac.temp}°C quanh giờ vào lớp (bộ đệm 10 phút chưa phản ánh lớp đông).")
            if temp_changed or fan_changed:
                if self._throttled(now):
                    notes.append("Có thay đổi cần gửi nhưng đang trong khoảng giãn cách tối thiểu giữa 2 lệnh IR.")
                else:
                    new_temp = calc.integer if temp_changed else self.ac.temp
                    step = cfg.ac_setpoint_max_step
                    if abs(new_temp - self.ac.temp) > step:  # đổi từ từ: lần đánh giá sau (10-15 phút) đi tiếp
                        new_temp = self.ac.temp + (step if new_temp > self.ac.temp else -step)
                    cmd = self._send(now, 1, new_temp, fan, "ACTIVE", urgent=False)
                    self.next_eval_at = t + cfg.hvac_interval_min * 60
                    ev.append(self._ev(t, "AC_SETPOINT", "info",
                                       f"Setpoint -> {new_temp}°C, quạt {fan} (N={inp.n_total:.1f}/{self.n_max}, "
                                       f"raw {calc.raw:.2f}; -người {calc.d_people:.2f} -ẩm {calc.d_humidity:.1f} -vùng {calc.d_zone:.1f}).",
                                       raw=calc.raw, n_total=inp.n_total))
            else:
                self.next_eval_at = t + cfg.hvac_interval_min * 60  # trong deadband: không bắn IR

        # ---- trạng thái cho dashboard ----------------------------------------------
        if cmd is not None and cmd.power == 0:
            mode = "FAILSAFE_OFF" if cmd.reason == "FAILSAFE_EMPTY" else "OFF"
        elif self.ac.power == 0:
            mode = "PRECOOL_WAIT" if (self.plan and not self.plan.started) else "OFF"
        elif self.plan and self.plan.started and now < self.plan.arrival.start:
            mode = "PRECOOL"
        elif in_class or occupied:
            mode = "ACTIVE"
        else:
            mode = "ON_IDLE"
        status = HvacStatus(
            t=t, mode=mode, setpoint_raw=self._last_raw,
            plan=self.plan.to_dict() if self.plan else None,
            next_eval_t=self.next_eval_at if self.next_eval_at > 0 else None,
            r_cool=self.r_cool_used, notes=notes,
        )
        return HvacResult(command=cmd, events=ev, status=status, cooling_rate_update=rate_upd)

    def _around_class_start(self, now: datetime, sess) -> bool:
        """Đang trong khoảng [-hold_before_class_min, +hold_after_class_start_min] quanh giờ bắt đầu một buổi."""
        cfg = self.cfg
        s0 = sess if sess is not None else (self.schedule.next_within(now, cfg.hold_before_class_min) if self.schedule else None)
        if s0 is None:
            return False
        delta_min = (now - s0.start).total_seconds() / 60.0
        return -cfg.hold_before_class_min <= delta_min <= cfg.hold_after_class_start_min

    # ---------------------------------------------------------------- kế hoạch
    def _lead(self, arr: Arrival, inp: HvacInputs) -> tuple[float, float]:
        cfg = self.cfg
        r = self._r_cool(arr.start)
        self.r_cool_used = r
        if inp.t_in is None:
            return min(cfg.lead_max_min, 20.0), r  # không có số đo: lùi về mặc định 20 phút
        t_out = inp.t_out if inp.t_out is not None else cfg.t_out_ref
        return t_lead_minutes(inp.t_in, t_out, r, cfg.t_target, cfg.gamma, cfg.t_out_ref, cfg.r_cool_min,
                              cfg.lead_max_min), r

    def _make_plan(self, arr: Arrival, inp: HvacInputs, now: datetime) -> PreCoolPlan:
        lead, r = self._lead(arr, inp)
        start_at = max(now, arr.start - timedelta(minutes=lead))
        return PreCoolPlan(arrival=arr, t_lead_min=lead, start_at=start_at, r_cool=r, t_in_plan=inp.t_in)

    def _refresh_plan(self, plan: PreCoolPlan, inp: HvacInputs, now: datetime) -> None:
        lead, r = self._lead(plan.arrival, inp)
        plan.t_lead_min, plan.r_cool = lead, r
        plan.start_at = max(now, plan.arrival.start - timedelta(minutes=lead))
        plan.t_in_plan = inp.t_in


# ===========================================================================
# LỚP KEO: cảm biến + bộ đệm -> bộ điều khiển -> MQTT/DB
# ===========================================================================
class ControlEngine:
    def __init__(self, cfg: AresSettings, clock, state, zonemap, light_buf, hvac_buf, schedule, habit, bus, db=None):
        self.cfg, self.clock, self.state = cfg, clock, state
        self.zonemap, self.light_buf, self.hvac_buf = zonemap, light_buf, hvac_buf
        self.schedule, self.habit, self.bus, self.db = schedule, habit, bus, db
        self.lighting = LightingController(cfg, zonemap.k_factors())
        self.hvac = HvacController(cfg, zonemap.room.n_max, zonemap.capacities(), schedule, habit)
        self._last_pub_pwm: list[float] | None = None
        self._last_pub_t: float = 0.0

    # ---- Thread 2 ----------------------------------------------------------------
    def lighting_tick(self) -> LightingState:
        t = self.clock.time()
        now = self.clock.now()
        cfg = self.cfg
        sensors = self.state.get_sensors()
        fresh = sensors is not None and (t - sensors.t) <= cfg.sensor_stale_s
        n = self.zonemap.n
        lux_c = list(sensors.lux_zones) if fresh else [None] * n
        lux_c += [None] * (n - len(lux_c))
        stats = self.light_buf.stats(t)
        vision_ok = stats.ok and stats.last_t is not None and (t - stats.last_t) <= cfg.frame_stale_s
        in_session = bool(self.schedule and self.schedule.current(now) is not None)
        out = self.lighting.step(LightInputs(
            t=t, n_avg=stats.mean_zones, vision_ok=vision_ok, in_session=in_session,
            lux_ceiling=lux_c, lux_window=sensors.lux_window if fresh else None,
        ))
        if not fresh:
            out.notes.append("Mất gói cảm biến ESP32 #1: giữ PWM, sau thời gian chờ chuyển PWM dự phòng.")
        self.state.set_lighting(out)
        self._publish_led(out, t)
        return out

    def _publish_led(self, out: LightingState, t: float) -> None:
        cfg = self.cfg
        prev = self._last_pub_pwm
        changed = prev is None or any(abs(a - b) >= cfg.led_publish_min_delta for a, b in zip(out.pwm, prev))
        heartbeat = (t - self._last_pub_t) >= cfg.led_heartbeat_s
        if changed or heartbeat:
            self.bus.publish_led(list(out.pwm), cfg.fade_ms)
            self._last_pub_pwm = list(out.pwm)
            self._last_pub_t = t

    # ---- Thread 3 ----------------------------------------------------------------
    def hvac_tick(self) -> HvacResult:
        t = self.clock.time()
        now = self.clock.now()
        cfg = self.cfg
        sensors = self.state.get_sensors()
        fresh = sensors is not None and (t - sensors.t) <= cfg.sensor_stale_s
        stats = self.hvac_buf.stats(t)
        vision_fresh = stats.last_t is not None and (t - stats.last_t) <= cfg.frame_stale_s
        res = self.hvac.step(HvacInputs(
            now=now,
            t_in=sensors.temp_in if fresh else None,
            t_out=sensors.temp_out if fresh else None,
            rh=sensors.humidity if fresh else None,
            n_total=stats.mean_total, coverage=stats.coverage, zone_avgs=stats.mean_zones,
            vision_fresh=vision_fresh,
        ))
        if not fresh:
            res.status.notes.append("Mất gói cảm biến ESP32 #1: không có T_in/T_out/RH.")
        for e in res.events:
            self.emit(e)
        if res.command is not None:
            self.bus.publish_ac(res.command.payload())
            self.state.set_ac(self.hvac.ac)
        if res.cooling_rate_update is not None and self.habit is not None:
            dow, slot, rate = res.cooling_rate_update
            try:
                self.habit.set_cooling_rate(dow, slot, rate)
            except Exception:
                log.exception("Không lưu được R_cool vào habit_profiles")
        self.state.set_hvac(res.status)
        return res

    def emit(self, e: Event) -> None:
        self.state.add_event(e)
        (log.warning if e.severity != "info" else log.info)("[%s] %s", e.kind, e.message)
        if self.db is not None:
            self.db.log_event(e)
