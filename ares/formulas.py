"""Các công thức của Control Engine (spec mục 6) dưới dạng hàm thuần.

Tách riêng để test từng công thức bằng số cụ thể, không cần luồng/MQTT/DB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def round_half_up(x: float) -> int:
    """Làm tròn 24.5 -> 25 (Python round() là banker's: 24.5 -> 24, 25.5 -> 26 — khó đoán khi đặt máy lạnh)."""
    return int(math.floor(x + 0.5))


# ---------------------------------------------------------------------------
# A. Đèn: vòng phản hồi tỉ lệ
# ---------------------------------------------------------------------------
def desk_lux(lux_ceiling: float, k: float) -> float:
    """L_actual,i = Lux_Ceiling_i × k_i."""
    return lux_ceiling * k


def lux_setpoint(n_zone: float, lux_occupied: float, lux_dense: float, lux_idle: float,
                 dense_people: float, occupied: bool) -> float:
    """400 lux nếu có người; 450 lux nếu > 8 người (bù bóng đổ); 100 lux nếu trống."""
    if not occupied:
        return lux_idle
    return lux_dense if n_zone > dense_people else lux_occupied


def pwm_update(prev_pwm: float, setpoint: float, actual: float, kp: float, dl_max: float,
               pwm_min: float, pwm_max: float = 100.0) -> float:
    """PWM(t) = clamp(PWM(t-1) + Kp · e/ΔL_max, PWM_min, 100 %).

    Đơn vị: spec viết PWM và tỉ số e/ΔL_max lẫn nhau. Ở đây PWM tính theo % nên số hạng
    hiệu chỉnh là 100 · Kp · e/ΔL_max (e = ΔL_max tương ứng đúng 100 % PWM). Nếu cộng
    thẳng Kp·e/ΔL_max (≤ 1) vào PWM tính theo % thì mỗi nhịp chỉ nhích < 1 % và đèn
    mất hàng chục phút mới sáng.
    """
    e = setpoint - actual
    return clamp(prev_pwm + 100.0 * kp * e / dl_max, pwm_min, pwm_max)


# ---------------------------------------------------------------------------
# B. Pre-cooling
# ---------------------------------------------------------------------------
def t_lead_minutes(t_in: float, t_out: float, r_cool: float, t_target: float, gamma: float,
                   t_out_ref: float = 30.0, r_min: float = 0.02, lead_max: float = 60.0) -> float:
    """t_lead = (T_in − T_target)/R_cool + γ · max(0, T_out − 30).

    Phần đầu không âm (phòng đã mát hơn đích thì không cần "bù" âm). Kết quả kẹp [0, lead_max].
    """
    r = max(r_cool, r_min)
    base = max(0.0, t_in - t_target) / r
    solar = gamma * max(0.0, t_out - t_out_ref)
    return clamp(base + solar, 0.0, lead_max)


def cooling_rate_sample(t_start: float, t_later: float, minutes: float) -> float:
    """R_cool_today = (T_in(t_start) − T_in(t_start + 15 phút)) / 15  [°C/phút]."""
    return (t_start - t_later) / minutes


def ema(old: float, new: float, alpha: float = 0.2) -> float:
    """R_cool_new = 0.8·old + 0.2·today."""
    return (1.0 - alpha) * old + alpha * new


# ---------------------------------------------------------------------------
# C. Setpoint điều hòa động
# ---------------------------------------------------------------------------
@dataclass
class SetpointCalc:
    raw: float  # sau kẹp [t_min, t_max], CHƯA làm tròn (dùng cho deadband)
    unclamped: float
    integer: int
    d_people: float
    d_humidity: float
    d_zone: float
    crowded_zones: list[int]  # chỉ số vùng quá đông


def t_setpoint(n_total: float, n_max: float, rh: float | None, zone_counts: Sequence[float],
               zone_caps: Sequence[float], *, t_base: float, people_gain: float, rh_high: float,
               rh_delta: float, zone_frac: float, zone_delta: float, t_min: float, t_max: float) -> SetpointCalc:
    """T_setpoint = clamp(round(T_base − ΔT_people − ΔT_humidity − ΔT_zone_crowd), 22, 27)."""
    ratio = clamp(n_total / max(1.0, n_max), 0.0, 1.0)
    d_people = people_gain * ratio
    d_hum = rh_delta if (rh is not None and rh > rh_high) else 0.0
    crowded = [i for i, (n, c) in enumerate(zip(zone_counts, zone_caps)) if n > zone_frac * c]
    d_zone = zone_delta if crowded else 0.0
    unclamped = t_base - d_people - d_hum - d_zone
    raw = clamp(unclamped, t_min, t_max)
    return SetpointCalc(raw=raw, unclamped=unclamped, integer=int(clamp(round_half_up(raw), t_min, t_max)),
                        d_people=d_people, d_humidity=d_hum, d_zone=d_zone, crowded_zones=crowded)


def fan_level(n_total: float, n_max: float, crowded: bool, t_in: float | None, setpoint: float,
              rh: float | None, rh_high: float) -> int:
    """Quạt 1..3. Theo tải người; vùng quá đông hoặc phòng còn nóng hơn setpoint >= 2 °C thì HIGH."""
    ratio = n_total / max(1.0, n_max)
    fan = 1 if ratio < 0.2 else 2 if ratio <= 0.7 else 3
    if crowded:
        fan = 3
    if t_in is not None and (t_in - setpoint) >= 2.0:
        fan = 3
    if rh is not None and rh > rh_high:
        fan = max(fan, 2)
    return fan


def deadband_pass(raw: float, last_sent: float | None, deadband: float) -> bool:
    """Chỉ bắn IR khi giá trị tính ra lệch setpoint đang đặt quá ±deadband.

    So `raw` (chưa làm tròn) với số nguyên đã gửi => có trễ thật quanh ranh giới làm tròn,
    tránh 24 -> 25 -> 24 chỉ vì raw dao động 24.49 / 24.51.
    """
    if last_sent is None:
        return True
    return abs(raw - last_sent) > deadband


def comfort_band(ratio: float) -> tuple[float, float]:
    """Dải nhiệt độ thoải mái theo mức lấp đầy (bảng ASHRAE 55 / TCVN 7114 trong spec). Dùng để đối chiếu/test."""
    if ratio < 0.20:
        return 26.0, 27.0
    if ratio <= 0.70:
        return 24.5, 25.5
    return 23.0, 24.0
