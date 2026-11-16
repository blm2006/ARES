"""Nhóm 1 — CCTV-led: RB-01 (mũ bảo hiểm), RB-06 (chở quá người), RB-09 (hàng cồng kềnh), RB-10 (tốc độ)."""

from __future__ import annotations

import statistics
from collections import deque
from typing import Any

import numpy as np

from traffic_safety_ai.core.association import maybe_child
from traffic_safety_ai.core.types import FrameContext, Overlay
from traffic_safety_ai.rules.base_rule import (RED, BaseRule, Requirement, RuleResult, is_moving, motion_arrow,
                                               track_box_overlay, union_of)
from traffic_safety_ai.utils.geometry import point_in_polygon


# ===================================================================== RB-01
class NoHelmetRule(BaseRule):
    """Người trên xe máy đang chạy không đội mũ bảo hiểm.

    Dữ liệu vào: track người có `attrs["helmet_obs"]` = deque[(t, nhãn, conf)] do HelmetEstimator ghi
    (nhãn: helmet | no_helmet | unknown). Cần model mũ bảo hiểm (models/helmet.pt).
    Khác spec: chỉ xét xe ĐANG CHẠY (người ngồi trên xe đỗ trước cổng trường không vi phạm);
    người ngồi sau thấp hơn hẳn người lái → có thể là trẻ dưới 6 tuổi (được miễn) → hạ xuống "nghi vấn".
    """

    code = "RB-01"
    name = "Không đội mũ bảo hiểm"
    channels = ("cctv",)
    requirements = (Requirement(("helmet",), "model mũ bảo hiểm (models/helmet.pt)"),)
    defaults = {"trigger_s": 0.4, "window_s": 1.2, "min_obs": 3, "min_no_frac": 0.65, "min_conf": 0.4,
                "child_height_ratio": 0.72}

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        for moto_id, rider_ids in ctx.riders.items():
            moto = ctx.by_id(moto_id)
            if moto is None or moto.cls != "motorcycle" or not is_moving(moto):
                continue
            driver = ctx.by_id(rider_ids[0]) if rider_ids else None
            for idx, pid in enumerate(rider_ids):
                p = ctx.by_id(pid)
                if p is None:
                    continue
                obs = [o for o in p.attrs.get("helmet_obs", ()) if o[0] >= ctx.t - self.p["window_s"]]
                known = [o for o in obs if o[1] in ("helmet", "no_helmet")]
                if len(known) < self.p["min_obs"]:
                    continue
                no = [o for o in known if o[1] == "no_helmet"]
                frac = len(no) / len(known)
                conf = float(np.mean([o[2] for o in no])) if no else 0.0
                hit = frac >= self.p["min_no_frac"] and conf >= self.p["min_conf"]
                variant = "driver" if idx == 0 else "passenger"
                notes = []
                severity = None
                if variant == "passenger" and driver is not None and maybe_child(p, driver, self.p["child_height_ratio"]):
                    severity = "nghi_van"
                    notes.append("Người ngồi sau thấp hơn hẳn người lái — có thể là trẻ em dưới 06 tuổi (được miễn), cần xác minh.")
                if variant == "passenger":
                    notes.append("Người điều khiển bị xử phạt về hành vi chở người không đội mũ (điểm i khoản 2 Điều 7).")
                out.append(RuleResult(
                    key=str(pid), hit=hit, primary_track_id=moto_id, track_ids=[moto_id, pid], vehicle_cls="motorcycle",
                    bbox=union_of([moto, p]), variant=variant, severity=severity, confidence=min(0.99, frac * conf + 0.1),
                    measurements={"ti_le_khong_mu": round(frac, 2), "so_lan_quan_sat": len(known),
                                  "vi_tri": "người lái" if idx == 0 else "người ngồi sau"},
                    notes=notes,
                    overlays=[track_box_overlay(p, RED, "KHÔNG MŨ"), track_box_overlay(moto, RED)],
                ))
        return out


# ===================================================================== RB-06
class ExcessPassengersRule(BaseRule):
    """Xe máy chở quá số người.

    Theo Luật TTATGTĐB 2024 (khoản 1 Điều 33): được chở tối đa 2 người khi chở trẻ em dưới 12 tuổi, người bệnh
    đi cấp cứu, người già yếu/khuyết tật, áp giải. Ở cổng trường tiểu học, phụ huynh chở 2 con là HỢP LỆ.
    → 3 người trên xe = "nghi vấn" (cần người xác minh); từ 4 người = "vi phạm" (điểm b khoản 3 Điều 7).
    """

    code = "RB-06"
    name = "Chở quá số người quy định"
    channels = ("cctv",)
    requirements = (Requirement(("vehicles",), "model nền"),)
    defaults = {"trigger_s": 0.8, "window_s": 1.0, "child_height_ratio": 0.72}

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self._counts: dict[int, deque] = {}

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        for moto_id, rider_ids in ctx.riders.items():
            moto = ctx.by_id(moto_id)
            if moto is None or moto.cls != "motorcycle":
                continue
            dq = self._counts.setdefault(moto_id, deque(maxlen=90))
            dq.append((ctx.t, len(rider_ids)))
            if not is_moving(moto):
                continue
            recent = [n for t, n in dq if t >= ctx.t - self.p["window_s"]]
            n = int(round(statistics.median(recent))) if recent else len(rider_ids)
            hit = n >= 3
            people = [ctx.by_id(i) for i in rider_ids]
            people = [p for p in people if p is not None]
            notes: list[str] = []
            if n >= 4:
                variant, severity = "4_nguoi_tro_len", "vi_pham"
            else:
                variant, severity = "3_nguoi", "nghi_van"
                notes.append("Chở 2 người là hợp lệ nếu chở trẻ em dưới 12 tuổi, người bệnh đi cấp cứu, người già yếu"
                             "/khuyết tật hoặc áp giải (khoản 1 Điều 33 Luật TTATGTĐB 2024) — cần người xác minh.")
            if people and any(maybe_child(p, people[0], self.p["child_height_ratio"]) for p in people[1:]):
                notes.append("Có người ngồi sau dáng nhỏ — có thể là trẻ em.")
            ov = [track_box_overlay(moto, RED, f"{n} NGƯỜI")] + [track_box_overlay(p, RED) for p in people]
            out.append(RuleResult(
                key=str(moto_id), hit=hit, primary_track_id=moto_id, track_ids=[moto_id] + list(rider_ids),
                vehicle_cls="motorcycle", bbox=union_of([moto] + people), variant=variant, severity=severity,
                confidence=0.9 if n >= 4 else 0.7, measurements={"so_nguoi_tren_xe": n}, notes=notes, overlays=ov,
            ))
        for k in [k for k in self._counts if ctx.by_id(k) is None]:
            self._counts.pop(k, None)
        return out


# ===================================================================== RB-09
class OverloadRule(BaseRule):
    """Xe máy chở hàng cồng kềnh.

    Dữ liệu vào: track xe có `attrs["cargo_obs"]` = deque[(t, tỉ lệ bề rộng, bề rộng mét|None, phương pháp)]
    do CargoEstimator ghi — bằng model hàng hoá (models/cargo.pt) hoặc tách nền (CCTV tĩnh) khi chưa có model.
    Ngưỡng tham khảo quy định xếp hàng: mỗi bên không quá 0,3 m so với giá đèo hàng → bề rộng tổng ~1,2 m.
    """

    code = "RB-09"
    name = "Chở hàng cồng kềnh"
    channels = ("cctv",)
    default_severity = "nghi_van"
    requirements = (Requirement(("cargo", "bgsub"), "model hàng hoá hoặc camera tĩnh để tách nền"),)
    defaults = {"trigger_s": 0.8, "window_s": 1.0, "max_ratio_w": 1.7, "max_width_m": 1.35, "min_obs": 3}

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        for moto in ctx.of_class("motorcycle"):
            obs = [o for o in moto.attrs.get("cargo_obs", ()) if o[0] >= ctx.t - self.p["window_s"]]
            if len(obs) < self.p["min_obs"] or not is_moving(moto):
                continue
            ratio = float(np.median([o[1] for o in obs]))
            widths = [o[2] for o in obs if o[2] is not None]
            width_m = float(np.median(widths)) if widths else None
            method = obs[-1][3]
            hit = (width_m is not None and width_m > self.p["max_width_m"]) or ratio > self.p["max_ratio_w"]
            env = moto.attrs.get("cargo_box")
            ov = [track_box_overlay(moto, RED)]
            if env is not None:
                ov.append(Overlay("box", [(env[0], env[1]), (env[2], env[3])], RED, thickness=2, text="HÀNG"))
            meas: dict[str, Any] = {"ti_le_be_rong": round(ratio, 2), "phuong_phap": method}
            if width_m is not None:
                meas["be_rong_uoc_luong_m"] = round(width_m, 2)
            out.append(RuleResult(
                key=str(moto.id), hit=hit, primary_track_id=moto.id, track_ids=[moto.id], vehicle_cls="motorcycle",
                bbox=moto.xyxy if env is None else np.array([min(moto.xyxy[0], env[0]), min(moto.xyxy[1], env[1]),
                                                             max(moto.xyxy[2], env[2]), max(moto.xyxy[3], env[3])]),
                variant=method, severity="vi_pham" if method == "model" and ratio > self.p["max_ratio_w"] * 1.2 else None,
                confidence=0.75 if method == "model" else 0.55, measurements=meas,
                notes=["Bề rộng ước lượng từ ảnh — cần người xác minh kích thước thực."], overlays=ov,
            ))
        return out


# ===================================================================== RB-10
def speed_variant(cls: str, excess: float) -> str | None:
    if excess < 5:
        return None
    if cls == "motorcycle":
        return "moto_5_10" if excess < 10 else ("moto_10_20" if excess <= 20 else "moto_20_plus")
    if excess < 10:
        return "car_5_10"
    if excess <= 20:
        return "car_10_20"
    return "car_20_35" if excess <= 35 else "car_35_plus"


class SpeedingRule(BaseRule):
    """Chạy quá tốc độ.

    Hai cách đo: (1) cặp vạch ảo cách nhau L mét ("speed trap" — tin cậy nhất); (2) homography + hồi quy.
    Giới hạn lấy theo biển báo tại chỗ (speed.limit_kmh / limits_by_class), KHÔNG cố định 30 km/h.
    Chỉ báo khi vượt ≥ 5 km/h (mức thấp nhất bị xử phạt) + biên sai số `tolerance_kmh`.
    Lưu ý: tốc độ đo bằng camera thường chỉ là ƯỚC LƯỢNG; xử phạt chính thức cần thiết bị đo đã kiểm định.
    """

    code = "RB-10"
    motion_sensitive = True
    name = "Chạy quá tốc độ quy định"
    channels = ("cctv",)
    requirements = (Requirement(("calibration", "speed_trap"), "hiệu chuẩn mặt đường hoặc cặp vạch đo tốc độ"),)
    defaults = {"trigger_s": 0.4, "max_sigma_kmh": 6.0, "classes": ["motorcycle", "car", "bus", "truck"]}

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self._hist: dict[int, list[float]] = {}

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        tol = float(self.scene.speed.tolerance_kmh)
        zones = self.scene.zones.speed_zone
        for tr in ctx.vehicles(tuple(self.p["classes"])):
            v, method, sigma = None, None, None
            trap = tr.attrs.get("trap_speed")
            if trap is not None and ctx.t - trap[1] < 0.5:
                v, method = float(trap[0]), "vach_do"
            elif ctx.calib is not None and tr.kin.speed_kmh is not None:
                if tr.kin.window_s < float(self.scene.speed.min_track_s):
                    continue
                if tr.kin.speed_sigma_kmh is not None and tr.kin.speed_sigma_kmh > self.p["max_sigma_kmh"]:
                    continue
                if not ctx.calib.in_calibrated_area(tr.bottom):
                    continue
                if zones and not any(point_in_polygon(tr.bottom, z.polygon) for z in zones):
                    continue
                v, method, sigma = tr.kin.speed_kmh, "homography", tr.kin.speed_sigma_kmh
            if v is None:
                continue
            limit = self.scene.speed_limit(tr.cls)
            excess = v - limit
            hit = excess >= 5.0 + tol
            hist = self._hist.setdefault(tr.id, [])
            if hit:
                hist.append(v)
            v_rep = float(np.median(hist[-15:])) if hist else v
            variant = speed_variant(tr.cls, v_rep - limit) or speed_variant(tr.cls, excess)
            meas: dict[str, Any] = {"toc_do_kmh": round(v_rep, 1), "gioi_han_kmh": round(limit, 1),
                                    "vuot_kmh": round(v_rep - limit, 1), "phuong_phap_do": method}
            if sigma is not None:
                meas["sai_so_kmh"] = round(sigma, 1)
            ov = [track_box_overlay(tr, RED, f"{v_rep:.0f} km/h")]
            arr = motion_arrow(tr)
            if arr:
                ov.append(arr)
            out.append(RuleResult(
                key=str(tr.id), hit=hit, primary_track_id=tr.id, track_ids=[tr.id], vehicle_cls=tr.cls,
                bbox=tr.xyxy, variant=variant, confidence=0.85 if method == "vach_do" else 0.7,
                measurements=meas, instant=(method == "vach_do"),
                notes=["Tốc độ ước lượng bằng camera — không thay thế thiết bị đo tốc độ đã kiểm định."],
                overlays=ov,
            ))
        for k in [k for k in self._hist if ctx.by_id(k) is None]:
            self._hist.pop(k, None)
        return out
