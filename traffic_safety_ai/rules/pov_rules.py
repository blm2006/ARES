"""Nhóm 3 — POV-led (camera hành trình): HZ-01, HZ-02, HZ-03.

Mọi phép đo chuyển động ở đây đã TRỪ chuyển động của chính camera (ego-motion) trước khi dùng.
"""

from __future__ import annotations

import math

import numpy as np

from traffic_safety_ai.core.ego_motion import focal_px
from traffic_safety_ai.core.kinematics import robust_fit
from traffic_safety_ai.core.types import LARGE_VEHICLES, MOTOR_VEHICLES, FrameContext, Overlay, Track
from traffic_safety_ai.rules.base_rule import (AMBER, RED, BaseRule, Requirement, RuleResult, StationaryMemory,
                                               find_zone, is_moving, track_box_overlay, walking_pedestrians)
from traffic_safety_ai.utils.geometry import angle_deg


def lateral_rel_speed(tr: Track, ctx: FrameContext) -> float:
    """Vận tốc ngang (độ rộng hộp/giây) sau khi trừ dịch chuyển do camera — dùng khi KHÔNG có hình học mặt đường."""
    vx = tr.kin.vx_px
    if ctx.ego is not None and ctx.ego.warp is not None:
        sx, _ = ctx.ego.shift_at(*tr.center)
        vx -= sx * ctx.fps
    return vx / max(tr.w, 1.0)


def pov_geom(ctx: FrameContext) -> tuple[int, float, float, float, float]:
    """(W, y_chân_trời, tiêu cự px, chiều cao camera m, yaw px/s) — lấy từ ego-motion, thiếu thì từ cấu hình POV."""
    W, H = ctx.frame_size
    e = ctx.ego
    pov = ctx.scene.pov
    hz = e.horizon_y if e is not None and e.horizon_y > 0 else pov.horizon_y_ratio * H
    f = e.f_px if e is not None and e.f_px > 0 else focal_px(W, pov.hfov_deg)
    yaw = e.yaw_px_s if e is not None else 0.0
    return W, hz, f, pov.camera_height_m, yaw


def ground_xz(ctx: FrameContext, u: float, v: float, min_dv: float = 3.0) -> tuple[float, float] | None:
    """Điểm tiếp đất (pixel) → (X ngang, Z phía trước) mét, giả thiết mặt đường phẳng."""
    W, hz, f, ch, _ = pov_geom(ctx)
    dv = v - hz
    if dv < min_dv:
        return None
    return (u - W / 2.0) * ch / dv, f * ch / dv


def ground_series(tr: Track, ctx: FrameContext, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts, xs, zs = [], [], []
    for p in tr.recent(seconds):
        g = ground_xz(ctx, *p.ground_px)
        if g is not None:
            ts.append(p.t)
            xs.append(g[0])
            zs.append(g[1])
    return np.array(ts), np.array(xs), np.array(zs)


def lateral_speed_ms(tr: Track, ctx: FrameContext, seconds: float = 1.0, min_pts: int = 4,
                     min_span: float = 0.3) -> tuple[float, float] | None:
    """Vận tốc NGANG thật (m/s, + là sang phải) của vật trên mặt đường và khoảng cách Z hiện tại.

    Dùng toạ độ mặt đường nên hiện tượng "nở" khi camera tiến tới không bị hiểu nhầm là rẽ;
    phần dịch ngang do camera quay (yaw) được trừ: ΔX_ảo = Z · yaw / f.
    """
    ts, xs, _ = ground_series(tr, ctx, seconds)
    if len(ts) < min_pts or ts[-1] - ts[0] < min_span:
        return None
    vx, _, _ = robust_fit(ts, xs)
    g = ground_xz(ctx, *tr.bottom)
    if g is None:
        return None
    _, _, f, _, yaw = pov_geom(ctx)
    return vx - g[1] * yaw / f, g[1]


def longitudinal_speed_ms(tr: Track, ctx: FrameContext, seconds: float = 1.0,
                          max_z: float = 30.0) -> tuple[float, float, float] | None:
    """Vận tốc dọc của CHÍNH xe kia (m/s; âm = đang chạy về phía camera) = dZ/dt + v_ego, kèm sai số và Z hiện tại."""
    v_ego = ctx.ego.ego_speed_mps if ctx.ego is not None else None
    if v_ego is None:
        return None
    ts, _, zs = ground_series(tr, ctx, seconds)
    if len(ts) < 5 or ts[-1] - ts[0] < 0.5 or zs[-1] > max_z:
        return None
    dz, sz, _ = robust_fit(ts, zs)
    return dz + v_ego, (sz if math.isfinite(sz) else 9.9), float(zs[-1])


def is_oncoming(tr: Track, ctx: FrameContext, ratio: float = 1.3, margin: float = 0.05, v_onc: float = 1.5,
                max_z: float = 30.0) -> tuple[bool, bool, float | None]:
    """(theo tốc độ nở hộp, theo vận tốc dọc, vận tốc dọc) — xe phía trước đang lao ngược chiều về phía camera."""
    s_static = ctx.ego.static_scale_rate(tr.xyxy[3]) if ctx.ego is not None else 0.0
    s_obj = tr.kin.scale_rate or 0.0
    by_scale = s_obj > max(0.2, ratio * s_static + margin) and s_static > 0
    lv = longitudinal_speed_ms(tr, ctx, 1.0, max_z)
    by_z = lv is not None and lv[0] + 2.0 * lv[1] < -v_onc
    return by_scale, by_z, (lv[0] if lv else None)


# ===================================================================== HZ-01
class TurnNoSignalRule(BaseRule):
    """Phương tiện phía trước chuyển hướng mà không bật xi-nhan.

    Rẽ/chuyển làn = vận tốc NGANG trên mặt đường (m/s, đã trừ yaw camera) vượt ngưỡng, HOẶC tỉ lệ rộng/cao của hộp
    tăng đột ngột (xe xoay ngang). Xi-nhan = chuỗi cường độ màu hổ phách ở hai góc đuôi xe dao động 1–2 Hz
    (TurnSignalTracker ghi `attrs["blink"]` = {"left": t_cuối_thấy_nháy|None, "right": ...}).
    """

    code = "HZ-01"
    name = "Chuyển hướng không bật tín hiệu"
    channels = ("pov",)
    default_severity = "nghi_van"
    requirements = (Requirement(("vehicles",), "model nền"),)
    defaults = {"trigger_s": 0.4, "window_s": 0.7, "lat_thr_ms": 0.8, "lat_thr_box": 0.6, "ar_jump": 1.3,
                "lookback_s": 2.5, "min_age_s": 1.0, "max_z_m": 40.0, "min_h_px": 16}

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        for tr in ctx.vehicles(MOTOR_VEHICLES):
            if tr.duration_s < self.p["min_age_s"] or tr.kin.n < 4 or tr.h < self.p["min_h_px"]:
                continue
            ls = lateral_speed_ms(tr, ctx, self.p["window_s"])
            if ls is not None:
                lat, z = ls
                if z > self.p["max_z_m"]:
                    continue
                lateral = abs(lat) > self.p["lat_thr_ms"]
                lat_txt = f"{lat:+.2f} m/s"
            else:
                lat = lateral_rel_speed(tr, ctx)
                lateral = abs(lat) > self.p["lat_thr_box"]
                lat_txt = f"{lat:+.2f} hộp/s"
            pts = tr.recent(1.0)
            ar_now = tr.w / max(tr.h, 1.0)
            b0 = pts[0].xyxy
            ar_then = (b0[2] - b0[0]) / max(b0[3] - b0[1], 1.0)
            turning = lateral or (ar_now / max(ar_then, 1e-6)) > self.p["ar_jump"]
            if not turning:
                continue
            side = "left" if lat < 0 else "right"
            blink = tr.attrs.get("blink", {}) or {}
            seen = [s for s in ("left", "right") if blink.get(s) is not None and ctx.t - blink[s] <= self.p["lookback_s"]]
            hit = not seen
            out.append(RuleResult(
                key=str(tr.id), hit=hit, primary_track_id=tr.id, track_ids=[tr.id], vehicle_cls=tr.cls,
                bbox=tr.xyxy, confidence=0.55,
                measurements={"huong_re": "trái" if side == "left" else "phải", "van_toc_ngang": lat_txt,
                              "xi_nhan": "không thấy" if hit else ", ".join(seen)},
                overlays=[track_box_overlay(tr, RED, "RẼ KHÔNG XI-NHAN")],
            ))
        return out


# ===================================================================== HZ-02
class BlockedSightlineRule(BaseRule):
    """Khuất tầm nhìn.

    POV: xe lớn (buýt/tải, hoặc ô tô rất gần) nằm ở rìa trái/phải, đứng yên hoặc chạy chậm so với nền, che
         một phần lớn khung → vẽ VÙNG NGUY CƠ từ mép trong của xe che ra lòng đường phía trước.
    CCTV (site audit): xe đứng yên lâu trong vùng tam giác tầm nhìn đã khai báo (cổng trường, lối ra hẻm).
    """

    code = "HZ-02"
    name = "Khuất tầm nhìn"
    channels = ("pov", "cctv")
    default_severity = "nguy_co"
    requirements = (Requirement(("vehicles",), "model nền"),)
    defaults = {"trigger_s": 0.6, "near_m": 15.0, "min_lat_m": 0.9, "max_lat_m": 7.0, "min_height_m": 1.3,
                "rel_static_tol": 0.35, "min_stationary_s": 20.0, "once_per_key": True}

    def scene_ready(self, channel: str) -> tuple[bool, str]:
        if channel == "cctv" and not self.scene.zones.sightline:
            return False, "chưa khai báo vùng tầm nhìn (zones.sightline)"
        return True, ""

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        return self._pov(ctx) if ctx.channel == "pov" else self._cctv(ctx)

    def _pov(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        W, H = ctx.frame_size
        _, hz, f, _, _ = pov_geom(ctx)
        for tr in ctx.vehicles(MOTOR_VEHICLES):
            g = ground_xz(ctx, *tr.bottom)
            if g is None:
                continue
            X, Z = g
            if Z > self.p["near_m"] or not (self.p["min_lat_m"] <= abs(X) <= self.p["max_lat_m"]):
                continue           # quá xa, hoặc nằm ngay trước mặt (là xe phía trước chứ không phải vật che bên đường)
            h_m = tr.h * Z / f
            if h_m < self.p["min_height_m"] and tr.cls not in LARGE_VEHICLES:
                continue           # thấp hơn một em nhỏ → không che khuất được
            s_static = ctx.ego.static_scale_rate(tr.xyxy[3]) if ctx.ego is not None else 0.0
            s_obj = tr.kin.scale_rate or 0.0
            near_static = s_static <= 0 or abs(s_obj - s_static) <= self.p["rel_static_tol"] * max(s_static, 0.2)
            left = X < 0
            x_in = tr.xyxy[2] if left else tr.xyxy[0]
            y_b = tr.xyxy[3]
            tip = (W * 0.5, max(hz + 0.15 * (H - hz), y_b - 0.35 * tr.h))
            zone = [(x_in, tr.xyxy[1] + 0.3 * tr.h), (x_in, y_b), (W * (0.62 if left else 0.38), y_b), tip]
            out.append(RuleResult(
                key=str(tr.id), hit=near_static, primary_track_id=None, track_ids=[tr.id], vehicle_cls=tr.cls,
                bbox=tr.xyxy, confidence=0.55,
                measurements={"ben": "trái" if left else "phải", "khoang_cach_m": round(Z, 1),
                              "lech_ngang_m": round(abs(X), 1), "chieu_cao_vat_che_m": round(h_m, 1)},
                notes=["Có thể có người/xe khuất sau phương tiện — giảm tốc, sẵn sàng dừng."],
                overlays=[track_box_overlay(tr, AMBER, "VẬT CHE"), Overlay("polygon", zone, RED, alpha=0.35, text="!")],
            ))
        return out

    def _cctv(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        if not hasattr(self, "_mem"):
            self._mem = StationaryMemory()
        cam_moving = bool(ctx.aux.get("camera_moving"))
        for tr in ctx.vehicles(MOTOR_VEHICLES):
            z = find_zone(tr.bottom, self.scene.zones.sightline, ctx.wall_time)
            if z is None:
                self._mem.release(tr.id)
                continue
            if tr.kin.n < 3:
                continue
            still = (tr.kin.speed_kmh is not None and tr.kin.speed_kmh < 2.0) or (
                tr.kin.speed_kmh is None and tr.kin.speed_norm < 0.15)
            if not still:
                if not cam_moving:
                    self._mem.release(tr.id)
                continue
            spot = self._mem.touch(tr, z.id, ctx.t)      # đồng hồ theo VỊ TRÍ: tracker đổi ID không đếm lại từ 0
            dur = ctx.t - spot["since"]
            key = f"{tr.id}:{z.id}"
            if spot["fired"]:
                self.fired.setdefault(key, ctx.t)
            hit = dur >= self.p["min_stationary_s"]
            if hit:
                spot["fired"] = True
            out.append(RuleResult(
                key=key, hit=hit, primary_track_id=tr.id,
                track_ids=[tr.id], vehicle_cls=tr.cls, bbox=tr.xyxy, instant=True, confidence=0.7,
                measurements={"vung_tam_nhin": z.name or z.id, "thoi_gian_dung_s": round(dur, 1)},
                overlays=[track_box_overlay(tr, AMBER, "CHE TẦM NHÌN"),
                          Overlay("polygon", list(z.polygon), RED, alpha=0.35, text="!")],
            ))
        self._mem.prune(ctx.t)
        return out


# ===================================================================== HZ-03
class SuddenEmergenceRule(BaseRule):
    """Người/xe xuất hiện bất ngờ từ sau vật che khuất.

    Điều kiện chung: track MỚI (≤ max_age_s), vị trí xuất hiện sát mép trái/phải của một phương tiện khác (vật che),
    đang di chuyển NGANG nhanh và ở gần. Phản ứng gần như tức thì (cần 2–3 khung để có vận tốc).
      * POV: vận tốc ngang tính trên mặt đường (m/s) và phải hướng VÀO lối đi của xe mình; ở gần trong near_m mét.
      * CCTV: chỉ xét NGƯỜI ĐI BỘ (không phải người ngồi trên xe) bước ra từ sau một xe ĐANG ĐỖ, đi cắt ngang trục
        đường, trong lòng đường đã khai báo — tránh báo giả hàng loạt khi đường đông (người trên xe máy xuất hiện
        liên tục bên cạnh ô tô).
    """

    code = "HZ-03"
    motion_sensitive = True
    name = "Xuất hiện bất ngờ"
    channels = ("pov", "cctv")
    default_severity = "nguy_co"
    requirements = (Requirement(("vehicles",), "model nền"),)
    defaults = {"trigger_s": 0.0, "max_age_s": 0.6, "min_frames": 3, "edge_gap": 0.35, "v_overlap": 0.3,
                "lat_thr": 0.8, "lat_thr_ms": 0.8, "near_y": 0.55, "near_m": 15.0, "min_cross_angle": 45.0,
                "pov_classes": ["person", "bicycle", "motorcycle"]}

    def scene_ready(self, channel: str) -> tuple[bool, str]:
        if channel == "cctv" and not self.scene.zones.roadway:
            return False, "chưa khai báo lòng đường (zones.roadway)"
        return True, ""

    def _occluder(self, ctx: FrameContext, tr: Track, first: np.ndarray, need_parked: bool) -> Track | None:
        for o in ctx.vehicles(MOTOR_VEHICLES):
            if o.id == tr.id or o.h < 0.8 * tr.h:
                continue
            if ctx.rider_of.get(tr.id) == o.id:
                continue
            if need_parked and is_moving(o):
                continue
            ov_top, ov_bot = max(first[1], o.xyxy[1]), min(first[3], o.xyxy[3])
            if (ov_bot - ov_top) < self.p["v_overlap"] * (first[3] - first[1]):
                continue
            if min(abs(first[2] - o.xyxy[0]), abs(first[0] - o.xyxy[2])) <= self.p["edge_gap"] * o.w:
                return o
        return None

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        W, H = ctx.frame_size
        pov = ctx.channel == "pov"
        walkers = None if pov else {p.id for p in walking_pedestrians(ctx)}
        for tr in ctx.tracks:
            if pov and tr.cls not in self.p["pov_classes"]:
                continue
            if not pov and (tr.cls != "person" or tr.id not in walkers):
                continue
            if tr.duration_s > self.p["max_age_s"] or len(tr.history) < self.p["min_frames"]:
                continue
            first = tr.history[0].xyxy
            occ = self._occluder(ctx, tr, first, need_parked=not pov)
            if occ is None:
                continue
            h0, h1 = tr.history[0], tr.history[-1]
            dt = max(1e-3, h1.t - h0.t)
            vx = ((h1.xyxy[0] + h1.xyxy[2]) - (h0.xyxy[0] + h0.xyxy[2])) / 2 / dt
            toward = True
            if pov:
                ls = lateral_speed_ms(tr, ctx, self.p["max_age_s"] + 0.2, min_pts=self.p["min_frames"], min_span=0.1)
                if ls is not None:
                    lat, z = ls
                    g = ground_xz(ctx, *tr.bottom)
                    X = g[0] if g is not None else 0.0
                    toward = abs(X) < 1.0 or (lat * X) < 0          # đang đi VÀO lối đi phía trước xe mình
                    fast = abs(lat) >= self.p["lat_thr_ms"]
                    near = z <= self.p["near_m"]
                    lat_txt = f"{lat:+.2f} m/s"
                    vx = lat
                else:
                    if ctx.ego is not None and ctx.ego.warp is not None:
                        vx -= ctx.ego.shift_at(*tr.center)[0] * ctx.fps
                    lat = vx / max(tr.w, 1.0)
                    fast = abs(lat) >= self.p["lat_thr"]
                    near = tr.xyxy[3] >= self.p["near_y"] * H
                    lat_txt = f"{lat:+.2f} hộp/s"
            else:
                lat = vx / max(tr.w, 1.0)
                fast = abs(lat) >= self.p["lat_thr"]
                near = find_zone(tr.bottom, self.scene.zones.roadway) is not None
                lat_txt = f"{lat:+.2f} hộp/s"
                lanes = [z for z in self.scene.zones.lanes if z.direction]
                if lanes:
                    lane = find_zone(tr.bottom, lanes) or lanes[0]
                    a, b = lane.direction
                    disp = (h1.ground_px[0] - h0.ground_px[0], h1.ground_px[1] - h0.ground_px[1])
                    if abs(disp[0]) + abs(disp[1]) > 1e-6:
                        ang = angle_deg(disp, (b[0] - a[0], b[1] - a[1]))
                        toward = min(ang, 180 - ang) >= self.p["min_cross_angle"]
            hit = fast and near and toward
            cx, cy = tr.center
            arrow = Overlay("arrow", [(cx, cy), (cx + np.sign(vx) * max(40.0, tr.w), cy)], RED, thickness=3)
            out.append(RuleResult(
                key=str(tr.id), hit=hit, primary_track_id=None, track_ids=[tr.id, occ.id], vehicle_cls=None,
                bbox=tr.xyxy, instant=True, confidence=0.6,
                measurements={"doi_tuong": tr.cls, "van_toc_ngang": lat_txt, "vat_che": occ.cls},
                overlays=[track_box_overlay(tr, RED, "LAO RA"), track_box_overlay(occ, AMBER, "VẬT CHE"), arrow],
            ))
        return out
