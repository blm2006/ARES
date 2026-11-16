"""Nhóm 2 — dùng cả CCTV và POV: RB-02, RB-03, RB-04, RB-05, RB-07, RB-08."""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import numpy as np

from traffic_safety_ai.core.types import (FOUR_WHEELERS, MOTOR_VEHICLES, VEHICLE_CLASSES, FrameContext, Overlay,
                                          Track)
from traffic_safety_ai.rules.base_rule import (BLUE, RED, BaseRule, Requirement, RuleResult, default_corridor,
                                               find_zone, is_moving, motion_arrow, track_box_overlay, union_of,
                                               world_or_px_disp, walking_pedestrians, touches_border)
from traffic_safety_ai.utils.geometry import (cosine_similarity, distance_point_segment, line_crossing,
                                              point_in_polygon, polyline_crossing, side_of_line, unit)

log = logging.getLogger(__name__)
VEHICLE_LEN_M = {"motorcycle": 1.9, "bicycle": 1.7, "car": 4.5, "bus": 10.5, "truck": 7.0}
VEHICLE_W_M = {"motorcycle": 0.8, "bicycle": 0.6, "car": 1.8, "bus": 2.5, "truck": 2.4}


def corridor_of(ctx: FrameContext) -> list[tuple[float, float]]:
    c = ctx.aux.get("corridor")
    if c:
        return c
    zones = ctx.scene.zones.ego_corridor
    if zones:
        return [tuple(p) for p in zones[0].polygon]
    w, h = ctx.frame_size
    return default_corridor(w, h, ctx.scene.pov.ego_corridor, ctx.scene.pov)


# ===================================================================== RB-02
class PhoneUseRule(BaseRule):
    """Dùng điện thoại khi điều khiển xe.

    CCTV: dựa trên `attrs["phone_obs"]` của người lái = deque[(t, tư_thế_nghe_gọi, điện_thoại_gần_tay, conf)]
          (PhoneEstimator: YOLO-pose kiểm tra cổ tay sát tai/mắt + dò điện thoại trên ảnh cắt thân trên).
    POV:  điện thoại xuất hiện ở nửa dưới khung hình (tay người lái) khi xe POV đang chạy; vùng giá đỡ điện thoại
          (pov.phone_mount_roi) được loại trừ để không báo nhầm khi chỉ xem bản đồ trên giá.
    Mặc định là "nghi vấn": góc cao nhìn không rõ, gãi đầu/chỉnh mũ cũng giống nghe điện thoại → cần người xem lại.
    """

    code = "RB-02"
    name = "Sử dụng điện thoại khi điều khiển xe"
    channels = ("cctv", "pov")
    default_severity = "nghi_van"
    requirements = (Requirement(("pose", "phone"), "model tư thế (pose) hoặc dò điện thoại"),)
    defaults = {"trigger_s": 1.2, "window_s": 1.5, "min_obs": 3, "min_frac": 0.6, "pov_cooldown_s": 30.0}

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        if ctx.channel == "pov":
            return self._pov(ctx)
        out: list[RuleResult] = []
        for moto_id, rider_ids in ctx.riders.items():
            moto = ctx.by_id(moto_id)
            if moto is None or not rider_ids or not is_moving(moto):
                continue
            driver = ctx.by_id(rider_ids[0])
            if driver is None:
                continue
            obs = [o for o in driver.attrs.get("phone_obs", ()) if o[0] >= ctx.t - self.p["window_s"]]
            if len(obs) < self.p["min_obs"]:
                continue
            pos = [o for o in obs if o[1] or o[2]]
            frac = len(pos) / len(obs)
            hit = frac >= self.p["min_frac"]
            kinds = []
            if any(o[1] for o in pos):
                kinds.append("tư thế áp điện thoại lên tai")
            if any(o[2] for o in pos):
                kinds.append("điện thoại gần tay")
            out.append(RuleResult(
                key=str(driver.id), hit=hit, primary_track_id=moto_id, track_ids=[moto_id, driver.id],
                vehicle_cls=moto.cls, bbox=union_of([moto, driver]), confidence=0.5 + 0.4 * frac,
                measurements={"ti_le_khung_co_dau_hieu": round(frac, 2), "dau_hieu": ", ".join(kinds) or "—"},
                notes=["Nhận dạng tư thế từ camera góc cao có thể nhầm với gãi đầu/chỉnh mũ — cần người xác minh."],
                overlays=[track_box_overlay(driver, RED, "ĐIỆN THOẠI?"), track_box_overlay(moto, RED)],
            ))
        return out

    def _pov(self, ctx: FrameContext) -> list[RuleResult]:
        ph = ctx.aux.get("ego_phone")
        moving = ctx.ego is not None and ctx.ego.moving
        if ph is None:
            return []
        seen, conf, box = ph
        hit = bool(seen) and moving
        ov = []
        if box is not None:
            ov.append(Overlay("box", [(box[0], box[1]), (box[2], box[3])], RED, thickness=3, text="ĐIỆN THOẠI"))
        self.p["once_per_key"] = False
        self.p["cooldown_s"] = self.p["pov_cooldown_s"]
        return [RuleResult(key="ego", hit=hit, primary_track_id=None, track_ids=[], vehicle_cls="motorcycle",
                           bbox=None if box is None else np.asarray(box, float), confidence=float(conf or 0.5),
                           measurements={"nguon": "camera hành trình (người lái POV)"}, overlays=ov,
                           notes=["Phát hiện điện thoại trong tay người lái xe gắn camera."])]


# ===================================================================== RB-03
class WrongWayRule(BaseRule):
    """Đi ngược chiều.

    CCTV: mỗi làn có mũi tên hướng đúng; véc-tơ chuyển động (hồi quy ~1 s, đổi sang mét nếu có hiệu chuẩn)
          có cosine < −0.5 (lệch > 120°) và quãng đi đủ dài → vi phạm.
    POV:  xe trong hành lang phía trước NỞ TO nhanh hơn mức một vật đứng yên ở cùng khoảng cách
          (so với dòng quang học nền, xem ego_motion.py) và đang đi xuống trong khung → đang lao ngược chiều tới.
          Có model hướng đầu/đuôi xe (models/vehicle_orientation.pt) thì dùng để khẳng định "nhìn thấy đầu xe".
    """

    code = "RB-03"
    motion_sensitive = True
    name = "Đi ngược chiều"
    channels = ("cctv", "pov")
    requirements = (Requirement(("vehicles",), "model nền"),)
    defaults = {"trigger_s": 0.8, "window_s": 1.0, "cos_thr": -0.5, "min_disp_m": 1.5, "min_disp_box": 0.6,
                "consensus_min": 3, "consensus_frac": 0.5, "consensus_window_s": 20.0,
                "pov_min_rate": 0.2, "pov_ratio": 1.3, "pov_margin": 0.05, "pov_v_onc": 1.5, "pov_max_z_m": 30.0}

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self._warned_at = -1e9
        self._lane_hist: dict[str, dict[int, tuple[float, bool]]] = {}   # làn → {id xe: (lần cuối thấy, đi ngược?)}

    def scene_ready(self, channel: str) -> tuple[bool, str]:
        if channel == "cctv" and not any(z.direction for z in self.scene.zones.lanes):
            return False, "chưa khai báo làn đường có mũi tên hướng đi (zones.lanes[].direction)"
        return True, ""

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        return self._pov(ctx) if ctx.channel == "pov" else self._cctv(ctx)

    def _cctv(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        lanes = [z for z in self.scene.zones.lanes if z.direction]
        W, H = ctx.frame_size
        for tr in ctx.vehicles(VEHICLE_CLASSES):
            lane = find_zone(tr.bottom, lanes, ctx.wall_time)
            if lane is None:
                continue
            if touches_border(tr.xyxy, W, H) or any(touches_border(p.xyxy, W, H) for p in tr.recent(self.p["window_s"])[:1]):
                continue               # hộp bị cắt ở mép khung → tâm/đáy hộp trượt theo phần xe lộ ra, không phải chuyển động thật
            disp, world = world_or_px_disp(tr, self.p["window_s"])
            if disp is None:
                continue
            mag = math.hypot(*disp)
            min_disp = self.p["min_disp_m"] if world else self.p["min_disp_box"] * tr.h
            if mag < min_disp:
                continue
            a, b = lane.direction
            if world and ctx.calib is not None:
                wa, wb = ctx.calib.to_world([a, b])
                expected = (wb[0] - wa[0], wb[1] - wa[1])
            else:
                expected = (b[0] - a[0], b[1] - a[1])
            cos = cosine_similarity(disp, expected)
            hit = cos < self.p["cos_thr"]
            ov = [track_box_overlay(tr, RED, "NGƯỢC CHIỀU"), Overlay("arrow", [tuple(a), tuple(b)], BLUE, thickness=3)]
            arr = motion_arrow(tr, self.p["window_s"])
            if arr:
                ov.append(arr)
            out.append(RuleResult(
                key=str(tr.id), hit=hit, primary_track_id=tr.id, track_ids=[tr.id], vehicle_cls=tr.cls, bbox=tr.xyxy,
                confidence=min(0.95, 0.6 + 0.4 * (-cos)), overlays=ov,
                measurements={"lan_duong": lane.name or lane.id, "cos_lech_huong": round(cos, 2),
                              "goc_lech_do": round(math.degrees(math.acos(max(-1, min(1, cos)))), 0),
                              "quang_duong": f"{mag:.1f} {'m' if world else 'px'}"},
            ))
        # Đồng thuận: thực tế chỉ LẺ TẺ vài xe đi ngược. Nếu trong consensus_window_s gần đây ĐA SỐ các xe khác nhau của
        # một làn đều "ngược chiều" (≥ consensus_min xe và > consensus_frac) → nhiều khả năng mũi tên hướng làn khai
        # báo ngược hoặc camera bị xoay → không báo, ghi cảnh báo. Tính theo XE KHÁC NHAU trong cửa sổ thời gian chứ
        # không theo từng khung: đường vắng mỗi lúc chỉ có 1–2 xe vẫn nhận ra được khai báo sai, và hai xe đi ngược
        # trong một làn đang trống không bị bỏ sót.
        win = float(self.p["consensus_window_s"])
        for r in out:
            lid = str(r.measurements.get("lan_duong"))
            self._lane_hist.setdefault(lid, {})[int(r.key)] = (ctx.t, bool(r.hit))
        for lid, hist in self._lane_hist.items():
            for k in [k for k, (tt, _) in hist.items() if ctx.t - tt > win]:
                del hist[k]
            n_all = len(hist)
            n_against = sum(1 for _, ag in hist.values() if ag)
            if n_against < self.p["consensus_min"] or n_against / max(n_all, 1) <= self.p["consensus_frac"]:
                continue
            suppressed = False
            for r in out:
                if r.hit and str(r.measurements.get("lan_duong")) == lid:
                    r.hit = False
                    suppressed = True
            if suppressed and ctx.t - self._warned_at > 30.0:
                self._warned_at = ctx.t
                log.warning("RB-03: %d/%d xe gần đây trong làn '%s' đi ngược mũi tên — bỏ qua (mũi tên hướng làn khai "
                            "báo ngược hoặc camera bị xoay?)", n_against, n_all, lid)
        return out

    def _pov(self, ctx: FrameContext) -> list[RuleResult]:
        from traffic_safety_ai.rules.pov_rules import is_oncoming

        out: list[RuleResult] = []
        corr = corridor_of(ctx)
        for tr in ctx.vehicles(MOTOR_VEHICLES):
            if not point_in_polygon(tr.bottom, corr) or tr.kin.scale_rate is None:
                continue
            s_obj = tr.kin.scale_rate
            s_static = ctx.ego.static_scale_rate(tr.xyxy[3]) if ctx.ego is not None else 0.0
            orient = tr.attrs.get("orientation")
            # hai tín hiệu: (1) hộp nở nhanh hơn vật đứng yên cùng khoảng cách; (2) vận tốc dọc của chính xe kia
            # = dZ/dt + v_ego âm rõ rệt (đang chạy về phía mình)
            by_scale, by_z, v_obj = is_oncoming(tr, ctx, self.p["pov_ratio"], self.p["pov_margin"],
                                                self.p["pov_v_onc"], self.p["pov_max_z_m"])
            hit = (by_scale or by_z) and tr.kin.vy_px > 0 and orient != "rear"
            conf = 0.55 + (0.3 if orient == "front" else 0.0) + (0.05 if by_scale and by_z else 0.0)
            meas = {"toc_do_no_hop": round(s_obj, 2), "muc_no_cua_vat_dung_yen": round(s_static, 2),
                    "huong_xe": orient or "chưa xác định"}
            if v_obj is not None:
                meas["van_toc_doc_xe_kia_ms"] = round(v_obj, 1)
            out.append(RuleResult(
                key=str(tr.id), hit=hit, primary_track_id=tr.id, track_ids=[tr.id], vehicle_cls=tr.cls, bbox=tr.xyxy,
                confidence=conf, severity=None if orient == "front" else "nghi_van", measurements=meas,
                notes=["Xe trong làn phía trước đang lao tới nhanh hơn vật đứng yên cùng khoảng cách."],
                overlays=[track_box_overlay(tr, RED, "NGƯỢC CHIỀU?"),
                          Overlay("polygon", list(corr), BLUE, alpha=0.12, thickness=1)],
            ))
        return out


# ===================================================================== RB-04
class LaneSidewalkRule(BaseRule):
    """Đi trên vỉa hè / đi sai làn / đè vạch liền.

    Khác spec: (1) chỉ xét xe ĐANG CHẠY (xe đỗ trên vỉa hè là lỗi khác — RB-11);
    (2) "đi qua vỉa hè để vào nhà, cơ quan, CỔNG TRƯỜNG" là được phép → khai báo `exempt_zones` (lối vào)
    và yêu cầu quãng đường đi DỌC vỉa hè ≥ min_travel_m; (3) vạch liền: cắt qua vạch rồi ở hẳn bên kia.
    """

    code = "RB-04"
    motion_sensitive = True
    name = "Đi trên vỉa hè / sai làn / đè vạch liền"
    channels = ("cctv", "pov")
    requirements = (Requirement(("vehicles",), "model nền"),)
    defaults = {"trigger_s": 0.8, "min_travel_m": 3.0, "min_travel_box": 1.5, "exempt_zones": [],
                "solid_settle_box": 0.15}

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self._entry: dict[str, tuple[float, tuple[float, float], bool]] = {}
        self._crossed: dict[str, float] = {}

    def scene_ready(self, channel: str) -> tuple[bool, str]:
        z = self.scene.zones
        if channel == "cctv" and not (z.sidewalks or z.restricted or self.scene.lines.solid_lines):
            return False, "chưa khai báo vỉa hè / làn hạn chế / vạch liền"
        return True, ""

    def check_available(self, capabilities: set[str], channel: str) -> tuple[bool, str]:
        ok, why = super().check_available(capabilities, channel)
        if ok and channel == "pov" and "lane_seg" not in capabilities:
            return False, "kênh POV cần model phân vùng vỉa hè/vạch (models/lane_seg.pt)"
        return ok, why

    def _track_state(self) -> tuple[dict, ...]:
        return (self._entry, self._crossed)

    def _travel(self, key: str, tr: Track, inside: bool) -> tuple[float, bool]:
        g_m = tr.ground_m
        g = g_m if g_m is not None else tr.bottom
        world = g_m is not None
        if not inside:
            self._entry.pop(key, None)
            return 0.0, world
        if key not in self._entry:
            self._entry[key] = (tr.last_t, g, world)
            return 0.0, world
        _, g0, w0 = self._entry[key]
        if w0 != world:
            self._entry[key] = (tr.last_t, g, world)
            return 0.0, world
        return math.hypot(g[0] - g0[0], g[1] - g0[1]), world

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        if ctx.channel == "pov":
            return self._pov(ctx)
        out: list[RuleResult] = []
        z = self.scene.zones
        exempt = [p for p in self.p.get("exempt_zones", []) if len(p) >= 3]
        for tr in ctx.vehicles(VEHICLE_CLASSES):
            g = tr.bottom
            moving = is_moving(tr)
            in_exempt = any(point_in_polygon(g, p) for p in exempt)
            # --- vỉa hè
            sw = find_zone(g, z.sidewalks, ctx.wall_time)
            key = f"{tr.id}:via_he"
            travel, world = self._travel(key, tr, sw is not None and moving and not in_exempt)
            need = self.p["min_travel_m"] if world else self.p["min_travel_box"] * tr.h
            if sw is not None and moving and not in_exempt:
                out.append(RuleResult(
                    key=key, hit=travel >= need, primary_track_id=tr.id, track_ids=[tr.id], vehicle_cls=tr.cls,
                    bbox=tr.xyxy, variant="via_he", confidence=0.8,
                    measurements={"khu_vuc": sw.name or sw.id, "quang_duong_tren_via_he": f"{travel:.1f} {'m' if world else 'px'}"},
                    overlays=[track_box_overlay(tr, RED, "VỈA HÈ"), Overlay("polygon", list(sw.polygon), RED, alpha=0.25)],
                ))
            # --- làn hạn chế theo loại xe
            rz = find_zone(g, [r for r in z.restricted if r.allowed_classes], ctx.wall_time)
            if rz is not None and moving:
                bad = tr.cls not in (rz.allowed_classes or [])
                out.append(RuleResult(
                    key=f"{tr.id}:sai_lan", hit=bad, primary_track_id=tr.id, track_ids=[tr.id], vehicle_cls=tr.cls,
                    bbox=tr.xyxy, variant="sai_lan", confidence=0.75,
                    measurements={"lan": rz.name or rz.id, "loai_xe_duoc_phep": ", ".join(rz.allowed_classes or [])},
                    overlays=[track_box_overlay(tr, RED, "SAI LÀN"), Overlay("polygon", list(rz.polygon), RED, alpha=0.2)],
                ))
            # --- vạch liền (camera đang rung: vị trí vạch trên ảnh lệch → không xét, không đụng trạng thái)
            if len(tr.history) >= 2 and not ctx.aux.get("camera_moving"):
                prev = tr.history[-2].ground_px
                for sl in self.scene.lines.solid_lines:
                    k = f"{tr.id}:de_vach:{sl.id}"
                    if polyline_crossing(prev, g, sl.points):
                        if k in self._crossed:
                            self._crossed.pop(k)        # cắt lần hai = quay lại bên cũ → chưa từng sang hẳn bên kia
                        else:
                            self._crossed[k] = ctx.t
                    t0 = self._crossed.get(k)
                    if t0 is None:
                        continue
                    pts = sl.points
                    dist = min(distance_point_segment(g, pts[i], pts[i + 1]) for i in range(len(pts) - 1))
                    settled = dist >= self.p["solid_settle_box"] * max(tr.w, 1.0)
                    if ctx.t - t0 > 3.0:
                        self._crossed.pop(k, None)
                        continue
                    if settled and moving:
                        out.append(RuleResult(
                            key=k, hit=True, primary_track_id=tr.id, track_ids=[tr.id], vehicle_cls=tr.cls,
                            bbox=tr.xyxy, variant="de_vach", confidence=0.7, instant=True,
                            measurements={"vach": sl.name or sl.id},
                            overlays=[track_box_overlay(tr, RED, "ĐÈ VẠCH"),
                                      Overlay("line", [tuple(p) for p in pts], RED, thickness=4)],
                        ))
                        self._crossed.pop(k, None)
        return out

    def _pov(self, ctx: FrameContext) -> list[RuleResult]:
        masks = ctx.aux.get("lane_masks") or {}
        sw = masks.get("sidewalk")
        if sw is None:
            return []
        out: list[RuleResult] = []
        H, W = sw.shape[:2]
        for tr in ctx.vehicles(VEHICLE_CLASSES):
            x, y = (int(min(W - 1, max(0, v))) for v in tr.bottom)
            inside = bool(sw[y, x])
            out.append(RuleResult(key=f"{tr.id}:via_he", hit=inside and is_moving(tr), primary_track_id=tr.id,
                                  track_ids=[tr.id], vehicle_cls=tr.cls, bbox=tr.xyxy, variant="via_he",
                                  confidence=0.6, severity="nghi_van",
                                  overlays=[track_box_overlay(tr, RED, "VỈA HÈ")]))
        # xe POV tự đi trên vỉa hè: vùng sát đáy giữa khung nằm trong mặt nạ vỉa hè
        band = sw[int(H * 0.9):, int(W * 0.4):int(W * 0.6)]
        if band.size and ctx.ego is not None:
            frac = float(band.mean())
            out.append(RuleResult(key="ego:via_he", hit=frac > 0.6 and ctx.ego.moving, vehicle_cls="motorcycle",
                                  variant="via_he", confidence=0.6, severity="nghi_van",
                                  measurements={"ti_le_banh_xe_tren_via_he": round(frac, 2)}))
        return out


# ===================================================================== RB-05
class RedLightRule(BaseRule):
    """Không chấp hành đèn tín hiệu (vượt đèn đỏ).

    CCTV, khác spec: dùng HAI vạch như hệ thống phạt nguội thực tế —
      (1) cắt vạch dừng theo đúng chiều khi đèn ĐỎ đã bật ≥ red_grace_s (bù độ trễ nhận màu đèn, thời điểm
          cắt được NỘI SUY giữa hai frame);
      (2) rồi phải đi qua vạch xác nhận phía bên kia giao lộ (hoặc đi tiếp ≥ confirm_distance_m) trong
          confirm_timeout_s → mới là vượt đèn đỏ. Xe chỉ nhích qua vạch dừng rồi dừng lại → lỗi nhẹ
          "dừng quá vạch" (tuỳ chọn report_stop_over_line).
      Làn được rẽ phải khi đèn đỏ: khai báo exempt_zones của vạch dừng.
    POV: đèn liên quan (to nhất nửa trên) đang ĐỎ, xe POV vẫn chạy và đèn trôi ra khỏi mép trên khung hình
         (đi qua chân đèn) → nghi vấn xe POV vượt đèn đỏ.
    """

    code = "RB-05"
    motion_sensitive = True
    name = "Vượt đèn đỏ"
    channels = ("cctv", "pov")
    requirements = (Requirement(("vehicles",), "model nền"),)
    defaults = {"trigger_s": 0.0, "red_grace_s": 0.4, "confirm_timeout_s": 6.0, "confirm_box_heights": 1.2,
                "yellow_is_violation": False, "report_stop_over_line": False, "pov_min_light_h": 0.025,
                "pov_lost_s": 0.6}

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self._cand: dict[str, dict[str, Any]] = {}
        self._pov_light: dict[str, Any] = {}

    def scene_ready(self, channel: str) -> tuple[bool, str]:
        if channel == "cctv":
            if not self.scene.lines.stop_lines:
                return False, "chưa khai báo vạch dừng (lines.stop_lines)"
            if not self.scene.traffic_lights:
                return False, "chưa khai báo vùng đèn tín hiệu (traffic_lights)"
            for sl in self.scene.lines.stop_lines:
                if self.scene.light_by_id(sl.light_id) is None:
                    return False, f"vạch dừng '{sl.id}' trỏ tới đèn '{sl.light_id}' không có trong traffic_lights"
        return True, ""

    def check_available(self, capabilities: set[str], channel: str) -> tuple[bool, str]:
        ok, why = super().check_available(capabilities, channel)
        if ok and channel == "pov" and "traffic_light_det" not in capabilities:
            return False, "model nền không có lớp đèn tín hiệu"
        return ok, why

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        return self._pov(ctx) if ctx.channel == "pov" else self._cctv(ctx)

    @staticmethod
    def _beyond(origin: tuple[float, float], g: tuple[float, float], line: list) -> bool:
        a, b = line[0], line[1]
        s0, s1 = side_of_line(origin, a, b), side_of_line(g, a, b)
        if s0 == 0 or s1 == 0 or s0 == s1:
            return False
        dx, dy = b[0] - a[0], b[1] - a[1]
        u = ((g[0] - a[0]) * dx + (g[1] - a[1]) * dy) / max(dx * dx + dy * dy, 1e-9)
        return -0.1 <= u <= 1.1                    # chỉ tính trong phạm vi vạch (xe rẽ ra ngoài hai đầu vạch thì không)

    @staticmethod
    def _forward_sign(sl: Any) -> int:
        a, b = sl.points[0], sl.points[1]
        s0 = side_of_line(sl.direction[0], a, b)
        s1 = side_of_line(sl.direction[1], a, b)
        if s0 == s1:
            return 0
        return 1 if s1 > s0 else -1

    def _cctv(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        if ctx.aux.get("camera_moving"):
            # camera rung/xoay: vạch dừng trên ảnh lệch khỏi vạch thật → không lập ứng viên mới và cũng KHÔNG tiêu
            # mất ứng viên đang chờ xác nhận (trước đây ứng viên bị huỷ ngầm nếu được xác nhận đúng lúc camera rung)
            return out
        trackers = ctx.aux.get("light_trackers", {})
        for sl in self.scene.lines.stop_lines:
            lt_cfg = self.scene.light_by_id(sl.light_id)
            lt = trackers.get(lt_cfg.id) if lt_cfg else None
            if lt is None:
                continue
            fwd = self._forward_sign(sl)
            for tr in ctx.vehicles(tuple(sl.applies_to)):
                if len(tr.history) < 2:
                    continue
                p0, p1 = tr.history[-2], tr.history[-1]
                key = f"{tr.id}:{sl.id}"
                d, frac = line_crossing(p0.ground_px, p1.ground_px, sl.points[0], sl.points[1], margin=0.05)
                if d != 0 and (fwd == 0 or d == fwd) and key not in self._cand:
                    t_cross = p0.t + frac * (p1.t - p0.t)
                    state = lt.state_at(t_cross)
                    red_for = lt.red_since(t_cross)
                    is_red = state == "red" and red_for is not None and red_for >= self.p["red_grace_s"]
                    is_yellow = state == "yellow" and self.p["yellow_is_violation"]
                    if is_red or is_yellow:
                        cx = p0.ground_px[0] + frac * (p1.ground_px[0] - p0.ground_px[0])
                        cy = p0.ground_px[1] + frac * (p1.ground_px[1] - p0.ground_px[1])
                        gm = None
                        if ctx.calib is not None:
                            gm = ctx.calib.point_to_world((cx, cy))
                        self._cand[key] = {"t": t_cross, "state": state, "red_for": red_for, "pt": (cx, cy),
                                           "pt_m": gm, "h": tr.h, "box": tr.xyxy.copy(), "frame": ctx.frame_idx}
                c = self._cand.get(key)
                if c is None:
                    continue
                g = p1.ground_px
                if any(point_in_polygon(g, z) for z in sl.exempt_zones):
                    self._cand.pop(key, None)       # rẽ phải được phép
                    continue
                confirmed = False
                if sl.confirm_points:
                    dc, _ = line_crossing(p0.ground_px, g, sl.confirm_points[0], sl.confirm_points[1], margin=0.1)
                    # ngoài khoảnh khắc cắt vạch: xe ĐÃ nằm bên kia vạch xác nhận (so với điểm cắt vạch dừng) cũng là
                    # đã đi qua — cần khi bước cắt vạch rơi vào lúc mất dấu hoặc lúc camera rung (bị bỏ qua)
                    confirmed = dc != 0 or self._beyond(c["pt"], g, sl.confirm_points)
                elif c["pt_m"] is not None and p1.ground_m is not None:
                    confirmed = math.dist(c["pt_m"], p1.ground_m) >= sl.confirm_distance_m
                else:
                    confirmed = math.dist(c["pt"], g) >= self.p["confirm_box_heights"] * max(c["h"], 1.0)
                expired = ctx.t - c["t"] > self.p["confirm_timeout_s"]
                if not confirmed and not expired:
                    continue
                self._cand.pop(key, None)
                variant = None if confirmed else "dung_qua_vach"
                if not confirmed and not self.p["report_stop_over_line"]:
                    continue
                ov = [track_box_overlay(tr, RED, "VƯỢT ĐÈN ĐỎ" if confirmed else "QUÁ VẠCH"),
                      Overlay("line", [tuple(sl.points[0]), tuple(sl.points[1])], RED, thickness=4, text="VẠCH DỪNG")]
                if sl.confirm_points:
                    ov.append(Overlay("line", [tuple(sl.confirm_points[0]), tuple(sl.confirm_points[1])], (0, 140, 255),
                                      thickness=2))
                x1, y1, x2, y2 = lt_cfg.roi
                ov.append(Overlay("box", [(x1, y1), (x2, y2)], (40, 40, 230), thickness=3, text="ĐÈN ĐỎ"))
                arr = motion_arrow(tr, 1.5)
                if arr:
                    ov.append(arr)
                out.append(RuleResult(
                    key=key, hit=True, primary_track_id=tr.id, track_ids=[tr.id], vehicle_cls=tr.cls, bbox=tr.xyxy,
                    variant=variant, instant=True, confidence=0.9 if confirmed else 0.6,
                    severity=None if confirmed else "nghi_van",
                    measurements={"trang_thai_den_khi_cat_vach": "ĐỎ" if c["state"] == "red" else "VÀNG",
                                  "den_do_da_bat_s": None if c["red_for"] is None else round(c["red_for"], 2),
                                  "vach_dung": sl.name or sl.id, "khung_cat_vach": c["frame"]},
                    overlays=ov,
                ))
        live = {f"{t.id}" for t in ctx.tracks}
        for k in [k for k in self._cand if k.split(":")[0] not in live and ctx.t - self._cand[k]["t"] > 2.0]:
            self._cand.pop(k, None)
        return out

    def _pov(self, ctx: FrameContext) -> list[RuleResult]:
        pl = ctx.aux.get("pov_light")
        st = self._pov_light
        W, H = ctx.frame_size
        moving = ctx.ego is not None and ctx.ego.moving
        if pl is not None:
            state, box = pl
            if state.state == "red" and (box[3] - box[1]) >= self.p["pov_min_light_h"] * H:
                st.update({"t": ctx.t, "box": np.asarray(box, float), "moving": moving})
            elif state.state in ("green", "yellow"):
                st.clear()
            return []
        if st and ctx.t - st["t"] >= self.p["pov_lost_s"]:
            box = st["box"]
            passed = box[1] < 0.25 * H and st["moving"] and moving
            self._pov_light = {}
            if passed:
                self.p["once_per_key"] = False
                return [RuleResult(key=f"ego:{int(st['t'] * 10)}", hit=True, vehicle_cls="motorcycle",
                                   instant=True, severity="nghi_van", confidence=0.55,
                                   measurements={"nguon": "camera hành trình"},
                                   notes=["Đèn đỏ ra khỏi khung phía trên khi xe POV vẫn chạy — cần xem lại clip."],
                                   overlays=[Overlay("box", [(box[0], box[1]), (box[2], box[3])], RED, 3, text="ĐÈN ĐỎ")])]
        return []


# ===================================================================== RB-07
class PedestrianConflictRule(BaseRule):
    """Xung đột với người đi bộ tại vạch qua đường.

    Hai chỉ số an toàn thay cho một ngưỡng khoảng cách đơn giản:
      * TTC/khoảng cách gần nhất dự báo (ngoại suy vận tốc hiện tại, horizon_s): xe sắp sượt qua người đang ở trên
        vạch mà KHÔNG giảm tốc (gia tốc ≥ −min_decel) → "không nhường đường" (nghi vấn).
      * PET (post-encroachment time): người vừa rời một ô của vạch, xe đi vào chính ô đó sau < pet_s giây
        → xung đột nguy hiểm đã xảy ra (chỉ số nguy cơ).
    POV: người đi bộ trong hành lang phía trước, TTC theo tốc độ nở hộp < pov_ttc_s → nguy cơ.
    """

    code = "RB-07"
    motion_sensitive = True
    name = "Xung đột với người đi bộ"
    channels = ("cctv", "pov")
    default_severity = "nghi_van"
    requirements = (Requirement(("persons",), "model nền (người)"),)
    defaults = {"trigger_s": 0.3, "horizon_s": 2.5, "d_safe_m": 1.5, "d_safe_box": 0.6, "min_speed_kmh": 8.0,
                "min_decel": 1.0, "buffer_m": 1.0, "pet_s": 1.0, "cell_m": 0.6, "cell_box": 0.35,
                "pov_ttc_s": 2.0}

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        # vệt người đi bộ trên vạch: id → deque[(t, vị trí, bán kính, đơn vị mét?)]
        self._trails: dict[int, deque[tuple[float, tuple[float, float], float, bool]]] = {}

    def scene_ready(self, channel: str) -> tuple[bool, str]:
        if channel == "cctv" and not self.scene.zones.crosswalks:
            return False, "chưa khai báo vạch qua đường (zones.crosswalks)"
        return True, ""

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        return self._pov(ctx) if ctx.channel == "pov" else self._cctv(ctx)

    def _pos_vel(self, tr: Track, world: bool) -> tuple[np.ndarray, np.ndarray]:
        if world and tr.ground_m is not None and tr.kin.vx_m is not None:
            return np.asarray(tr.ground_m), np.array([tr.kin.vx_m, tr.kin.vy_m])
        return np.asarray(tr.bottom), np.array([tr.kin.vx_px, tr.kin.vy_px])

    def _pet(self, v: Track, ctx: FrameContext) -> float | None:
        """PET nhỏ nhất: xe đang ở đúng chỗ một người đi bộ vừa RỜI đi bao nhiêu giây trước (None nếu không có).

        So khoảng cách thật giữa điểm chân xe và các điểm người đã đứng (mét nếu có hiệu chuẩn; không thì pixel với
        bán kính theo chiều cao NGƯỜI). Bản cũ chia lưới ô theo chiều cao hộp của từng đối tượng nên ô của người và
        ô của xe khác cỡ nhau, không so được khi chưa hiệu chuẩn.
        """
        best = None
        for trail in self._trails.values():
            for tp, pos, r, in_m in reversed(trail):
                g = v.ground_m if in_m else v.bottom
                if g is None or math.dist(g, pos) > r:
                    continue
                dt = ctx.t - tp                      # lần GẦN NHẤT người ở trong bán kính quanh chỗ xe đang đứng
                if 1e-6 < dt <= self.p["pet_s"] and (best is None or dt < best):
                    best = dt
                break                                # dt ≈ 0: người vẫn đang ở đó → không phải "vừa rời"
        return best

    def _cctv(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        cws = self.scene.zones.crosswalks
        world = ctx.calib is not None
        peds = []
        for p in walking_pedestrians(ctx):
            cw = find_zone(p.bottom, cws, ctx.wall_time)
            if cw is not None:
                peds.append((p, cw))
                in_m = world and p.ground_m is not None
                pos = p.ground_m if in_m else p.bottom
                r = self.p["cell_m"] if in_m else max(4.0, self.p["cell_box"] * p.h)
                self._trails.setdefault(p.id, deque(maxlen=120)).append((ctx.t, pos, float(r), in_m))
        horizon = max(5.0, float(self.p["pet_s"]))
        for pid in [k for k, tr in self._trails.items() if not tr or ctx.t - tr[-1][0] > horizon]:
            del self._trails[pid]
        vehs = [v for v in ctx.vehicles(VEHICLE_CLASSES) if is_moving(v, self.p["min_speed_kmh"], 0.4)]
        for v in vehs:
            # PET: xe đi vào chỗ người vừa rời
            if find_zone(v.bottom, cws, ctx.wall_time) is not None:
                pet = self._pet(v, ctx)
                if pet is not None:
                    out.append(RuleResult(
                        key=f"{v.id}:pet", hit=True, primary_track_id=v.id, track_ids=[v.id], vehicle_cls=v.cls,
                        bbox=v.xyxy, variant="pet", severity="nguy_co", instant=True, confidence=0.6,
                        measurements={"PET_s": round(pet, 2)},
                        overlays=[track_box_overlay(v, RED, "PET THẤP")]))
            for p, cw in peds:
                pp, vp = self._pos_vel(p, world)
                pv, vv = self._pos_vel(v, world)
                rel = pv - pp
                vrel = vv - vp
                vv2 = float(vrel @ vrel)
                if vv2 < 1e-9:
                    continue
                tca = float(np.clip(-(rel @ vrel) / vv2, 0.0, self.p["horizon_s"]))
                dca = float(np.linalg.norm(rel + vrel * tca))
                safe = self.p["d_safe_m"] if world else self.p["d_safe_box"] * max(v.h, p.h)
                approaching = (rel @ vrel) < 0
                decel = self._decelerating(v)
                # chưa đủ dữ liệu để biết xe có giảm tốc hay không → chưa kết luận
                hit = approaching and tca < self.p["horizon_s"] and dca <= safe and decel is False
                ov = [track_box_overlay(v, RED, f"TTC {tca:.1f}s"), track_box_overlay(p, (0, 200, 255), "NGƯỜI ĐI BỘ"),
                      Overlay("polygon", list(cw.polygon), RED, alpha=0.25)]
                arr = motion_arrow(v)
                if arr:
                    ov.append(arr)
                out.append(RuleResult(
                    key=f"{v.id}:{p.id}", hit=hit, primary_track_id=v.id, track_ids=[v.id, p.id], vehicle_cls=v.cls,
                    bbox=union_of([v, p]), confidence=0.65,
                    measurements={"TTC_s": round(tca, 2), "khoang_cach_gan_nhat": f"{dca:.2f} {'m' if world else 'px'}",
                                  "toc_do_xe_kmh": None if v.kin.speed_kmh is None else round(v.kin.speed_kmh, 1),
                                  "vach_qua_duong": cw.name or cw.id},
                    overlays=ov,
                ))
        return out

    def _decelerating(self, v: Track) -> bool | None:
        if v.kin.accel_ms2 is not None:
            return v.kin.accel_ms2 < -self.p["min_decel"]
        pts = v.recent(1.0)
        if len(pts) < 8 or pts[-1].t - pts[0].t < 0.6:
            return None
        mid = len(pts) // 2

        def spd(a, b):
            dt = max(1e-6, b.t - a.t)
            return math.dist(a.ground_px, b.ground_px) / dt

        v1, v2 = spd(pts[0], pts[mid]), spd(pts[mid], pts[-1])
        return v2 < 0.85 * v1

    def _pov(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        if ctx.ego is None or not ctx.ego.moving:
            return out
        corr = corridor_of(ctx)
        for p in walking_pedestrians(ctx):
            if not point_in_polygon(p.bottom, corr) or not p.kin.scale_rate or p.kin.scale_rate <= 0:
                continue
            ttc = 1.0 / p.kin.scale_rate
            out.append(RuleResult(key=str(p.id), hit=ttc < self.p["pov_ttc_s"], primary_track_id=None,
                                  track_ids=[p.id], vehicle_cls=None, bbox=p.xyxy, severity="nguy_co",
                                  confidence=0.6, measurements={"TTC_s": round(ttc, 2)},
                                  overlays=[track_box_overlay(p, RED, f"TTC {ttc:.1f}s")]))
        return out


# ===================================================================== RB-08
class FollowingRule(BaseRule):
    """Dừng đột ngột / bám đuôi không an toàn — CHỈ SỐ NGUY CƠ (luật chỉ phạt khi có va chạm hoặc có biển cự ly).

    CCTV (cần hiệu chuẩn mét): phanh gấp = gia tốc < −a_thr khi đang chạy > v_min;
      bám đuôi = cùng làn, khoảng cách thời gian (time headway = khe hở / tốc độ xe sau) < headway_s ở tốc độ cao.
    POV: TTC với xe ngay phía trước = 1 / (d ln w / dt) < ttc_s; bám sát = xe trước chiếm > area_frac khung khi đang chạy.
    """

    code = "RB-08"
    motion_sensitive = True
    name = "Dừng đột ngột / bám đuôi"
    channels = ("cctv", "pov")
    default_severity = "nguy_co"
    requirements = (Requirement(("vehicles",), "model nền"),)
    defaults = {"trigger_s": 1.5, "a_thr": 4.0, "v_min_kmh": 15.0, "min_drop_ms": 2.0, "min_track_s": 1.2,
                "headway_s": 1.0, "v_follow_min_kmh": 25.0, "lat_tol_m": 0.4, "ttc_s": 1.5, "area_frac": 0.12,
                "pov_headway_s": 0.8}

    def scene_ready(self, channel: str) -> tuple[bool, str]:
        if channel == "cctv" and not self.scene.calibration.enabled:
            return False, "cần hiệu chuẩn mét (calibration) để đo gia tốc/khoảng cách"
        return True, ""

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        return self._pov(ctx) if ctx.channel == "pov" else self._cctv(ctx)

    def _cctv(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        vehs = [v for v in ctx.vehicles(MOTOR_VEHICLES) if v.ground_m is not None and v.kin.speed_kmh is not None]
        for v in vehs:
            a = v.kin.accel_ms2
            if a is None or v.duration_s < self.p["min_track_s"]:
                continue
            if ctx.calib is not None and not ctx.calib.in_calibrated_area(v.bottom):
                continue                       # ngoài vùng hiệu chuẩn → sai số mét lớn
            hist = v.attrs.get("_speed_hist") or ()
            win = [s for t, s in hist if t >= ctx.t - 1.2]
            v_before = max(win, default=0.0)
            v_now = v.kin.speed_kmh / 3.6
            drop = v_before - v_now
            # cần đồng thời: gia tốc âm vượt ngưỡng, có ý nghĩa thống kê, và tốc độ THỰC SỰ giảm rõ rệt
            sig = v.kin.accel_sigma_ms2 if v.kin.accel_sigma_ms2 is not None else 0.0
            hit = (a < -self.p["a_thr"] and a + 2.0 * sig < -0.5 * self.p["a_thr"]
                   and v_before * 3.6 >= self.p["v_min_kmh"] and drop >= self.p["min_drop_ms"])
            out.append(RuleResult(key=f"{v.id}:phanh", hit=hit, primary_track_id=v.id, track_ids=[v.id],
                                  vehicle_cls=v.cls, bbox=v.xyxy, variant="phanh_gap", instant=True, confidence=0.6,
                                  measurements={"gia_toc_ms2": round(a, 2),
                                                "toc_do_truoc_kmh": round(v_before * 3.6, 1),
                                                "toc_do_sau_kmh": round(v_now * 3.6, 1)},
                                  overlays=[track_box_overlay(v, RED, "PHANH GẤP")]))
        lanes = self.scene.zones.lanes
        for f in vehs:
            if f.kin.speed_kmh < self.p["v_follow_min_kmh"] or f.kin.vx_m is None:
                continue
            u = unit((f.kin.vx_m, f.kin.vy_m))
            lane_f = find_zone(f.bottom, lanes) if lanes else None
            best = None
            for l in vehs:
                if l.id == f.id:
                    continue
                d = np.asarray(l.ground_m) - np.asarray(f.ground_m)
                along = float(d @ u)
                lat = abs(float(d[0] * -u[1] + d[1] * u[0]))
                if along <= 0:
                    continue
                # cùng "vệt" chuyển động: lệch ngang không quá nửa tổng bề rộng hai xe + dung sai
                lat_max = 0.5 * (VEHICLE_W_M.get(l.cls, 1.8) + VEHICLE_W_M.get(f.cls, 1.8)) + self.p["lat_tol_m"]
                if lat > lat_max:
                    continue
                if lanes and (lane_f is None or find_zone(l.bottom, lanes) is not lane_f):
                    continue
                if l.kin.vx_m is not None and float(np.dot(unit((l.kin.vx_m, l.kin.vy_m)), u)) < 0.5 \
                        and l.kin.moving:
                    continue                   # xe phía trước đi hướng khác (cắt ngang/ngược chiều) → không phải bám đuôi
                if best is None or along < best[1]:
                    best = (l, along)
            if best is None:
                continue
            l, along = best
            gap = max(0.0, along - 0.5 * (VEHICLE_LEN_M.get(l.cls, 4.0) + VEHICLE_LEN_M.get(f.cls, 4.0)))
            headway = gap / max(f.kin.speed_kmh / 3.6, 0.1)
            out.append(RuleResult(key=f"{f.id}:bam", hit=headway < self.p["headway_s"], primary_track_id=f.id,
                                  track_ids=[f.id, l.id], vehicle_cls=f.cls, bbox=union_of([f, l]), variant="bam_duoi",
                                  confidence=0.55,
                                  measurements={"khoang_cach_thoi_gian_s": round(headway, 2), "khe_ho_m": round(gap, 1),
                                                "toc_do_kmh": round(f.kin.speed_kmh, 1)},
                                  overlays=[track_box_overlay(f, RED, f"{headway:.1f}s"), track_box_overlay(l, BLUE)]))
        return out

    def _pov(self, ctx: FrameContext) -> list[RuleResult]:
        from traffic_safety_ai.rules.pov_rules import is_oncoming

        corr = corridor_of(ctx)
        W, H = ctx.frame_size
        lead = None
        for v in ctx.vehicles(VEHICLE_CLASSES):
            if not point_in_polygon(v.bottom, corr):
                continue
            if any(is_oncoming(v, ctx)[:2]):
                continue               # xe đang lao ngược chiều tới → thuộc RB-03, không phải "xe phía trước"
            if lead is None or v.xyxy[3] > lead.xyxy[3]:
                lead = v
        if lead is None or ctx.ego is None or not ctx.ego.moving:
            return []
        out: list[RuleResult] = []
        if lead.kin.scale_rate is not None and lead.kin.scale_rate > 0:
            ttc = 1.0 / lead.kin.scale_rate
            out.append(RuleResult(key=f"{lead.id}:ttc", hit=ttc < self.p["ttc_s"], primary_track_id=lead.id,
                                  track_ids=[lead.id], vehicle_cls=lead.cls, bbox=lead.xyxy, variant="ttc_thap",
                                  instant=True, confidence=0.6, measurements={"TTC_s": round(ttc, 2)},
                                  overlays=[track_box_overlay(lead, RED, f"TTC {ttc:.1f}s")]))
        from traffic_safety_ai.rules.pov_rules import ground_xz

        g = ground_xz(ctx, *lead.bottom)
        v_ego = ctx.ego.ego_speed_mps
        if g is not None and v_ego is not None and v_ego > 2.0:
            gap = max(0.0, g[1] - 0.5 * VEHICLE_LEN_M.get(lead.cls, 4.0))
            headway = gap / v_ego
            out.append(RuleResult(key=f"{lead.id}:bam", hit=headway < self.p["pov_headway_s"], primary_track_id=lead.id,
                                  track_ids=[lead.id], vehicle_cls=lead.cls, bbox=lead.xyxy, variant="bam_duoi",
                                  confidence=0.5, measurements={"khoang_cach_thoi_gian_s": round(headway, 2),
                                                                "khe_ho_m": round(gap, 1),
                                                                "toc_do_xe_minh_kmh": round(v_ego * 3.6, 1)},
                                  overlays=[track_box_overlay(lead, RED, f"BÁM SÁT {headway:.1f}s")]))
            return out
        frac = (lead.w * lead.h) / float(W * H)
        out.append(RuleResult(key=f"{lead.id}:bam", hit=frac > self.p["area_frac"], primary_track_id=lead.id,
                              track_ids=[lead.id], vehicle_cls=lead.cls, bbox=lead.xyxy, variant="bam_duoi",
                              confidence=0.5, measurements={"ti_le_dien_tich_xe_truoc": round(frac, 3)},
                              overlays=[track_box_overlay(lead, RED, "BÁM SÁT")]))
        return out


FOUR = FOUR_WHEELERS  # re-export cho test
