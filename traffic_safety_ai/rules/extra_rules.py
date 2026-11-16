"""Rule bổ sung (ngoài 13 mã của spec) — hay gặp ở cổng trường:

  RB-11  Dừng/đỗ sai quy định: trên vạch qua đường, nơi cấm dừng/đỗ, vỉa hè trước cổng.
  RB-12  Đi dàn hàng ngang từ 03 xe trở lên (học sinh đi xe đạp/xe máy điện thành hàng).
  HZ-04  Người đi bộ băng qua lòng đường ngoài vạch → dữ liệu đề xuất vị trí qua đường an toàn (KHÔNG xử phạt trẻ em).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from traffic_safety_ai.core.types import MOTOR_VEHICLES, TWO_WHEELERS, VEHICLE_CLASSES, FrameContext, Overlay
from traffic_safety_ai.rules.base_rule import (AMBER, RED, BaseRule, Requirement, RuleResult, StationaryMemory,
                                               find_zone, track_box_overlay, union_of, walking_pedestrians)
from traffic_safety_ai.utils.geometry import angle_deg

DEFAULT_LIMITS_S = {"crosswalk": 10.0, "no_stopping": 15.0, "no_parking": 60.0, "sidewalk": 60.0}


# ===================================================================== RB-11
class ParkingRule(BaseRule):
    code = "RB-11"
    name = "Dừng, đỗ xe sai quy định"
    channels = ("cctv",)
    requirements = (Requirement(("vehicles",), "model nền"),)
    defaults = {"trigger_s": 0.0, "still_kmh": 2.0, "still_norm": 0.12, "limits_s": DEFAULT_LIMITS_S,
                "classes": list(VEHICLE_CLASSES), "reacquire_s": 10.0}

    def scene_ready(self, channel: str) -> tuple[bool, str]:
        if not self.scene.zones.no_parking:
            return False, "chưa khai báo vùng cấm dừng/đỗ (zones.no_parking)"
        return True, ""

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self._mem = StationaryMemory(float(self.p["reacquire_s"]))

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        cam_moving = bool(ctx.aux.get("camera_moving"))
        for tr in ctx.vehicles(tuple(self.p["classes"])):
            z = find_zone(tr.bottom, self.scene.zones.no_parking, ctx.wall_time)
            if z is None:
                self._mem.release(tr.id)
                continue
            if tr.kin.n < 3:
                continue                       # track vừa sinh: chưa đủ dữ liệu để biết đang đứng hay đi
            still = (tr.kin.speed_kmh < self.p["still_kmh"]) if tr.kin.speed_kmh is not None else (
                tr.kin.speed_norm < self.p["still_norm"])
            if not still:
                if not cam_moving:             # camera rung làm mọi xe "trôi" → giữ nguyên đồng hồ, không đếm lại
                    self._mem.release(tr.id)
                continue
            spot = self._mem.touch(tr, z.id, ctx.t)
            dur = ctx.t - spot["since"]
            kind = z.kind or "no_parking"
            limit = z.max_stationary_s or float(self.p["limits_s"].get(kind, 60.0))
            key = f"{tr.id}:{z.id}"
            if spot["fired"]:
                self.fired.setdefault(key, ctx.t)     # cùng một xe (ID cũ đã bị báo) → không lập hồ sơ thứ hai
            hit = dur >= limit
            if hit:
                spot["fired"] = True
            out.append(RuleResult(
                key=key, hit=hit, primary_track_id=tr.id, track_ids=[tr.id],
                vehicle_cls=tr.cls, bbox=tr.xyxy, variant=kind, instant=True, confidence=0.8,
                measurements={"vi_tri": z.name or z.id, "loai_vung": kind, "thoi_gian_dung_s": round(dur, 1),
                              "nguong_s": limit},
                overlays=[track_box_overlay(tr, RED, f"ĐỖ {dur:.0f}s"),
                          Overlay("polygon", list(z.polygon), RED, alpha=0.25)],
            ))
        self._mem.prune(ctx.t)
        return out


# ===================================================================== RB-12
class AbreastRule(BaseRule):
    code = "RB-12"
    motion_sensitive = True
    name = "Đi dàn hàng ngang từ 03 xe trở lên"
    channels = ("cctv",)
    requirements = (Requirement(("vehicles",), "model nền"),)
    defaults = {"trigger_s": 2.5, "heading_deg": 20.0, "speed_ratio": 0.7, "long_tol_m": 1.5, "lat_min_m": 0.35,
                "lat_max_m": 2.2, "long_tol_box": 0.6, "lat_min_box": 0.3, "lat_max_box": 1.8, "min_group": 3}

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self._groups: dict[str, set[int]] = {}       # khoá nhóm ("<id nhỏ nhất lúc lập>:g") → các thành viên đã thấy

    def _group_key(self, comp: set[int]) -> str:
        """Nhóm thêm/bớt một xe vẫn là NHÓM CŨ (trùng ≥ 2 thành viên) → cùng khoá → chỉ một hồ sơ, điểm tích luỹ
        không bị chia nhỏ theo từng tổ hợp ID."""
        for k, members in self._groups.items():
            if len(members & comp) >= 2:
                members |= comp
                return k
        k = f"{min(comp)}:g"
        self._groups[k] = set(comp)
        return k

    def forget_tracks(self, track_ids) -> None:  # type: ignore[no-untyped-def]
        gone = set(track_ids)
        for k in [k for k, m in self._groups.items() if not (m - gone)]:
            del self._groups[k]
            for d in (self.scores, self.last_t, self.fired):
                d.pop(k, None)
        for m in self._groups.values():
            m -= gone

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        cyc = [t for t in ctx.vehicles(TWO_WHEELERS) if t.kin.speed_norm > 0.3 or (t.kin.speed_kmh or 0) > 5]
        if len(cyc) < self.p["min_group"]:
            return []
        world = ctx.calib is not None and all(t.ground_m is not None and t.kin.vx_m is not None for t in cyc)
        pos = {t.id: np.asarray(t.ground_m if world else t.bottom, float) for t in cyc}
        vel = {t.id: np.asarray((t.kin.vx_m, t.kin.vy_m) if world else (t.kin.vx_px, t.kin.vy_px), float) for t in cyc}
        adj: dict[int, set[int]] = {t.id: set() for t in cyc}
        for i, a in enumerate(cyc):
            for b in cyc[i + 1:]:
                va, vb = vel[a.id], vel[b.id]
                na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
                if na < 1e-6 or nb < 1e-6 or min(na, nb) / max(na, nb) < self.p["speed_ratio"]:
                    continue
                if angle_deg(va, vb) > self.p["heading_deg"]:
                    continue
                u = (va / na + vb / nb)
                u = u / max(float(np.linalg.norm(u)), 1e-9)
                d = pos[b.id] - pos[a.id]
                along = abs(float(d @ u))
                lat = abs(float(d[0] * -u[1] + d[1] * u[0]))
                if world:
                    ok = along <= self.p["long_tol_m"] and self.p["lat_min_m"] <= lat <= self.p["lat_max_m"]
                else:
                    s = 0.5 * (a.h + b.h)
                    ok = along <= self.p["long_tol_box"] * s and self.p["lat_min_box"] * s <= lat <= self.p["lat_max_box"] * s
                if ok:
                    adj[a.id].add(b.id)
                    adj[b.id].add(a.id)
        out: list[RuleResult] = []
        seen: set[int] = set()
        by_id = {t.id: t for t in cyc}
        for start in adj:
            if start in seen or not adj[start]:
                continue
            comp, stack = set(), [start]
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                stack.extend(adj[x] - comp)
            seen |= comp
            if len(comp) < self.p["min_group"]:
                continue
            members = [by_id[i] for i in sorted(comp)]
            cls = "bicycle" if sum(m.cls == "bicycle" for m in members) > len(members) / 2 else "motorcycle"
            key = self._group_key(comp)
            out.append(RuleResult(
                key=key, hit=True, primary_track_id=members[0].id, track_ids=[m.id for m in members], vehicle_cls=cls,
                bbox=union_of(members), confidence=0.6, measurements={"so_xe_hang_ngang": len(members)},
                notes=["Biển số xe đầu tiên trong nhóm được dùng làm đại diện — các xe còn lại xem trong ảnh."],
                overlays=[track_box_overlay(m, RED) for m in members] + [
                    Overlay("box", [(union_of(members)[0], union_of(members)[1]),
                                    (union_of(members)[2], union_of(members)[3])], AMBER, 2, text=f"{len(members)} XE")],
            ))
        return out


# ===================================================================== HZ-04
class JaywalkRule(BaseRule):
    """Người đi bộ băng ngang lòng đường ngoài vạch.

    Chống nhầm người ngồi trên xe là người đi bộ (lúc mới xuất hiện, phép gán người–xe chưa kịp chốt):
      * bỏ qua người có phần thân dưới chồng lên hộp xe máy/xe đạp;
      * tốc độ phải ở mức đi bộ/chạy (≤ max_walk_ms) khi có hiệu chuẩn;
      * hướng dịch chuyển phải CẮT NGANG trục đường (góc với hướng làn ≥ min_cross_angle) nếu có khai báo làn.
    """

    code = "HZ-04"
    motion_sensitive = True
    name = "Đi bộ băng qua đường ngoài vạch"
    channels = ("cctv",)
    default_severity = "nguy_co"
    requirements = (Requirement(("persons",), "model nền (người)"),)
    defaults = {"trigger_s": 0.0, "min_time_s": 1.2, "min_cross_m": 2.5, "min_cross_box": 1.0,
                "max_walk_ms": 3.2, "min_cross_angle": 40.0, "rider_ioa": 0.25}

    def scene_ready(self, channel: str) -> tuple[bool, str]:
        if not self.scene.zones.roadway:
            return False, "chưa khai báo lòng đường (zones.roadway)"
        return True, ""

    def _road_axis(self, ctx: FrameContext, pt: tuple[float, float], world: bool) -> np.ndarray | None:
        lanes = [z for z in self.scene.zones.lanes if z.direction]
        if not lanes:
            return None
        lane = find_zone(pt, lanes) or lanes[0]
        a, b = lane.direction
        if world and ctx.calib is not None:
            a, b = ctx.calib.point_to_world(a), ctx.calib.point_to_world(b)
        v = np.asarray(b, float) - np.asarray(a, float)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else None

    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        out: list[RuleResult] = []
        walkers = {p.id for p in walking_pedestrians(ctx, self.p["max_walk_ms"], self.p["rider_ioa"])}
        for p in ctx.pedestrians():
            road = find_zone(p.bottom, self.scene.zones.roadway, ctx.wall_time)
            on_cw = find_zone(p.bottom, self.scene.zones.crosswalks, ctx.wall_time) is not None
            if road is None or on_cw or p.id not in walkers:
                p.attrs.pop("_road_entry", None)      # người ngồi trên xe / chạy quá nhanh → không tính
                continue
            g = p.ground_m if p.ground_m is not None else p.bottom
            world = p.ground_m is not None
            entry = p.attrs.setdefault("_road_entry", (ctx.t, g, world))
            if entry[2] != world:
                p.attrs["_road_entry"] = entry = (ctx.t, g, world)
            dur = ctx.t - entry[0]
            d = np.asarray(g, float) - np.asarray(entry[1], float)
            disp = float(np.linalg.norm(d))
            need = self.p["min_cross_m"] if world else self.p["min_cross_box"] * p.h
            crossing = True
            ang = None
            axis = self._road_axis(ctx, p.bottom, world)
            if axis is not None and disp > 1e-6:
                ang = angle_deg(d, axis)
                ang = min(ang, 180.0 - ang)            # 0° = dọc đường, 90° = cắt ngang
                crossing = ang >= self.p["min_cross_angle"]
            hit = dur >= self.p["min_time_s"] and disp >= need and crossing
            meas = {"thoi_gian_trong_long_duong_s": round(dur, 1),
                    "quang_duong": f"{disp:.1f} {'m' if world else 'px'}",
                    "_x": round(p.bottom[0], 1), "_y": round(p.bottom[1], 1)}
            if ang is not None:
                meas["goc_cat_ngang_do"] = round(ang)
            out.append(RuleResult(
                key=str(p.id), hit=hit, primary_track_id=None, track_ids=[p.id], vehicle_cls=None, bbox=p.xyxy,
                instant=True, confidence=0.7, measurements=meas,
                notes=["Dữ liệu dùng để đề xuất vị trí vạch/đèn qua đường cho học sinh — không nhằm xử phạt."],
                overlays=[track_box_overlay(p, AMBER, "QUA ĐƯỜNG NGOÀI VẠCH"),
                          Overlay("polygon", list(road.polygon), RED, alpha=0.12)],
            ))
        return out


_ = MOTOR_VEHICLES
