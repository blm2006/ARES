"""Lớp nền cho mọi rule + cơ chế chống báo giả.

Chống báo giả (theo spec, nhưng đổi sang đơn vị THỜI GIAN để không phụ thuộc FPS):
  điểm(key) += Δt khi frame có dấu hiệu vi phạm, −= decay_ratio·Δt khi không;
  phát sự kiện khi điểm ≥ trigger_s (giây "vi phạm tích luỹ"). Mặc định decay_ratio = 0.5 như spec.
Sự kiện tức thời (instant) — vd xe đã cắt qua vạch dừng lúc đèn đỏ và đi hết giao lộ — bỏ qua tích luỹ.
Mỗi key (thường là id track) mặc định chỉ phát MỘT sự kiện (once_per_key), hoặc theo cooldown_s.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from traffic_safety_ai.core.config import SiteConfig, ZoneCfg
from traffic_safety_ai.core.types import FrameContext, Overlay, Track, ViolationEvent
from traffic_safety_ai.rules.catalog import LegalCatalog
from traffic_safety_ai.utils.geometry import ioa, point_in_polygon, union_box

RED = (53, 57, 229)          # #E53935 (BGR)
BLUE = (229, 136, 30)        # #1E88E5 (BGR) — hướng đúng quy định
AMBER = (0, 170, 255)


@dataclass
class RuleResult:
    key: str
    hit: bool
    primary_track_id: int | None = None
    track_ids: list[int] = field(default_factory=list)
    vehicle_cls: str | None = None
    bbox: np.ndarray | None = None
    variant: str | None = None
    severity: str | None = None
    confidence: float = 0.8
    measurements: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    overlays: list[Overlay] = field(default_factory=list)
    instant: bool = False


@dataclass
class Requirement:
    """Nhóm năng lực: cần ÍT NHẤT MỘT trong các năng lực liệt kê."""

    any_of: tuple[str, ...]
    label: str


class BaseRule(ABC):
    code: str = "XX-00"
    name: str = ""
    name_en: str = ""
    channels: tuple[str, ...] = ("cctv",)
    default_severity: str = "vi_pham"
    requirements: tuple[Requirement, ...] = ()
    defaults: dict[str, Any] = {}
    # True: rule dựa trên hướng/vận tốc trong ảnh → tạm ngưng khi camera CCTV đang rung/xoay (ctx.aux["camera_moving"])
    motion_sensitive: bool = False
    base_defaults: dict[str, Any] = {"trigger_s": 0.5, "decay_ratio": 0.5, "once_per_key": True,
                                     "cooldown_s": 60.0, "max_dt": 0.25, "enabled": True}

    def __init__(self, scene: SiteConfig, catalog: LegalCatalog, params: dict[str, Any] | None = None):
        self.scene = scene
        self.catalog = catalog
        self.p: dict[str, Any] = {**self.base_defaults, **self.defaults, **(params or {})}
        # tham số gõ sai tên trong YAML sẽ bị bỏ qua lặng lẽ → ghi lại để lệnh `check` cảnh báo
        self.unknown_params = sorted(set(params or {}) - set(self.base_defaults) - set(self.defaults) - {"severity"})
        if catalog is not None and catalog.name(self.code, ""):
            self.name = catalog.name(self.code, self.name)
            self.name_en = catalog.name_en(self.code) or self.name_en
        self.scores: dict[str, float] = {}
        self.last_t: dict[str, float] = {}
        self.fired: dict[str, float] = {}
        self.live_overlays: list[Overlay] = []
        self.live_suspects: set[int] = set()

    # ------------------------------------------------------------- khả dụng
    def scene_ready(self, channel: str) -> tuple[bool, str]:
        """Rule có đủ cấu hình địa điểm để chạy không (vd cần đa giác vỉa hè). Mặc định: có."""
        return True, ""

    def check_available(self, capabilities: set[str], channel: str) -> tuple[bool, str]:
        if channel not in self.channels:
            return False, f"chỉ áp dụng cho kênh {', '.join(c.upper() for c in self.channels)}"
        if not self.p.get("enabled", True):
            return False, "đã tắt trong cấu hình"
        for req in self.requirements:
            if not any(c in capabilities for c in req.any_of):
                return False, f"thiếu {req.label}"
        return self.scene_ready(channel)

    # ------------------------------------------------------------- vòng chính
    @abstractmethod
    def observe(self, ctx: FrameContext) -> list[RuleResult]:
        ...

    def evaluate(self, ctx: FrameContext) -> list[ViolationEvent]:
        results = self.observe(ctx)
        if self.motion_sensitive and ctx.aux.get("camera_moving"):
            for r in results:          # vẫn cập nhật điểm (giảm dần) nhưng không coi khung này là có dấu hiệu
                r.hit = False
        events: list[ViolationEvent] = []
        seen: set[str] = set()
        overlays: list[Overlay] = []
        suspects: set[int] = set()
        max_dt = float(self.p["max_dt"])
        for r in results:
            seen.add(r.key)
            dt = min(max_dt, max(1.0 / max(ctx.fps, 1.0), ctx.t - self.last_t.get(r.key, ctx.t - 1.0 / max(ctx.fps, 1.0))))
            self.last_t[r.key] = ctx.t
            s = self.scores.get(r.key, 0.0)
            s = s + dt if r.hit else max(0.0, s - self.p["decay_ratio"] * dt)
            self.scores[r.key] = s
            if r.hit:
                overlays.extend(r.overlays)
                if r.primary_track_id is not None:
                    suspects.add(r.primary_track_id)
            if r.hit and (r.instant or s >= float(self.p["trigger_s"])) and self._may_fire(r.key, ctx.t):
                events.append(self._make_event(ctx, r))
                self.fired[r.key] = ctx.t
        for k in list(self.scores):
            if k in seen:
                continue
            last = self.last_t.get(k, ctx.t)
            dt = min(max_dt, max(0.0, ctx.t - last))
            self.last_t[k] = ctx.t
            self.scores[k] = max(0.0, self.scores[k] - self.p["decay_ratio"] * dt)
            if self.scores[k] <= 0.0 and k not in self.fired:
                self.scores.pop(k, None)
                self.last_t.pop(k, None)
        self.live_overlays = overlays
        self.live_suspects = suspects
        return events

    def forget_tracks(self, track_ids: Iterable[int]) -> None:
        """Dọn trạng thái của các track đã kết thúc (gọi khi tracker xoá track)."""
        ids = {str(i) for i in track_ids}
        for d in (self.scores, self.last_t, self.fired, *self._track_state()):
            for k in [k for k in d if str(k).split(":")[0] in ids]:
                d.pop(k, None)

    def _track_state(self) -> tuple[dict, ...]:
        """Các dict trạng thái riêng của rule có khoá bắt đầu bằng id track ("12" hoặc "12:...") — được dọn cùng
        lúc track kết thúc, tránh phình bộ nhớ khi chạy RTSP nhiều giờ."""
        return ()

    def _may_fire(self, key: str, t: float) -> bool:
        if key not in self.fired:
            return True
        if self.p.get("once_per_key", True):
            return False
        return t - self.fired[key] >= float(self.p.get("cooldown_s", 60.0))

    def _make_event(self, ctx: FrameContext, r: RuleResult) -> ViolationEvent:
        sev = r.severity or self.p.get("severity") or self.default_severity
        ev = ViolationEvent(
            rule_code=self.code, rule_name=self.name, severity=sev, channel=ctx.channel,
            frame_idx=ctx.frame_idx, t=ctx.t, wall_time=ctx.wall_time, track_ids=list(r.track_ids),
            primary_track_id=r.primary_track_id, vehicle_cls=r.vehicle_cls,
            bbox=None if r.bbox is None else np.asarray(r.bbox, dtype=np.float64), confidence=float(r.confidence),
            variant=r.variant, measurements=dict(r.measurements), notes=list(r.notes), overlays=list(r.overlays),
            key=f"{self.code}:{r.key}",
        )
        ref = self.catalog.lookup(self.code, r.vehicle_cls, r.variant) if self.catalog else None
        if ref is not None:
            ev.legal = ref.to_dict()
        return ev


# ===================================================================== tiện ích chung
def zone_active(zone: ZoneCfg, when: datetime | None) -> bool:
    if not zone.active_hours or when is None:
        return True
    hm = when.hour * 60 + when.minute
    for span in zone.active_hours:
        try:
            a, b = span.split("-")
            ah, am = (int(x) for x in a.strip().split(":"))
            bh, bm = (int(x) for x in b.strip().split(":"))
        except ValueError:
            continue
        lo, hi = ah * 60 + am, bh * 60 + bm
        if lo <= hi:
            if lo <= hm <= hi:
                return True
        elif hm >= lo or hm <= hi:       # khoảng qua nửa đêm, vd "22:00-05:00"
            return True
    return False


def find_zone(pt: Sequence[float], zones: Sequence[ZoneCfg], when: datetime | None = None) -> ZoneCfg | None:
    for z in zones:
        if zone_active(z, when) and point_in_polygon(pt, z.polygon):
            return z
    return None


def track_box_overlay(tr: Track, color: tuple[int, int, int] = RED, text: str = "") -> Overlay:
    x1, y1, x2, y2 = tr.xyxy
    return Overlay("box", [(x1, y1), (x2, y2)], color, thickness=3, text=text)


def motion_arrow(tr: Track, seconds: float = 1.0, color: tuple[int, int, int] = RED,
                 min_len: float = 25.0) -> Overlay | None:
    pts = tr.recent(seconds)
    if len(pts) < 2:
        return None
    a, b = pts[0].ground_px, pts[-1].ground_px
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return None
    if n < min_len:
        b = (a[0] + dx / n * min_len, a[1] + dy / n * min_len)
    return Overlay("arrow", [a, b], color, thickness=3)


def union_of(tracks: Sequence[Track]) -> np.ndarray:
    return union_box([t.xyxy for t in tracks])


def world_or_px_disp(tr: Track, seconds: float) -> tuple[tuple[float, float] | None, bool]:
    """Dịch chuyển trong `seconds` giây: ưu tiên mét (nếu có hiệu chuẩn), không thì pixel."""
    pts = tr.recent(seconds)
    if len(pts) < 2:
        return None, False
    if pts[0].ground_m is not None and pts[-1].ground_m is not None:
        return (pts[-1].ground_m[0] - pts[0].ground_m[0], pts[-1].ground_m[1] - pts[0].ground_m[1]), True
    return (pts[-1].ground_px[0] - pts[0].ground_px[0], pts[-1].ground_px[1] - pts[0].ground_px[1]), False


def touches_border(xyxy: Sequence[float], width: int, height: int, margin: float = 3.0) -> bool:
    """Hộp chạm mép khung (xe mới vào/sắp ra) → tâm và điểm tiếp đất không đáng tin."""
    x1, y1, x2, y2 = xyxy
    return x1 <= margin or x2 >= width - margin or y2 >= height - margin


def on_two_wheeler(ctx: FrameContext, p: Track, min_ioa: float = 0.25) -> bool:
    """Phần thân dưới của người chồng lên hộp xe máy/xe đạp → nhiều khả năng là người ngồi trên xe."""
    x1, y1, x2, y2 = p.xyxy
    lower = (x1, y1 + 0.45 * (y2 - y1), x2, y2)
    return any(ioa(lower, c.xyxy) >= min_ioa for c in ctx.tracks if c.cls in ("motorcycle", "bicycle"))


def walking_pedestrians(ctx: FrameContext, max_walk_ms: float = 3.2, min_ioa: float = 0.25) -> list[Track]:
    """Người đi bộ "chắc chắn": không được gán lên xe, không chồng lên xe hai bánh, tốc độ ở mức đi bộ/chạy.

    Lúc track mới xuất hiện, phép gán người–xe chưa kịp chốt → nếu chỉ dùng ctx.pedestrians() sẽ coi nhầm
    người đang ngồi trên xe máy là người đi bộ (báo giả cho RB-07/HZ-04).
    """
    out = []
    for p in ctx.pedestrians():
        if on_two_wheeler(ctx, p, min_ioa):
            continue
        if p.kin.speed_kmh is not None and p.kin.speed_kmh / 3.6 > max_walk_ms:
            continue
        out.append(p)
    return out


class StationaryMemory:
    """Nhớ chỗ xe đang đứng yên để đồng hồ "đã đỗ bao lâu" KHÔNG về 0 khi tracker đổi ID.

    Xe đỗ trước cổng trường bị dòng xe đi ngang che liên tục → mất dấu > 1,5 s → tracker cấp ID mới. Nếu đồng hồ
    gắn với ID thì không bao giờ đủ 60 s. Ở đây đồng hồ gắn với VỊ TRÍ: track đứng yên mới xuất hiện đúng chỗ một
    xe vừa mất dấu (≤ reacquire_s, lệch ≤ max_shift × bề rộng hộp, cùng nhóm) sẽ kế thừa mốc thời gian và trạng
    thái "đã báo" của xe đó → không đếm lại từ đầu và không lập hồ sơ thứ hai cho cùng một xe.
    """

    def __init__(self, reacquire_s: float = 10.0, max_shift: float = 0.5):
        self.reacquire_s = reacquire_s
        self.max_shift = max_shift
        self.spots: dict[int, dict[str, Any]] = {}       # id track → {group, pos, w, since, last, fired}

    def touch(self, tr: Track, group: str, t: float) -> dict[str, Any]:
        """Gọi mỗi khung cho track đang đứng yên trong vùng `group`. Trả bản ghi (since, fired...)."""
        s = self.spots.get(tr.id)
        if s is None or s["group"] != group:
            s = {"group": group, "since": t, "fired": False}
            for oid, o in list(self.spots.items()):
                if oid == tr.id or o["group"] != group or t - o["last"] > self.reacquire_s or t - o["last"] <= 0:
                    continue
                if math.dist(o["pos"], tr.bottom) <= self.max_shift * max(o["w"], tr.w, 1.0):
                    s = {"group": group, "since": o["since"], "fired": o["fired"]}
                    del self.spots[oid]
                    break
            self.spots[tr.id] = s
        s["pos"], s["w"], s["last"] = tr.bottom, tr.w, t
        return s

    def release(self, track_id: int) -> None:
        """Xe đã chuyển bánh / ra khỏi vùng → xoá mốc."""
        self.spots.pop(track_id, None)

    def prune(self, t: float) -> None:
        for k in [k for k, s in self.spots.items() if t - s["last"] > self.reacquire_s]:
            del self.spots[k]


def is_moving(tr: Track, min_kmh: float = 4.0, min_norm: float = 0.3) -> bool:
    if tr.kin.speed_kmh is not None:
        return tr.kin.speed_kmh >= min_kmh
    return tr.kin.speed_norm >= min_norm


def default_corridor(width: int, height: int, pts_norm: Sequence[Sequence[float]] | None = None,
                     pov: Any = None, half_width_m: float = 1.6, max_z_m: float = 30.0) -> list[tuple[float, float]]:
    """Hành lang phía trước xe POV.

    pts_norm: toạ độ chuẩn hoá 0..1 nếu người dùng khai báo. Không khai báo mà có cấu hình POV (chân trời, chiều cao
    camera, góc nhìn) → dựng hành lang rộng ±half_width_m mét, dài tới max_z_m mét theo hình học mặt đường phẳng
    (ở gần thì rộng trên ảnh, ở xa thì hẹp — đúng phối cảnh). Không có gì → hình thang mặc định.
    """
    if pts_norm:
        return [(float(x) * width, float(y) * height) for x, y in pts_norm]
    if pov is not None:
        f = (width / 2.0) / math.tan(math.radians(pov.hfov_deg) / 2.0)
        hz = pov.horizon_y_ratio * height
        dv_far = f * pov.camera_height_m / max_z_m
        v_far = hz + dv_far
        if v_far < height - 10:
            hw_near = half_width_m * (height - hz) / pov.camera_height_m
            hw_far = half_width_m * dv_far / pov.camera_height_m
            cx = width / 2.0
            return [(cx - hw_near, float(height)), (cx - hw_far, v_far), (cx + hw_far, v_far), (cx + hw_near, float(height))]
    return [(0.30 * width, height), (0.46 * width, 0.48 * height), (0.54 * width, 0.48 * height), (0.70 * width, height)]
