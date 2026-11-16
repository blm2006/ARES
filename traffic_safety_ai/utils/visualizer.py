"""Vẽ khung hiển thị theo chuẩn màu của dự án.

  * Hộp vi phạm: viền đỏ #E53935.
  * Vùng nguy cơ (va chạm / khuất tầm nhìn): đa giác đỏ mờ (alpha 0.35) + biểu tượng tam giác cảnh báo.
  * Mũi tên đỏ: hướng di chuyển vi phạm / lao ra bất ngờ.  Mũi tên xanh dương: hướng đi đúng quy định.
  * Badge góc trên: mã lỗi (vd "RB-01 Không đội mũ bảo hiểm") + kênh ("CCTV view" / "POV view").
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from traffic_safety_ai.core.config import SiteConfig
from traffic_safety_ai.core.traffic_light import STATE_BGR, STATE_VI
from traffic_safety_ai.core.types import CLASS_LABEL_VI, FrameContext, Overlay, ViolationEvent
from traffic_safety_ai.utils.text import TextRenderer, default_renderer

RED = (53, 57, 229)
BLUE = (229, 136, 30)
WHITE = (255, 255, 255)
DARK = (28, 28, 28)
CLASS_COLOR = {"person": (230, 200, 60), "motorcycle": (90, 220, 90), "bicycle": (200, 220, 90),
               "car": (230, 170, 60), "bus": (200, 120, 240), "truck": (160, 120, 230)}


def _pts(p: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray([[int(round(x)), int(round(y))] for x, y in p], dtype=np.int32)


def fill_poly(img: np.ndarray, pts: Sequence[Sequence[float]], color: tuple[int, int, int], alpha: float) -> None:
    if len(pts) < 3:
        return
    poly = _pts(pts)
    x, y, w, h = cv2.boundingRect(poly)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    cv2.fillPoly(over, [poly - np.array([x0, y0])], color)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)


def warning_triangle(img: np.ndarray, center: tuple[float, float], size: int = 34,
                     text: TextRenderer | None = None) -> None:
    cx, cy = int(center[0]), int(center[1])
    h = int(size * math.sqrt(3) / 2)
    tri = np.array([[cx, cy - h // 2 - 4], [cx - size // 2, cy + h // 2], [cx + size // 2, cy + h // 2]], np.int32)
    cv2.fillPoly(img, [tri], (0, 215, 255))
    cv2.polylines(img, [tri], True, DARK, 2, cv2.LINE_AA)
    (text or default_renderer()).put(img, "!", (cx, cy - h // 2 + 2), size=int(size * 0.6), color=DARK,
                                     bold=True, anchor="ct", pad=0)


class Visualizer:
    """labels = "gon" (mặc định): xe bình thường chỉ có khung mảnh, KHÔNG chữ — chỉ xe đang bị theo dõi/vi phạm mới có nhãn.
    labels = "day_du": mọi xe đều có nhãn "mã · loại · tốc độ · biển" (dùng khi cần soi lỗi nhận diện)."""

    def __init__(self, text: TextRenderer | None = None, labels: str = "gon"):
        self.text = text or default_renderer()
        self.labels = labels

    # ------------------------------------------------------------------ cảnh
    def draw_scene(self, img: np.ndarray, scene: SiteConfig, light_states: dict | None = None) -> None:
        z = scene.zones
        for sw in z.sidewalks:
            fill_poly(img, sw.polygon, (150, 150, 150), 0.18)
        for cw in z.crosswalks:
            fill_poly(img, cw.polygon, (255, 255, 255), 0.12)
            cv2.polylines(img, [_pts(cw.polygon)], True, (235, 235, 235), 1, cv2.LINE_AA)
        for np_ in z.no_parking:
            cv2.polylines(img, [_pts(np_.polygon)], True, (60, 60, 200), 1, cv2.LINE_AA)
        for ln in z.lanes:
            if ln.direction:
                a, b = ln.direction
                cv2.arrowedLine(img, tuple(int(v) for v in a), tuple(int(v) for v in b), BLUE, 3, cv2.LINE_AA,
                                tipLength=0.06)
        for sl in scene.lines.stop_lines:
            cv2.line(img, tuple(int(v) for v in sl.points[0]), tuple(int(v) for v in sl.points[1]), (240, 240, 240), 3)
            if sl.confirm_points:
                cv2.line(img, tuple(int(v) for v in sl.confirm_points[0]), tuple(int(v) for v in sl.confirm_points[1]),
                         (0, 140, 255), 1)
        for so in scene.lines.solid_lines:
            cv2.polylines(img, [_pts(so.points)], False, (0, 220, 255), 1, cv2.LINE_AA)
        for c in scene.lines.counting:
            cv2.line(img, tuple(int(v) for v in c.points[0]), tuple(int(v) for v in c.points[1]), (255, 200, 0), 1)
        for lt in scene.traffic_lights:
            st = (light_states or {}).get(lt.id)
            col = STATE_BGR.get(st.state if st else "unknown", (160, 160, 160))
            x1, y1, x2, y2 = (int(v) for v in lt.roi)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

    # ------------------------------------------------------------------ overlay
    def draw_overlays(self, img: np.ndarray, overlays: Sequence[Overlay]) -> None:
        for o in overlays:
            if o.kind == "polygon":
                if o.alpha > 0:
                    fill_poly(img, o.pts, o.color, o.alpha)
                cv2.polylines(img, [_pts(o.pts)], True, o.color, max(1, o.thickness), cv2.LINE_AA)
                if o.text == "!":
                    c = np.mean(np.asarray(o.pts, float), axis=0)
                    warning_triangle(img, (c[0], c[1]), text=self.text)
            elif o.kind == "box":
                (x1, y1), (x2, y2) = o.pts[0], o.pts[1]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), o.color, max(2, o.thickness))
                if o.text:
                    self.text.put(img, o.text, (x1, y1), size=18, color=WHITE, bg=o.color, bold=True, anchor="lb", pad=3)
            elif o.kind == "arrow":
                a, b = o.pts[0], o.pts[1]
                cv2.arrowedLine(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), o.color, max(2, o.thickness),
                                cv2.LINE_AA, tipLength=0.25)
            elif o.kind == "line":
                cv2.polylines(img, [_pts(o.pts)], False, o.color, max(2, o.thickness), cv2.LINE_AA)
                if o.text:
                    self.text.put(img, o.text, o.pts[0], size=16, color=WHITE, bg=o.color, anchor="lb", pad=2)
            elif o.kind == "point":
                cv2.circle(img, (int(o.pts[0][0]), int(o.pts[0][1])), 6, o.color, -1)
            elif o.kind == "text":
                self.text.put(img, o.text, o.pts[0], size=18, color=WHITE, bg=o.color)

    # ------------------------------------------------------------------ khung
    def draw_frame(self, frame: np.ndarray, ctx: FrameContext, overlays: Sequence[Overlay],
                   recent_events: Sequence[ViolationEvent] = (), fps: float | None = None,
                   suspects: set[int] | None = None, plates: dict[int, str] | None = None,
                   draw_scene: bool = True) -> np.ndarray:
        img = frame.copy()
        if draw_scene:
            self.draw_scene(img, ctx.scene, ctx.lights)
        suspects = suspects or set()
        plates = plates or {}
        k = min(1.0, max(0.6, img.shape[0] / 720.0))          # video nhỏ → chữ nhỏ theo, khỏi che hình
        for tr in ctx.tracks:
            x1, y1, x2, y2 = (int(v) for v in tr.xyxy)
            hot = tr.id in suspects
            col = RED if hot else CLASS_COLOR.get(tr.cls, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if hot else 1, cv2.LINE_AA)
            if self.labels != "day_du" and not hot:
                continue
            if self.labels == "day_du":
                label = f"{tr.id} {CLASS_LABEL_VI.get(tr.cls, tr.cls)}"
                if tr.kin.speed_kmh is not None and tr.cls != "person":
                    label += f" {tr.kin.speed_kmh:.0f}km/h"
                if tr.id in plates:
                    label += f" | {plates[tr.id]}"
            else:
                label = plates.get(tr.id) or CLASS_LABEL_VI.get(tr.cls, tr.cls)
            self.text.put(img, label, (x1, y1), size=int(round(15 * k)), color=WHITE, bg=col if hot else (50, 50, 50),
                          anchor="lb", pad=2)
        self.draw_overlays(img, overlays)
        self.draw_badges(img, ctx, recent_events, fps)
        return img

    def draw_badges(self, img: np.ndarray, ctx: FrameContext, recent_events: Sequence[ViolationEvent],
                    fps: float | None) -> None:
        W = img.shape[1]
        k = min(1.0, max(0.55, img.shape[0] / 720.0))
        big, small, pad = int(round(20 * k)), int(round(17 * k)), int(round(8 * k))
        x = pad
        if self.labels == "day_du":
            chan = "CCTV view" if ctx.channel == "cctv" else "POV view"
            x = self.text.put(img, chan, (x, pad), size=big, color=WHITE, bg=(70, 70, 70), bold=True)[2] + pad
        many = len(ctx.lights) > 1
        for lid, st in ctx.lights.items():
            name = f"Đèn {lid}" if (many or self.labels == "day_du") else "Đèn"
            x = self.text.put(img, f"{name}: {STATE_VI.get(st.state, st.state)}", (x, pad), size=big,
                              color=WHITE, bg=STATE_BGR.get(st.state, (120, 120, 120)), bold=True)[2] + pad
        info = ctx.wall_time.strftime("%H:%M:%S %d/%m/%Y")
        if fps and self.labels == "day_du":
            info += f"  |  {fps:.1f} FPS"
        self.text.put(img, info, (W - pad, pad), size=small, color=WHITE, bg=(40, 40, 40), anchor="rt")
        y = pad + int(big * 1.9)
        for ev in list(recent_events)[-4:]:
            col = RED if ev.severity == "vi_pham" else ((0, 140, 255) if ev.severity == "nghi_van" else (0, 180, 230))
            txt = f"{ev.rule_code} {ev.rule_name}" if self.labels == "day_du" else ev.rule_name
            self.text.put(img, txt, (pad, y), size=big, color=WHITE, bg=col, bold=True)
            y += int(big * 1.7)
