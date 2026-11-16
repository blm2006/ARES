"""Tầng nhận thức thứ hai: chạy model chuyên biệt trên ẢNH CẮT của đối tượng đã track và ghi kết quả vào attrs.

Rule chỉ đọc attrs → test được rule mà không cần model; còn ở đây lo phần "nhìn".
  HelmetEstimator      → người: attrs["helmet_obs"]  (t, helmet|no_helmet|unknown, conf)
  PhoneEstimator       → người lái: attrs["phone_obs"] (t, tư_thế_gọi, điện_thoại_gần_tay, conf)
  CargoEstimator       → xe máy: attrs["cargo_obs"] (t, tỉ lệ bề rộng, bề rộng m|None, phương pháp), attrs["cargo_box"]
  TurnSignalTracker    → xe: attrs["blink"] {"left": t, "right": t}
  OrientationEstimator → xe (POV): attrs["orientation"] front|rear
  LaneSegEstimator     → ctx.aux["lane_masks"] {sidewalk, solid_line, crosswalk, road}
  EgoPhoneDetector     → ctx.aux["ego_phone"] (thấy?, conf, hộp)
  POVLightEstimator    → ctx.aux["pov_light"] (LightState, hộp)
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Any

import cv2
import numpy as np

from traffic_safety_ai.core.traffic_light import LightStateTracker, pick_relevant_light
from traffic_safety_ai.core.types import MOTOR_VEHICLES, Detection, FrameContext, Track
from traffic_safety_ai.utils.geometry import crop as crop_box
from traffic_safety_ai.utils.geometry import expand_box, iou, union_box


def _obs(tr: Track, key: str, maxlen: int = 90) -> deque:
    return tr.attrs.setdefault(key, deque(maxlen=maxlen))


def _moving(tr: Track) -> bool:
    if tr.kin.speed_kmh is not None:
        return tr.kin.speed_kmh >= 4.0
    return tr.kin.speed_norm >= 0.3


# ===================================================================== mũ bảo hiểm
class HelmetEstimator:
    def __init__(self, model: Any, every_n: int = 2, max_crops: int = 16, min_head_px: int = 10,
                 max_hz: float = 0.0):
        self.model = model
        self.max_hz = max_hz
        self.every_n = max(1, every_n)
        self.max_crops = max_crops
        self.min_head_px = min_head_px

    @staticmethod
    def head_region(p: Track) -> np.ndarray:
        x1, y1, x2, y2 = p.xyxy
        w, h = x2 - x1, y2 - y1
        return np.array([x1 - 0.15 * w, y1 - 0.08 * h, x2 + 0.15 * w, y1 + 0.38 * h])

    def update(self, ctx: FrameContext) -> None:
        if self.model is None:
            return
        cands = []
        for moto_id, pids in ctx.riders.items():
            moto = ctx.by_id(moto_id)
            if moto is None or not _moving(moto):
                continue
            for pid in pids:
                p = ctx.by_id(pid)
                if p is not None and (ctx.frame_idx + pid) % self.every_n == 0 and _due(p, "_helmet_t", ctx.t,
                                                                                      self.max_hz):
                    cands.append(p)
        cands = sorted(cands, key=lambda p: -p.h)[: self.max_crops]
        crops, keep = [], []
        for p in cands:
            c, _ = crop_box(ctx.frame, self.head_region(p))
            if c.shape[0] >= self.min_head_px and c.shape[1] >= self.min_head_px:
                crops.append(c)
                keep.append(p)
        if not crops:
            return
        outs = self.model.predict(crops)
        for p, o, c in zip(keep, outs, crops):
            label, conf = "unknown", 0.0
            if o.probs:
                pr = {k: v for k, v in o.probs.items() if k in ("helmet", "no_helmet")}
                if pr:
                    label = max(pr, key=pr.get)
                    conf = pr[label]
            else:
                ch = c.shape[0]
                best = None
                for d in o.boxes:
                    if d.cls not in ("helmet", "no_helmet") or d.center[1] > 0.75 * ch:
                        continue
                    if best is None or d.conf > best.conf:
                        best = d
                if best is not None:
                    label, conf = best.cls, best.conf
            _obs(p, "helmet_obs").append((ctx.t, label, conf))


# ===================================================================== điện thoại (CCTV)
KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR, KP_LSHO, KP_RSHO, KP_LWRI, KP_RWRI = 0, 1, 2, 3, 4, 5, 6, 9, 10


def call_posture(kp: np.ndarray, crop_h: float, min_conf: float = 0.35, ratio: float = 0.5) -> bool:
    """Cổ tay sát tai/mắt/mũi (đang áp điện thoại lên tai)."""
    if kp is None or len(kp) < 11:
        return False

    def ok(i: int) -> bool:
        return kp[i][2] >= min_conf

    if ok(KP_LSHO) and ok(KP_RSHO):
        s = math.dist(kp[KP_LSHO][:2], kp[KP_RSHO][:2])
    else:
        s = 0.25 * crop_h
    s = max(s, 0.12 * crop_h)
    heads = [i for i in (KP_LEAR, KP_REAR, KP_LEYE, KP_REYE, KP_NOSE) if ok(i)]
    for w in (KP_LWRI, KP_RWRI):
        if not ok(w) or not heads:
            continue
        d = min(math.dist(kp[w][:2], kp[h][:2]) for h in heads)
        if d < ratio * s:
            return True
    return False


def _due(tr: Any, key: str, t: float, max_hz: float) -> bool:
    """Giới hạn số lần chạy model ảnh cắt cho MỘT đối tượng: ≤ max_hz lần/giây (0 = không giới hạn)."""
    if max_hz <= 0:
        return True
    last = tr.attrs.get(key)
    if last is not None and t - last < 1.0 / max_hz - 1e-6:
        return False
    tr.attrs[key] = t
    return True


class PhoneEstimator:
    def __init__(self, pose_model: Any | None, phone_model: Any | None, every_n: int = 2, max_crops: int = 12,
                 phone_conf: float = 0.15, max_hz: float = 0.0):
        self.max_hz = max_hz
        self.pose = pose_model
        self.phone = phone_model
        self.every_n = max(1, every_n)
        self.max_crops = max_crops
        self.phone_conf = phone_conf

    def update(self, ctx: FrameContext) -> None:
        if self.pose is None and self.phone is None:
            return
        drivers = []
        for moto_id, pids in ctx.riders.items():
            moto = ctx.by_id(moto_id)
            if moto is None or not pids or not _moving(moto):
                continue
            d = ctx.by_id(pids[0])
            if d is not None and (ctx.frame_idx + d.id) % self.every_n == 0 and _due(d, "_phone_t", ctx.t,
                                                                                      self.max_hz):
                drivers.append(d)
        drivers = sorted(drivers, key=lambda p: -p.h)[: self.max_crops]
        crops, keep = [], []
        for p in drivers:
            c, _ = crop_box(ctx.frame, p.xyxy, pad=0.15)
            if c.shape[0] < 32:
                continue
            s = 256 / c.shape[0] if c.shape[0] < 256 else 1.0
            crops.append(cv2.resize(c, None, fx=s, fy=s) if s != 1.0 else c)
            keep.append(p)
        if not crops:
            return
        kps: list[np.ndarray | None] = [None] * len(crops)
        if self.pose is not None:
            for i, o in enumerate(self.pose.predict(crops)):
                if o.keypoints is not None and len(o.keypoints):
                    j = int(np.argmax([b.area for b in o.boxes])) if o.boxes else 0
                    kps[i] = o.keypoints[j]
        phones: list[list[Detection]] = [[] for _ in crops]
        if self.phone is not None:
            uppers = [c[: max(1, int(c.shape[0] * 0.7))] for c in crops]
            for i, o in enumerate(self.phone.predict(uppers, conf=self.phone_conf, classes=["phone"])):
                phones[i] = [d for d in o.boxes if d.cls == "phone"]
        for p, c, kp, ph in zip(keep, crops, kps, phones):
            call = call_posture(kp, c.shape[0]) if kp is not None else False
            near = False
            conf = 0.0
            for d in ph:
                conf = max(conf, d.conf)
                if kp is not None:
                    for w in (KP_LWRI, KP_RWRI):
                        if kp[w][2] >= 0.3 and math.dist(d.center, kp[w][:2]) < 0.35 * c.shape[0]:
                            near = True
                else:
                    near = True
            _obs(p, "phone_obs").append((ctx.t, call, near, max(conf, 0.6 if call else 0.0)))


# ===================================================================== hàng cồng kềnh
class CargoEstimator:
    """Model hàng hoá (nếu có) hoặc tách nền MOG2 trên camera tĩnh để đo bề rộng xe + hàng."""

    def __init__(self, model: Any | None = None, use_bgsub: bool = True, scale: float = 0.5, every_n: int = 2):
        self.model = model
        self.use_bgsub = use_bgsub and model is None
        self.scale = scale
        self.every_n = max(1, every_n)
        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=24, detectShadows=True) \
            if self.use_bgsub else None
        self.fg: np.ndarray | None = None

    def update(self, ctx: FrameContext) -> None:
        if self.bg is not None:
            small = cv2.resize(ctx.frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            self.fg = self.bg.apply(small)
        if ctx.frame_idx % self.every_n:
            return
        motos = [m for m in ctx.of_class("motorcycle") if _moving(m)]
        if not motos:
            return
        if self.model is not None:
            self._model(ctx, motos)
        elif self.fg is not None and ctx.frame_idx > 60:      # chờ nền ổn định
            self._bgsub(ctx, motos)

    def _width_m(self, ctx: FrameContext, x1: float, x2: float, y: float) -> float | None:
        if ctx.calib is None:
            return None
        return ctx.calib.width_m_at(x1, x2, y)

    def _ref_width(self, ctx: FrameContext, m: Track) -> float:
        riders = [ctx.by_id(i) for i in ctx.riders.get(m.id, [])]
        return max([m.w] + [r.w for r in riders if r is not None])

    def _model(self, ctx: FrameContext, motos: list[Track]) -> None:
        regions = [expand_box(m.xyxy, 1.0, 1.2, 1.0, 0.1) for m in motos]
        crops = [crop_box(ctx.frame, r) for r in regions]
        outs = self.model.predict([c for c, _ in crops])
        for m, (c, off), o in zip(motos, crops, outs):
            boxes = [d.xyxy + np.array([off[0], off[1], off[0], off[1]]) for d in o.boxes if d.cls == "cargo"]
            boxes = [b for b in boxes if iou(b, expand_box(m.xyxy, 0.6, 1.0, 0.6, 0.1)) > 0.02]
            if not boxes:
                continue
            env = union_box([m.xyxy, *boxes])
            ratio = (env[2] - env[0]) / max(self._ref_width(ctx, m), 1.0)
            _obs(m, "cargo_obs").append((ctx.t, float(ratio), self._width_m(ctx, env[0], env[2], m.xyxy[3]), "model"))
            m.attrs["cargo_box"] = env

    def _bgsub(self, ctx: FrameContext, motos: list[Track]) -> None:
        fg = self.fg
        s = self.scale
        H, W = fg.shape[:2]
        mask_all = np.where(fg == 255, 255, 0).astype(np.uint8)          # bỏ bóng (127)
        for m in motos:
            own = {m.id, *ctx.riders.get(m.id, [])}
            riders = [ctx.by_id(i) for i in ctx.riders.get(m.id, [])]
            top = min([m.xyxy[1]] + [r.xyxy[1] for r in riders if r is not None])
            reg = np.array([m.xyxy[0] - 1.0 * m.w, top, m.xyxy[2] + 1.0 * m.w, m.xyxy[3]]) * s
            x1, y1, x2, y2 = (int(v) for v in (max(0, reg[0]), max(0, reg[1]), min(W, reg[2]), min(H, reg[3])))
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            sub = mask_all[y1:y2, x1:x2].copy()
            for t in ctx.tracks:
                if t.id in own:
                    continue
                bx = (t.xyxy * s).astype(int)
                sub[max(0, bx[1] - y1):max(0, bx[3] - y1), max(0, bx[0] - x1):max(0, bx[2] - x1)] = 0
            sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(sub)
            cx = int(m.center[0] * s) - x1
            cy = int((m.xyxy[1] * 0.4 + m.xyxy[3] * 0.6) * s) - y1
            if not (0 <= cx < sub.shape[1] and 0 <= cy < sub.shape[0]) or n <= 1:
                continue
            k = lab[cy, cx]
            if k == 0:
                continue
            bx, by, bw, bh = stats[k][:4]
            env = np.array([bx + x1, by + y1, bx + x1 + bw, by + y1 + bh], float) / s
            ratio = (env[2] - env[0]) / max(self._ref_width(ctx, m), 1.0)
            _obs(m, "cargo_obs").append((ctx.t, float(ratio), self._width_m(ctx, env[0], env[2], m.xyxy[3]), "bgsub"))
            m.attrs["cargo_box"] = env


# ===================================================================== xi-nhan
def amber_fraction(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (H >= 8) & (H <= 30) & (S >= 90) & (V >= 170)
    return float(m.mean())


def detect_blink(series: list[tuple[float, float]], min_amp: float = 0.02, f_lo: float = 0.75,
                 f_hi: float = 2.6) -> bool:
    """Chuỗi (t, cường độ hổ phách) có dao động tuần hoàn 0.75–2.6 Hz (xi-nhan: 60–120 lần/phút) không."""
    if len(series) < 8:
        return False
    v = np.array([x[1] for x in series])
    t = np.array([x[0] for x in series])
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < min_amp:
        return False
    th_hi, th_lo = lo + 0.6 * (hi - lo), lo + 0.4 * (hi - lo)
    state = v[0] > th_hi
    rises = []
    for ti, vi in zip(t, v):
        if not state and vi > th_hi:
            state = True
            rises.append(ti)
        elif state and vi < th_lo:
            state = False
    if len(rises) < 2:
        return False
    periods = np.diff(rises)
    f = 1.0 / float(np.median(periods))
    return f_lo <= f <= f_hi and float(np.std(periods)) < 0.35 * float(np.median(periods))


class TurnSignalTracker:
    def __init__(self, window_s: float = 2.5, min_h: float = 16.0):
        self.window_s = window_s
        self.min_h = min_h

    def update(self, ctx: FrameContext) -> None:
        for tr in ctx.vehicles(MOTOR_VEHICLES):
            if tr.h < self.min_h:
                continue
            x1, y1, x2, y2 = tr.xyxy
            w, h = x2 - x1, y2 - y1
            band = (y1 + 0.3 * h, y1 + 0.85 * h)
            left, _ = crop_box(ctx.frame, [x1, band[0], x1 + 0.3 * w, band[1]])
            right, _ = crop_box(ctx.frame, [x2 - 0.3 * w, band[0], x2, band[1]])
            ser = tr.attrs.setdefault("amber_series", deque(maxlen=180))
            ser.append((ctx.t, amber_fraction(left), amber_fraction(right)))
            recent = [s for s in ser if s[0] >= ctx.t - self.window_s]
            blink = tr.attrs.setdefault("blink", {})
            for i, side in ((1, "left"), (2, "right")):
                if detect_blink([(s[0], s[i]) for s in recent]):
                    blink[side] = ctx.t


# ===================================================================== POV
class OrientationEstimator:
    def __init__(self, model: Any, every_n: int = 3):
        self.model = model
        self.every_n = max(1, every_n)

    def update(self, ctx: FrameContext) -> None:
        if self.model is None:
            return
        cands = [t for t in ctx.vehicles(MOTOR_VEHICLES) if (ctx.frame_idx + t.id) % self.every_n == 0 and t.h > 32]
        crops = [crop_box(ctx.frame, t.xyxy, pad=0.05)[0] for t in cands]
        if not crops:
            return
        for t, o in zip(cands, self.model.predict(crops)):
            label = None
            if o.probs:
                label = max(o.probs, key=o.probs.get)
            elif o.boxes:
                label = max(o.boxes, key=lambda d: d.conf).cls
            if label in ("front", "rear"):
                votes = t.attrs.setdefault("_orient_votes", Counter())
                votes[label] += 1
                t.attrs["orientation"] = votes.most_common(1)[0][0]


class LaneSegEstimator:
    CLASSES = ("sidewalk", "solid_line", "crosswalk", "road")

    def __init__(self, model: Any, every_n: int = 2):
        self.model = model
        self.every_n = max(1, every_n)
        self.last: dict[str, np.ndarray] = {}

    def update(self, ctx: FrameContext) -> None:
        if self.model is None:
            return
        if ctx.frame_idx % self.every_n == 0 or not self.last:
            o = self.model.predict([ctx.frame])[0]
            H, W = ctx.frame.shape[:2]
            masks = {c: np.zeros((H, W), np.uint8) for c in self.CLASSES}
            for d, poly in zip(o.boxes, o.masks_xy or []):
                if d.cls in masks and poly is not None and len(poly) >= 3:
                    cv2.fillPoly(masks[d.cls], [poly.astype(np.int32)], 1)
            self.last = {k: v.astype(bool) for k, v in masks.items()}
        ctx.aux["lane_masks"] = self.last


class EgoPhoneDetector:
    def __init__(self, model: Any, every_n: int = 2, mount_roi: tuple[float, float, float, float] | None = None,
                 conf: float = 0.25):
        self.model = model
        self.every_n = max(1, every_n)
        self.mount = mount_roi
        self.conf = conf
        self._last: tuple[bool, float, Any] = (False, 0.0, None)

    def update(self, ctx: FrameContext) -> None:
        if self.model is None:
            return
        if ctx.frame_idx % self.every_n == 0:
            H, W = ctx.frame.shape[:2]
            y0 = int(H * 0.5)
            o = self.model.predict([ctx.frame[y0:]], conf=self.conf, classes=["phone"])[0]
            best = None
            for d in o.boxes:
                b = d.xyxy + np.array([0, y0, 0, y0])
                if self.mount is not None:
                    mx1, my1, mx2, my2 = self.mount
                    mb = np.array([mx1 * W, my1 * H, mx2 * W, my2 * H])
                    if iou(b, mb) > 0.2:
                        continue
                if best is None or d.conf > best[1]:
                    best = (True, d.conf, b)
            self._last = best or (False, 0.0, None)
        ctx.aux["ego_phone"] = self._last


class POVLightEstimator:
    def __init__(self) -> None:
        self.tracker = LightStateTracker("POV", roi=None, min_consecutive=2, hold_s=0.7)
        self._last_box: np.ndarray | None = None

    def update(self, ctx: FrameContext, light_dets: list[Detection]) -> None:
        W, H = ctx.frame_size
        boxes = [d.xyxy for d in light_dets]
        i = pick_relevant_light(boxes, W, H)
        if i is None:
            ctx.aux["pov_light"] = None
            return
        crop, _ = crop_box(ctx.frame, boxes[i])
        st = self.tracker.update_from_crop(crop, ctx.t)
        self._last_box = boxes[i]
        ctx.aux["pov_light"] = (st, boxes[i])
        ctx.lights = {"POV": st}
