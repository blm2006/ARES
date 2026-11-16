"""Ước lượng vận tốc / gia tốc / tốc độ nở hộp cho mỗi track.

Khác spec gốc (lấy hiệu 2 điểm rồi EMA): sai phân 2 frame khuếch đại rung của bbox lên rất nhiều lần
(bbox rung 2 px ở 30 FPS = 60 px/s "ảo"). Ở đây dùng HỒI QUY TUYẾN TÍNH có loại điểm ngoại lai trên
một cửa sổ thời gian (mặc định 1 s), kèm độ lệch chuẩn của ước lượng → biết lúc nào số đo đáng tin.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from traffic_safety_ai.core.types import Kinematics, Track


@dataclass
class KinematicsConfig:
    window_s: float = 1.0
    min_span_s: float = 0.3
    min_points: int = 3
    moving_speed_norm: float = 0.25      # (chiều cao hộp)/giây — dưới mức này coi như đứng yên
    moving_speed_kmh: float = 3.0
    accel_window_s: float = 0.8
    scale_window_s: float = 0.6
    mature_frac: float = 0.7
    scale: bool = True                   # tốc độ nở hộp — chỉ luật POV dùng; CCTV tắt để đỡ tốn


def robust_fit(ts: np.ndarray, vals: np.ndarray) -> tuple[float, float, float]:
    """Hồi quy vals = a + b·t, bỏ điểm lệch > 3·MAD rồi khớp lại. Trả (b, sigma_b, residual_std)."""
    t = np.asarray(ts, dtype=np.float64)
    v = np.asarray(vals, dtype=np.float64)
    if len(t) < 2:
        return 0.0, float("inf"), 0.0
    for _ in range(2):
        tc = t - t.mean()
        denom = float((tc ** 2).sum())
        if denom <= 1e-12:
            return 0.0, float("inf"), 0.0
        b = float((tc * (v - v.mean())).sum() / denom)
        a = v.mean() - b * t.mean()
        res = v - (a + b * t)
        mad = float(np.median(np.abs(res - np.median(res))))
        keep = np.abs(res) <= max(3.0 * 1.4826 * mad, 1e-9)
        if keep.all() or keep.sum() < 3:
            break
        t, v = t[keep], v[keep]
    tc = t - t.mean()
    denom = float((tc ** 2).sum())
    b = float((tc * (v - v.mean())).sum() / denom) if denom > 1e-12 else 0.0
    a = v.mean() - b * t.mean()
    res = v - (a + b * t)
    dof = max(1, len(t) - 2)
    rstd = float(math.sqrt(float((res ** 2).sum()) / dof))
    sigma_b = rstd / math.sqrt(denom) if denom > 1e-12 else float("inf")
    return b, sigma_b, rstd


def robust_fit_xy(ts: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[tuple[float, float, float],
                                                                         tuple[float, float, float]]:
    """robust_fit cho hai chuỗi cùng mốc thời gian. Đường nhanh: khớp cả hai cột một lần; chỉ cột nào có điểm
    ngoại lai mới khớp lại riêng (phần lớn quỹ đạo không có ngoại lai → bớt ~một nửa phép tính mỗi khung)."""
    t = np.asarray(ts, dtype=np.float64)
    if len(t) < 3:
        return robust_fit(t, xs), robust_fit(t, ys)
    V = np.stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)], axis=1)
    tc = t - t.mean()
    denom = float((tc ** 2).sum())
    if denom <= 1e-12:
        return (0.0, float("inf"), 0.0), (0.0, float("inf"), 0.0)
    vm = V.mean(axis=0)
    b = (tc[:, None] * (V - vm)).sum(axis=0) / denom
    res = V - (vm + tc[:, None] * b)
    med = np.median(res, axis=0)
    mad = np.median(np.abs(res - med), axis=0)
    out = []
    for j in range(2):
        if np.all(np.abs(res[:, j]) <= max(3.0 * 1.4826 * float(mad[j]), 1e-9)):
            dof = max(1, len(t) - 2)
            rstd = float(np.sqrt(float((res[:, j] ** 2).sum()) / dof))
            out.append((float(b[j]), rstd / np.sqrt(denom), rstd))
        else:
            out.append(robust_fit(t, V[:, j]))
    return out[0], out[1]


class KinematicsEstimator:
    def __init__(self, cfg: KinematicsConfig | None = None):
        self.cfg = cfg or KinematicsConfig()

    def update(self, tr: Track) -> Kinematics:
        cfg = self.cfg
        k = Kinematics()
        pts = tr.recent(cfg.window_s)
        k.n = len(pts)
        if len(pts) < cfg.min_points:
            tr.kin = k
            return k
        ts = np.array([p.t for p in pts])
        span = float(ts[-1] - ts[0])
        k.window_s = span
        if span < cfg.min_span_s:
            tr.kin = k
            return k
        gx = np.array([p.ground_px[0] for p in pts])
        gy = np.array([p.ground_px[1] for p in pts])
        (vx, _, _), (vy, _, _) = robust_fit_xy(ts, gx, gy)
        k.vx_px, k.vy_px = vx, vy
        k.speed_px = math.hypot(vx, vy)
        hs = np.array([max(1.0, p.xyxy[3] - p.xyxy[1]) for p in pts])
        k.speed_norm = k.speed_px / float(np.median(hs))

        # tốc độ nở hộp (TTC trên POV): d ln h / dt
        sp = [p for p in pts if p.t >= ts[-1] - cfg.scale_window_s] if cfg.scale else []
        if len(sp) >= cfg.min_points:
            st = np.array([p.t for p in sp])
            if st[-1] - st[0] >= cfg.min_span_s * 0.5:
                lh = np.log(np.array([max(1.0, p.xyxy[3] - p.xyxy[1]) for p in sp]))
                k.scale_rate, _, _ = robust_fit(st, lh)

        # mét (khi có hiệu chuẩn)
        wm = [p for p in pts if p.ground_m is not None]
        if len(wm) >= cfg.min_points:
            wt = np.array([p.t for p in wm])
            if wt[-1] - wt[0] >= cfg.min_span_s:
                wx = np.array([p.ground_m[0] for p in wm])
                wy = np.array([p.ground_m[1] for p in wm])
                (vxm, sx, _), (vym, sy, _) = robust_fit_xy(wt, wx, wy)
                k.vx_m, k.vy_m = vxm, vym
                v = math.hypot(vxm, vym)
                k.speed_kmh = v * 3.6
                k.speed_sigma_kmh = math.hypot(sx, sy) * 3.6 if math.isfinite(sx) and math.isfinite(sy) else None
                # chỉ lưu tốc độ "chín" (cửa sổ đủ dài) để tính gia tốc — tránh gia tốc ảo lúc track mới sinh
                if wt[-1] - wt[0] >= cfg.mature_frac * cfg.window_s:
                    hist: deque = tr.attrs.setdefault("_speed_hist", deque(maxlen=120))
                    hist.append((float(wt[-1]), v))
                    ah = [(t, s) for t, s in hist if t >= wt[-1] - cfg.accel_window_s]
                    if len(ah) >= 4 and ah[-1][0] - ah[0][0] >= cfg.accel_window_s * 0.6:
                        a, sa, _ = robust_fit(np.array([x[0] for x in ah]), np.array([x[1] for x in ah]))
                        k.accel_ms2 = a
                        k.accel_sigma_ms2 = sa if math.isfinite(sa) else None

        if k.speed_kmh is not None:
            k.moving = k.speed_kmh >= cfg.moving_speed_kmh and k.speed_norm >= cfg.moving_speed_norm * 0.5
        else:
            k.moving = k.speed_norm >= cfg.moving_speed_norm
        tr.kin = k
        return k


def displacement(tr: Track, seconds: float, world: bool = False) -> tuple[float, float] | None:
    """Véc-tơ dịch chuyển của điểm tiếp đất trong `seconds` giây gần nhất (pixel hoặc mét)."""
    pts = tr.recent(seconds)
    if len(pts) < 2:
        return None
    if world:
        if pts[0].ground_m is None or pts[-1].ground_m is None:
            return None
        return (pts[-1].ground_m[0] - pts[0].ground_m[0], pts[-1].ground_m[1] - pts[0].ground_m[1])
    return (pts[-1].ground_px[0] - pts[0].ground_px[0], pts[-1].ground_px[1] - pts[0].ground_px[1])
