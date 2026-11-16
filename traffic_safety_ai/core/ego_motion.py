"""Khử chuyển động của chính camera (POV): optical flow thưa + RANSAC, cộng hình học mặt đường phẳng.

Trả về:
  * warp (affine 2x3, frame trước → frame này) để tracker bù chuyển động (ý tưởng GMC của BoT-SORT);
  * hệ số giãn nền k: với mặt đường phẳng, một vật ĐỨNG YÊN ở hàng ảnh y có tốc độ nở
        d(ln h)/dt = k · (y − y_horizon)
    → vật nào nở nhanh hơn mức này là đang lao về phía camera (dùng cho RB-03 POV, HZ-03);
  * tốc độ xe POV ước lượng v = k · f · H_camera (cần chiều cao gắn camera và góc nhìn ngang);
  * hình học mặt đường: điểm tiếp đất (u, v) → (X ngang, Z dọc) mét so với camera
        Z = f·H_cam / (v − y_horizon),  X = (u − W/2) · H_cam / (v − y_horizon)
    → vận tốc NGANG thật của xe phía trước không bị lẫn với hiện tượng "nở" do camera tiến tới;
  * tốc độ quay ngang (yaw) của camera, đo bằng dịch chuyển nền tại điểm tụ, để trừ khi tính vận tốc ngang.

Ước lượng từng khung khá nhiễu (vân mặt đường, rung máy) → k và yaw được lọc trung vị trong ~1 giây.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EgoMotion:
    warp: np.ndarray | None
    n_points: int = 0
    inliers: int = 0
    median_flow_px: float = 0.0
    k_ground: float | None = None
    ego_speed_mps: float | None = None
    moving: bool = False
    horizon_y: float = 0.0
    f_px: float = 0.0              # tiêu cự (pixel); 0 = chưa biết → rule tự tính từ hfov
    cam_h: float = 1.2
    frame_w: int = 0
    yaw_px_s: float = 0.0          # dịch ngang của nền tại điểm tụ (px/s) do camera quay

    def static_scale_rate(self, y_bottom: float) -> float:
        if self.k_ground is None:
            return 0.0
        return max(0.0, self.k_ground * (y_bottom - self.horizon_y))

    def shift_at(self, x: float, y: float) -> tuple[float, float]:
        """Dịch chuyển (px/frame) mà một điểm tĩnh tại (x, y) có do camera chuyển động."""
        if self.warp is None:
            return 0.0, 0.0
        p = self.warp @ np.array([x, y, 1.0])
        return float(p[0] - x), float(p[1] - y)

    def ground_xz(self, u: float, v: float, min_dv: float = 3.0) -> tuple[float, float] | None:
        """Điểm tiếp đất trên ảnh → (X, Z) mét (X>0 bên phải, Z phía trước). None nếu ở trên/sát chân trời."""
        dv = v - self.horizon_y
        if dv < min_dv or self.f_px <= 0 or self.frame_w <= 0:
            return None
        return (u - self.frame_w / 2.0) * self.cam_h / dv, self.f_px * self.cam_h / dv


def focal_px(width: int, hfov_deg: float) -> float:
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


class EgoMotionEstimator:
    def __init__(self, horizon_y_ratio: float = 0.45, camera_height_m: float = 1.2, hfov_deg: float = 110.0,
                 downscale: float = 0.5, max_corners: int = 500, moving_speed_mps: float = 1.5,
                 smooth_s: float = 1.0, fb_max_px: float = 1.0):
        self.horizon_y_ratio = horizon_y_ratio
        self.camera_height_m = camera_height_m
        self.hfov_deg = hfov_deg
        self.ds = downscale
        self.max_corners = max_corners
        self.moving_speed_mps = moving_speed_mps
        self.smooth_s = smooth_s
        self.fb_max = fb_max_px
        self._prev: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None
        self._prev_t: float | None = None
        self._k_hist: deque[tuple[float, float]] = deque(maxlen=200)
        self._yaw_hist: deque[tuple[float, float]] = deque(maxlen=200)

    def reset(self) -> None:
        self._prev = None
        self._prev_mask = None
        self._prev_t = None
        self._k_hist.clear()
        self._yaw_hist.clear()

    def _median_recent(self, hist: deque, t: float) -> float | None:
        vals = [v for tt, v in hist if tt >= t - self.smooth_s - 1e-9]
        return float(np.median(vals)) if vals else None

    def update(self, frame: np.ndarray, t: float, exclude_boxes: Sequence[Sequence[float]] = ()) -> EgoMotion:
        H, W = frame.shape[:2]
        horizon = self.horizon_y_ratio * H
        f = focal_px(W, self.hfov_deg)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, None, fx=self.ds, fy=self.ds, interpolation=cv2.INTER_AREA) if self.ds != 1.0 else gray
        mask = np.full(small.shape, 255, np.uint8)
        for b in exclude_boxes:          # bỏ điểm nằm trên xe/người (chúng tự chuyển động)
            x1, y1, x2, y2 = (int(round(v * self.ds)) for v in b)
            mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0
        prev, prev_mask, prev_t = self._prev, self._prev_mask, self._prev_t
        self._prev, self._prev_mask, self._prev_t = small, mask, t
        out = EgoMotion(None, horizon_y=horizon, f_px=f, cam_h=self.camera_height_m, frame_w=W)
        if prev is None or prev.shape != small.shape:
            return out
        dt = max(1e-3, t - (prev_t if prev_t is not None else t - 1 / 30))
        p0 = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 7, mask=prev_mask)
        if p0 is None or len(p0) < 8:
            return self._finish(out, t)
        lk = dict(winSize=(21, 21), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st, _err = cv2.calcOpticalFlowPyrLK(prev, small, p0, None, **lk)
        if p1 is None:
            return self._finish(out, t)
        # kiểm tra tiến–lùi: điểm bám sai (vân lặp, mép xe) sẽ không quay về chỗ cũ
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(small, prev, p1, None, **lk)
        fb = np.linalg.norm(p0.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1) if p0r is not None else np.zeros(len(p0))
        ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1 if st2 is not None else True) & (fb <= self.fb_max)
        a = p0.reshape(-1, 2)[ok] / self.ds
        b = p1.reshape(-1, 2)[ok] / self.ds
        out.n_points = int(len(a))
        if len(a) < 8:
            return self._finish(out, t)
        A, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0 / self.ds)
        if A is not None:
            out.warp = A.astype(np.float64)
            out.inliers = int(inl.sum()) if inl is not None else 0
            yaw = float((A @ np.array([W / 2.0, horizon, 1.0]))[0] - W / 2.0) / dt
            self._yaw_hist.append((t, yaw))
        flow = b - a
        out.median_flow_px = float(np.median(np.linalg.norm(flow, axis=1)))

        # điểm trên mặt đường ở dải giữa (dưới chân trời, chưa quá gần đáy khung) → hệ số giãn nền
        rel = a[:, 1] - horizon
        sel = (rel > 0.06 * H) & (rel < 0.42 * H)
        if sel.sum() >= 6:
            vy = flow[sel, 1] / dt
            k_i = vy / (rel[sel] ** 2)
            k = float(np.median(k_i))
            if k > 0:
                self._k_hist.append((t, k))
        return self._finish(out, t)

    def _finish(self, out: EgoMotion, t: float) -> EgoMotion:
        k = self._median_recent(self._k_hist, t)
        if k is not None and k > 0:
            out.k_ground = k
            out.ego_speed_mps = k * out.f_px * self.camera_height_m
        yaw = self._median_recent(self._yaw_hist, t)
        out.yaw_px_s = yaw or 0.0
        if out.ego_speed_mps is not None:
            out.moving = out.ego_speed_mps >= self.moving_speed_mps
        else:
            out.moving = out.median_flow_px >= 2.0
        return out


@dataclass
class CameraMotion:
    moving_now: bool
    recently_moved: bool
    shift_px: float
    scale: float
    warp: np.ndarray | None


class CameraMotionMonitor:
    """CCTV: phát hiện camera bị rung/xoay/zoom (gió, camera PTZ, quay tay) bằng dòng quang học của NỀN.

    Khi camera vừa chuyển động, mọi xe trong khung đều "trôi" theo → các rule dựa trên hướng/tốc độ (ngược chiều,
    tốc độ, vượt vạch...) sẽ báo giả hàng loạt. Pipeline tạm ngưng các rule đó trong lúc camera chuyển động và
    `hold_s` giây sau đó (cửa sổ hồi quy vận tốc còn chứa phần trôi).
    """

    def __init__(self, thr_px: float = 1.2, thr_scale: float = 0.004, hold_s: float = 1.5, downscale: float = 0.5):
        self.est = EgoMotionEstimator(downscale=downscale, smooth_s=0.0)
        self.thr_px = thr_px
        self.thr_scale = thr_scale
        self.hold_s = hold_s
        self.last_move_t: float | None = None
        self.n_moves = 0

    def update(self, frame: np.ndarray, t: float, exclude_boxes: Sequence[Sequence[float]] = ()) -> CameraMotion:
        em = self.est.update(frame, t, exclude_boxes)
        A = em.warp
        shift, scale = 0.0, 1.0
        moving = False
        if A is not None and em.inliers >= 12:
            scale = float(math.hypot(A[0, 0], A[1, 0]))
            H, W = frame.shape[:2]
            c = A @ np.array([W / 2.0, H / 2.0, 1.0])
            shift = float(math.hypot(c[0] - W / 2.0, c[1] - H / 2.0))
            moving = shift > self.thr_px or abs(scale - 1.0) > self.thr_scale
        if moving:
            if self.last_move_t is None or t - self.last_move_t > self.hold_s:
                self.n_moves += 1
            self.last_move_t = t
        recent = self.last_move_t is not None and t - self.last_move_t <= self.hold_s
        return CameraMotion(moving, recent, shift, scale, A if moving else None)
