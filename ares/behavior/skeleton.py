"""Khung xương COCO-17 và đặc trưng tư thế từng khung hình.

Toạ độ ảnh: x sang phải, y đi XUỐNG. Mọi khoảng cách/tốc độ được chia cho "chiều cao tham chiếu" của
người đó (ref_h, xem tracker.Track.ref_h) để không phụ thuộc người đứng gần hay xa camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 0 mũi, 1-2 mắt T/P, 3-4 tai T/P, 5-6 vai, 7-8 khuỷu, 9-10 cổ tay, 11-12 hông, 13-14 gối, 15-16 cổ chân
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
HEAD = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
LIMBS = (L_WRI, R_WRI, L_ANK, R_ANK)
N_JOINTS = 17

EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)
# cặp trái/phải để lật ảnh khi tăng cường dữ liệu
FLIP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)

KP_CONF = 0.3  # điểm khớp có độ tin cậy dưới mức này coi như không thấy

POSTURES = ("unknown", "upright", "sitting", "slumped", "lying")


@dataclass
class FrameFeat:
    angle: float | None  # góc thân (vai->hông) so với phương thẳng đứng, độ: 0 = đứng thẳng, 90 = nằm ngang
    aspect: float  # rộng / cao của khung người
    hip_y: float | None  # y của trung điểm hông (px)
    center: tuple[float, float]  # tâm khung người (px)
    head_drop: bool | None  # đầu thấp ngang/dưới vai (gục)
    sitting: bool | None  # đùi gần nằm ngang (hông ~ ngang gối)
    posture: str


def _mean_pts(kp: np.ndarray, idx) -> np.ndarray | None:
    sel = [kp[i, :2] for i in idx if kp[i, 2] >= KP_CONF]
    if not sel:
        return None
    return np.mean(sel, axis=0)


def frame_features(kp: np.ndarray, box: np.ndarray, ref_h: float) -> FrameFeat:
    """kp: (17,3) [x, y, conf] theo pixel; box: (4,) x1,y1,x2,y2; ref_h: chiều cao tham chiếu (px)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    aspect = bw / bh
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    sho = _mean_pts(kp, (L_SHO, R_SHO))
    hip = _mean_pts(kp, (L_HIP, R_HIP))
    head = _mean_pts(kp, HEAD)
    knee = _mean_pts(kp, (L_KNE, R_KNE))

    angle = None
    if sho is not None and hip is not None:
        dx, dy = sho[0] - hip[0], sho[1] - hip[1]
        n = math.hypot(dx, dy)
        if n >= 0.06 * ref_h:  # thân quá ngắn trên ảnh = không đáng tin
            angle = math.degrees(math.acos(max(-1.0, min(1.0, -dy / n))))  # 0 = vai thẳng trên hông
            if angle > 90:  # vai thấp hơn hông (lộn ngược / nằm đầu thấp): chắc chắn không đứng
                angle = max(180.0 - angle, 75.0)

    head_drop = None
    if head is not None and sho is not None:
        head_drop = head[1] >= sho[1] - 0.03 * ref_h
    sitting = None
    if hip is not None and knee is not None:
        sitting = (knee[1] - hip[1]) < 0.18 * ref_h and bh < 0.85 * ref_h

    posture = classify_posture(angle, aspect, head_drop, sitting)
    return FrameFeat(angle, aspect, None if hip is None else float(hip[1]), center, head_drop, sitting, posture)


def classify_posture(angle: float | None, aspect: float, head_drop: bool | None, sitting: bool | None) -> str:
    if angle is not None:
        if angle >= 60:
            return "lying"
        if head_drop and (sitting or 20 <= angle < 60):
            return "slumped"
        if sitting:
            return "sitting"
        if angle < 30:
            return "upright" if aspect < 1.0 else "sitting"
        return "sitting"
    # không thấy thân: dựa vào tỉ lệ khung
    if aspect >= 1.4:
        return "lying"
    if aspect <= 0.7:
        return "upright"
    return "unknown"


def visible_mean(kps: np.ndarray) -> np.ndarray:
    """kps: (n,17,3) -> (17,3) trung bình các khung, chỉ những điểm thấy được; conf = tỉ lệ khung thấy."""
    vis = kps[..., 2] >= KP_CONF
    cnt = vis.sum(axis=0)
    out = np.zeros((N_JOINTS, 3), dtype=np.float32)
    for j in range(N_JOINTS):
        if cnt[j]:
            out[j, :2] = kps[vis[:, j], j, :2].mean(axis=0)
            out[j, 2] = cnt[j] / len(kps)
    return out


def pose_motion(a: np.ndarray, b: np.ndarray, ref_h: float) -> float | None:
    """Độ dịch của các khớp CỬ ĐỘNG NHIỀU NHẤT (trung bình 2 khớp dịch nhiều nhất), theo đơn vị ref_h.

    Khác pose_shift (trung bình mọi khớp): người nằm chỉ cử động tay/đầu/chân thì 2-3 khớp dịch mạnh, 14 khớp còn lại
    đứng yên -> trung bình bị pha loãng ~6 lần và bị tính nhầm là "bất động". Lấy trung bình 2 khớp (vd khuỷu + cổ tay)
    để 1 điểm nhiễu đơn lẻ không tự biến thành "cử động"; vị trí khớp đã được làm mượt theo 0,3 s (visible_mean)."""
    ok = (a[:, 2] > 0.5) & (b[:, 2] > 0.5)
    if ok.sum() < 4:
        return None
    d = np.sort(np.linalg.norm(a[ok, :2] - b[ok, :2], axis=1))[::-1]
    return float(d[:2].mean() / max(ref_h, 1.0))


def pose_shift(a: np.ndarray, b: np.ndarray, ref_h: float) -> float | None:
    """Độ dịch trung bình của các khớp thấy được ở cả hai tư thế a, b (17,3), theo đơn vị ref_h."""
    ok = (a[:, 2] > 0.5) & (b[:, 2] > 0.5)
    if ok.sum() < 4:
        return None
    return float(np.linalg.norm(a[ok, :2] - b[ok, :2], axis=1).mean() / max(ref_h, 1.0))
