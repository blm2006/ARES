"""Nhận màu nền biển số (trắng / vàng / xanh / đỏ) — điền trường "Màu biển" như tra cứu phạt nguội.

Theo TT 79/2024/TT-BCA: nền trắng chữ đen (cá nhân, tổ chức), nền vàng chữ đen (ô tô kinh doanh vận tải),
nền xanh chữ trắng (cơ quan nhà nước). Nền đỏ chữ trắng là biển quân sự (do Bộ Quốc phòng quản lý).
Ảnh hồng ngoại ban đêm không có màu → trả "unknown".
"""

from __future__ import annotations

import cv2
import numpy as np

from traffic_safety_ai.lpr.vn_plate import COLOR_LABELS


def classify_plate_color(crop: np.ndarray) -> tuple[str, str]:
    if crop is None or crop.size == 0 or crop.ndim != 3:
        return "unknown", COLOR_LABELS["unknown"]
    h, w = crop.shape[:2]
    # bỏ viền 8% để tránh nền xe
    c = crop[int(h * 0.08):max(int(h * 0.92), 1), int(w * 0.06):max(int(w * 0.94), 1)]
    if c.size == 0:
        return "unknown", COLOR_LABELS["unknown"]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(np.int16)
    S = hsv[..., 1].astype(np.int16)
    V = hsv[..., 2].astype(np.int16)
    n = float(H.size)
    lit = V > 60
    # ảnh xám (IR) → không xác định màu
    if float(np.median(S)) < 12 and float((S > 60).mean()) < 0.05 and float((V > 150).mean()) < 0.3:
        return "unknown", COLOR_LABELS["unknown"]
    sat = lit & (S > 80)
    yellow = float((sat & (H >= 15) & (H <= 35)).sum()) / n
    blue = float((sat & (H >= 95) & (H <= 130)).sum()) / n
    red = float((sat & ((H <= 9) | (H >= 166))).sum()) / n
    white = float((lit & (S < 50) & (V > 140)).sum()) / n
    scores = {"yellow": yellow, "blue": blue, "red": red, "white": white}
    best = max(scores, key=scores.get)
    if scores[best] < 0.22:
        return "unknown", COLOR_LABELS["unknown"]
    return best, COLOR_LABELS[best]
