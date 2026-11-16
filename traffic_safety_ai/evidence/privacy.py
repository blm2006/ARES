"""Bảo vệ dữ liệu cá nhân trên ảnh xuất ra (Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15, hiệu lực 01/01/2026).

  * Bản XUẤT/CHIA SẺ: làm mờ đầu người không liên quan (khu vực trường học có nhiều trẻ em) và biển số
    của xe không liên quan; bản nội bộ để xác minh giữ nguyên.
  * Vùng riêng tư cố định (cửa sổ nhà dân...) khai báo trong zones.privacy_mask luôn bị che.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import cv2
import numpy as np


def pixelate(img: np.ndarray, box: Sequence[float], block: int = 10) -> None:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    img[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def head_box(xyxy: Sequence[float], frac: float = 0.24) -> list[float]:
    x1, y1, x2, y2 = xyxy
    h = y2 - y1
    pad = 0.08 * (x2 - x1)
    return [x1 - pad, y1 - 0.03 * h, x2 + pad, y1 + frac * h]


def blur_bystanders(img: np.ndarray, others: Iterable[dict[str, Any]], all_faces: bool = False,
                    involved: Iterable[dict[str, Any]] = ()) -> np.ndarray:
    out = img.copy()
    for o in others:
        if o.get("cls") == "person":
            pixelate(out, head_box(o["xyxy"]), block=max(6, int((o["xyxy"][3] - o["xyxy"][1]) / 10)))
        pb = o.get("plate_box")
        if pb is not None:
            pixelate(out, pb, block=6)
    if all_faces:
        for o in involved:
            if o.get("cls") == "person":
                pixelate(out, head_box(o["xyxy"]), block=8)
    return out


def apply_privacy_mask(img: np.ndarray, polygons: Iterable[Sequence[Sequence[float]]]) -> np.ndarray:
    polys = [np.asarray(p, dtype=np.int32) for p in polygons if len(p) >= 3]
    if not polys:
        return img
    out = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, polys, 255)
    blurred = cv2.GaussianBlur(out, (0, 0), 25)
    out[mask > 0] = blurred[mask > 0]
    return out
