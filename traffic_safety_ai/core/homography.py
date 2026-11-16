"""Hiệu chuẩn mặt đường: pixel ↔ mét bằng homography (cho RB-10 tốc độ, RB-08 khoảng cách, RB-12...).

Khác spec gốc:
  * Dùng cv2.findHomography (≥4 điểm, RANSAC khi >4) thay vì getPerspectiveTransform (chỉ đúng 4 điểm)
    → khai báo thêm điểm sẽ giảm sai số.
  * Luôn tính SAI SỐ CHIẾU LẠI (mét). Sai số lớn = hiệu chuẩn hỏng → tốc độ đo được vô nghĩa, hệ thống cảnh báo.
  * Chỉ tin phép đo bên trong vùng đã hiệu chuẩn (xa hơn thì sai số phình lên rất nhanh).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from traffic_safety_ai.utils.geometry import points_in_polygon


class CalibrationError(ValueError):
    pass


class GroundCalibration:
    def __init__(self, image_points: Sequence[Sequence[float]], world_points: Sequence[Sequence[float]],
                 max_reproj_error_m: float = 0.5):
        src = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(world_points, dtype=np.float64).reshape(-1, 2)
        if len(src) < 4 or len(src) != len(dst):
            raise CalibrationError("Cần ≥4 cặp điểm ảnh ↔ mét có cùng số lượng")
        if _degenerate(src) or _degenerate(dst):
            raise CalibrationError("Các điểm hiệu chuẩn gần như thẳng hàng — chọn 4 góc của một hình chữ nhật trên mặt đường")
        method = 0 if len(src) == 4 else cv2.RANSAC
        H, _mask = cv2.findHomography(src, dst, method, 0.3)
        if H is None:
            raise CalibrationError("Không tính được ma trận homography từ các điểm đã khai báo")
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.image_points = src
        self.world_points = dst
        proj = self.to_world(src)
        self.reproj_errors_m = np.linalg.norm(proj - dst, axis=1)
        self.reproj_error_m = float(self.reproj_errors_m.mean())
        self.max_reproj_error_m = max_reproj_error_m
        self.valid_polygon_px = _hull(src)

    # ------------------------------------------------------------------ chiếu
    def to_world(self, pts_px: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        p = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(-1, 2)

    def to_image(self, pts_m: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        p = np.asarray(pts_m, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H_inv).reshape(-1, 2)

    def point_to_world(self, pt: Sequence[float]) -> tuple[float, float]:
        w = self.to_world([pt])[0]
        return float(w[0]), float(w[1])

    def distance_m(self, p1_px: Sequence[float], p2_px: Sequence[float]) -> float:
        w = self.to_world([p1_px, p2_px])
        return float(np.linalg.norm(w[0] - w[1]))

    def width_m_at(self, x_left: float, x_right: float, y: float) -> float:
        """Bề rộng (mét) của đoạn ngang tại hàng ảnh y — dùng ước lượng bề rộng hàng hoá trên xe."""
        return self.distance_m((x_left, y), (x_right, y))

    def in_calibrated_area(self, pt_px: Sequence[float], margin: float = 0.25) -> bool:
        """Điểm có nằm trong (hoặc gần) vùng đã hiệu chuẩn không. margin: nới vùng theo tỉ lệ."""
        poly = self.valid_polygon_px
        if margin > 0:
            c = poly.mean(axis=0)
            poly = c + (poly - c) * (1.0 + margin)
        return bool(points_in_polygon(np.asarray([pt_px], dtype=np.float64), poly)[0])

    @property
    def ok(self) -> bool:
        return self.reproj_error_m <= self.max_reproj_error_m

    def summary(self) -> str:
        state = "đạt" if self.ok else "KHÔNG ĐẠT — kiểm tra lại các điểm"
        return (f"Hiệu chuẩn {len(self.image_points)} điểm, sai số chiếu lại TB {self.reproj_error_m:.3f} m "
                f"(ngưỡng {self.max_reproj_error_m} m): {state}")

    def scaled(self, sx: float, sy: float) -> GroundCalibration:
        src = self.image_points * np.array([sx, sy])
        return GroundCalibration(src, self.world_points, self.max_reproj_error_m)


def _degenerate(pts: np.ndarray) -> bool:
    if len(pts) < 3:
        return True
    c = pts - pts.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    return bool(s[-1] < 1e-6 * max(1.0, s[0]))


def _hull(pts: np.ndarray) -> np.ndarray:
    h = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    return h.astype(np.float64)


def rectangle_world_points(width_m: float, length_m: float) -> list[tuple[float, float]]:
    """4 góc hình chữ nhật thực tế theo thứ tự trái-dưới, phải-dưới, phải-trên, trái-trên (gần → xa camera).

    Người dùng click 4 điểm ảnh đúng thứ tự này trong công cụ hiệu chuẩn.
    """
    return [(0.0, 0.0), (width_m, 0.0), (width_m, length_m), (0.0, length_m)]


def build_calibration(image_points: Sequence[Sequence[float]], world_points: Sequence[Sequence[float]],
                      max_reproj_error_m: float = 0.5) -> GroundCalibration | None:
    if len(image_points) < 4:
        return None
    return GroundCalibration(image_points, world_points, max_reproj_error_m)
