"""Bộ lọc Kalman vận tốc không đổi cho hộp dạng (cx, cy, w, h).

Tự viết theo mô tả trong bài báo SORT/ByteTrack/BoT-SORT (trạng thái 8 chiều, nhiễu tỉ lệ với kích thước hộp).
"""

from __future__ import annotations

import numpy as np


class KalmanXYWH:
    def __init__(self, std_weight_position: float = 1.0 / 20, std_weight_velocity: float = 1.0 / 160):
        self.sp = std_weight_position
        self.sv = std_weight_velocity
        self._H = np.eye(4, 8)

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        F = np.eye(8)
        for i in range(4):
            F[i, 4 + i] = dt
        return F

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=np.float64).reshape(4)
        mean = np.r_[z, np.zeros(4)]
        w, h = max(z[2], 1.0), max(z[3], 1.0)
        std = np.array([2 * self.sp * w, 2 * self.sp * h, 2 * self.sp * w, 2 * self.sp * h,
                        10 * self.sv * w, 10 * self.sv * h, 10 * self.sv * w, 10 * self.sv * h])
        return mean, np.diag(std ** 2)

    def predict(self, mean: np.ndarray, cov: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        w, h = max(mean[2], 1.0), max(mean[3], 1.0)
        std = np.array([self.sp * w, self.sp * h, self.sp * w, self.sp * h,
                        self.sv * w, self.sv * h, self.sv * w, self.sv * h])
        Q = np.diag((std ** 2) * max(dt, 1e-3))
        F = self._F(dt)
        mean = F @ mean
        cov = F @ cov @ F.T + Q
        # không để kích thước âm khi dự đoán xa
        mean[2] = max(mean[2], 1.0)
        mean[3] = max(mean[3], 1.0)
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, h = max(mean[2], 1.0), max(mean[3], 1.0)
        std = np.array([self.sp * w, self.sp * h, self.sp * w, self.sp * h])
        R = np.diag(std ** 2)
        return self._H @ mean, self._H @ cov @ self._H.T + R

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov)
        PHt = cov @ self._H.T
        try:
            K = np.linalg.solve(proj_cov.T, PHt.T).T
        except np.linalg.LinAlgError:
            K = PHt @ np.linalg.pinv(proj_cov)
        innov = np.asarray(z, dtype=np.float64).reshape(4) - proj_mean
        mean = mean + K @ innov
        cov = cov - K @ proj_cov @ K.T
        return mean, cov

    @staticmethod
    def apply_affine(mean: np.ndarray, cov: np.ndarray, warp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bù chuyển động camera (GMC): warp 2x3 biến toạ độ frame trước → frame hiện tại."""
        A = warp[:, :2]
        b = warp[:, 2]
        s = float(np.sqrt(abs(np.linalg.det(A)))) or 1.0
        m = mean.copy()
        m[0:2] = A @ mean[0:2] + b
        m[2:4] = mean[2:4] * s
        m[4:6] = A @ mean[4:6]
        m[6:8] = mean[6:8] * s
        R = np.zeros((8, 8))
        R[0:2, 0:2] = A
        R[2:4, 2:4] = np.eye(2) * s
        R[4:6, 4:6] = A
        R[6:8, 6:8] = np.eye(2) * s
        return m, R @ cov @ R.T


def xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float64)


def xywh_to_xyxy(m: np.ndarray) -> np.ndarray:
    cx, cy, w, h = m[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)
