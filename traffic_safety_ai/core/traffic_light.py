"""Nhận trạng thái đèn tín hiệu từ vùng ROI cố định (CCTV) hoặc từ hộp đèn do YOLO phát hiện (POV).

Lưu ý thực tế đã xử lý:
  * Đèn LED nhấp nháy theo PWM: camera đôi khi chụp đúng lúc đèn "tắt" → giữ trạng thái cũ trong hold_s.
  * Đổi trạng thái cần ≥ min_consecutive frame liên tiếp (tránh nhiễu do xe đi ngang che đèn).
  * Ưu tiên vị trí: đèn dọc thì đỏ ở trên, vàng giữa, xanh dưới; đèn ngang đỏ bên trái.
  * Lưu lịch sử chuyển trạng thái → hỏi được "đèn đang màu gì tại thời điểm t" (dùng nội suy lúc cắt vạch).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

STATE_VI = {"red": "ĐỎ", "yellow": "VÀNG", "green": "XANH", "off": "TẮT", "unknown": "CHƯA RÕ"}
STATE_BGR = {"red": (40, 40, 230), "yellow": (0, 200, 255), "green": (60, 200, 60), "off": (120, 120, 120),
             "unknown": (160, 160, 160)}


@dataclass
class LightState:
    id: str
    state: str
    since_t: float
    confidence: float
    raw: str

    def duration(self, t: float) -> float:
        return max(0.0, t - self.since_t)

    @property
    def label_vi(self) -> str:
        return STATE_VI.get(self.state, self.state)


def classify_light_crop(crop: np.ndarray, layout: str = "auto", min_frac: float = 0.006) -> tuple[str, dict[str, float]]:
    """Phân loại màu đèn trong ảnh cắt. Trả (trạng thái, điểm từng màu)."""
    if crop is None or crop.size == 0:
        return "unknown", {}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(np.int16)
    S = hsv[..., 1].astype(np.int16)
    V = hsv[..., 2].astype(np.int16)
    v_thr = max(140.0, float(np.percentile(V, 92)) * 0.8)
    bright = (V >= v_thr) & (S >= 60)
    masks = {
        "red": bright & ((H <= 10) | (H >= 160)),
        "yellow": bright & (H >= 11) & (H <= 34),
        "green": bright & (H >= 40) & (H <= 100),
    }
    h, w = crop.shape[:2]
    area = float(h * w)
    if layout == "auto":
        layout = "vertical" if h >= 1.5 * w else ("horizontal" if w >= 1.5 * h else "none")
    expected = {"vertical": {"red": (None, 1 / 6), "yellow": (None, 0.5), "green": (None, 5 / 6)},
                "horizontal": {"red": (1 / 6, None), "yellow": (0.5, None), "green": (5 / 6, None)}}
    scores: dict[str, float] = {}
    for c, m in masks.items():
        n = int(m.sum())
        s = n / area
        if n and layout in expected:
            ys, xs = np.nonzero(m)
            ex, ey = expected[layout][c]
            if ey is not None:
                d = abs(float(ys.mean()) / h - ey)
            else:
                d = abs(float(xs.mean()) / w - ex)
            s *= 1.0 + 0.6 * max(0.0, 1.0 - d / 0.34) - 0.4 * min(1.0, max(0.0, d - 0.34) / 0.34)
        scores[c] = s
    best = max(scores, key=scores.get)
    if scores[best] < min_frac:
        return "off", scores
    return best, scores


class LightStateTracker:
    def __init__(self, light_id: str, roi: tuple[float, float, float, float] | None = None,
                 layout: str = "auto", min_consecutive: int = 2, hold_s: float = 0.7):
        self.id = light_id
        self.roi = roi
        self.layout = layout
        self.min_consecutive = min_consecutive
        self.hold_s = hold_s
        self.state = "unknown"
        self.since_t = 0.0
        self.conf = 0.0
        self._cand: str | None = None
        self._cand_n = 0
        self._cand_t = 0.0
        self._last_on_t = -1e9
        self.transitions: deque[tuple[float, str]] = deque(maxlen=256)

    def update(self, frame: np.ndarray, t: float) -> LightState:
        if self.roi is None:
            return self.current(t, "unknown")
        x1, y1, x2, y2 = (int(round(v)) for v in self.roi)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        return self.update_from_crop(crop, t)

    def update_from_crop(self, crop: np.ndarray, t: float) -> LightState:
        raw, scores = classify_light_crop(crop, self.layout)
        tot = sum(scores.values()) or 1.0
        if raw in ("red", "yellow", "green"):
            self._last_on_t = t
        if raw == self.state:
            self._cand, self._cand_n = None, 0
            self.conf = scores.get(raw, 0.0) / tot if raw in scores else self.conf
            return self.current(t, raw)
        if raw in ("off", "unknown") and self.state in ("red", "yellow", "green") and t - self._last_on_t <= self.hold_s:
            return self.current(t, raw)    # nhấp nháy PWM: giữ trạng thái cũ
        if raw == self._cand:
            self._cand_n += 1
        else:
            self._cand, self._cand_n, self._cand_t = raw, 1, t
        if self._cand_n >= self.min_consecutive:
            self.state = raw
            self.since_t = self._cand_t
            self.conf = scores.get(raw, 0.0) / tot if raw in scores else 0.5
            self.transitions.append((self.since_t, raw))
            self._cand, self._cand_n = None, 0
        return self.current(t, raw)

    def current(self, t: float, raw: str) -> LightState:
        return LightState(self.id, self.state, self.since_t, self.conf, raw)

    def state_at(self, t: float) -> str:
        """Trạng thái đèn tại thời điểm t (theo lịch sử chuyển trạng thái)."""
        s = "unknown"
        for tt, st in self.transitions:
            if tt <= t:
                s = st
            else:
                break
        return s

    def red_since(self, t: float) -> float | None:
        """Đèn đỏ đã bật được bao lâu tính đến t (None nếu tại t không đỏ)."""
        start = None
        for tt, st in self.transitions:
            if tt > t:
                break
            start = tt if st == "red" else None
        return None if start is None else t - start


def pick_relevant_light(boxes: list[np.ndarray], frame_w: int, frame_h: int) -> int | None:
    """POV: chọn đèn liên quan nhất — to nhất trong nửa trên, gần giữa khung hình."""
    best, best_s = None, 0.0
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        if y2 > frame_h * 0.65:
            continue
        cx = (x1 + x2) / 2
        s = (x2 - x1) * (y2 - y1) * (1.0 - 0.6 * abs(cx - frame_w / 2) / (frame_w / 2))
        if s > best_s:
            best, best_s = i, s
    return best
