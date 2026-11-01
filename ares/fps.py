"""Điều tiết FPS theo trạng thái train (workflow: "Bình thường: Full FPS — Khi Train: 1 FPS").

Vì sao: train và suy luận chạy trên CÙNG một máy (thường cùng một GPU). Trong lúc train, vision vẫn phải chạy để
đèn/điều hòa có số người, nhưng không cần 15 khung/giây — bộ đệm đèn là trung bình 30 giây, điều hòa 10 phút, nên
1 khung/giây vẫn đủ. Hạ FPS nhường GPU/CPU cho tiến trình train; train xong tự trả lại FPS bình thường.

Không đụng tới dữ liệu điều khiển: bộ đệm tính trung bình CÓ TRỌNG SỐ THỜI GIAN, nên đổi FPS không làm lệch số người.
"""

from __future__ import annotations

import threading
from typing import Callable

from ares.config import AresSettings
from ares.state import Event


class FpsGovernor:
    def __init__(self, cfg: AresSettings, is_training: Callable[[], bool], clock=None,
                 emit: Callable[[Event], None] | None = None):
        self.cfg = cfg
        self.is_training = is_training
        self.clock = clock
        self.emit = emit if emit is not None else (lambda e: None)
        self._lock = threading.Lock()
        self._was = False

    # ---------------------------------------------------------------- trạng thái
    @property
    def training(self) -> bool:
        try:
            tr = bool(self.is_training())
        except Exception:  # noqa: BLE001 - không bao giờ để lỗi ở đây làm chết luồng vision
            tr = False
        with self._lock:
            changed, self._was = tr != self._was, tr
        if changed and self.clock is not None:
            cfg = self.cfg
            if tr:
                self.emit(Event(self.clock.time(), "FPS_THROTTLE", "info",
                                f"Đang train: hạ camera chính xuống {cfg.vision_fps_training:g} FPS, camera phụ "
                                f"{cfg.hub_fps_training:g} vòng/giây để nhường GPU/CPU."))
            else:
                self.emit(Event(self.clock.time(), "FPS_RESTORED", "info",
                                f"Train xong: trả camera chính về {cfg.vision_fps:g} FPS, camera phụ {cfg.hub_fps:g} vòng/giây."))
        return tr

    # ---------------------------------------------------------------- chu kỳ (giây giữa 2 nhịp)
    def vision_period(self) -> float:
        cfg = self.cfg
        if cfg.simulation_mode:  # mô phỏng: nhịp theo đồng hồ giả, giữ như cũ
            base = cfg.vision_interval_s
            return max(base, 1.0 / cfg.vision_fps_training) if self.training else base
        return 1.0 / (cfg.vision_fps_training if self.training else cfg.vision_fps)

    def hub_period(self) -> float:
        cfg = self.cfg
        if cfg.simulation_mode:
            base = cfg.hub_interval_s
            return max(base, 1.0 / cfg.hub_fps_training) if self.training else base
        return 1.0 / (cfg.hub_fps_training if self.training else cfg.hub_fps)

    def status(self) -> dict:
        tr = self.training
        return {"mode": "training" if tr else "normal",
                "vision_fps_target": round(1.0 / self.vision_period(), 2),
                "hub_fps_target": round(1.0 / self.hub_period(), 2),
                "vision_fps_normal": self.cfg.vision_fps, "vision_fps_training": self.cfg.vision_fps_training}
