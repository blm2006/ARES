"""Auto rollback theo telemetry thật (1.7), không theo metric offline.

Model thắng trên test set vẫn có thể tệ hơn trên camera thật: ánh sáng, góc đặt,
nén của camera, người đi kiểu khác — không có cái nào trong test set.

Ba tín hiệu, bất kỳ cái nào vượt ngưỡng là rollback:
  - people_count drift so với baseline (đếm sai là lỗi người dùng nhìn thấy đầu tiên)
  - id_switch_per_min tăng (tracking hỏng dù detection có vẻ ổn)
  - FPS tụt (model nặng hơn mà không ai để ý lúc benchmark)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from common.config import Settings, get_settings
from common.metrics import mean, recent_telemetry

log = logging.getLogger(__name__)


@dataclass
class RollbackCheck:
    should_rollback: bool
    signals: dict
    reason: str = ""


def _agg(rows, attr: str) -> float:
    return mean([getattr(r, attr) for r in rows])


def compare(
    cam_id: str,
    baseline_minutes: int = 1440,
    window_minutes: int = 60,
    role_new: str = "canary",
    settings: Settings | None = None,
) -> RollbackCheck:
    s = settings or get_settings()
    new_rows = recent_telemetry(cam_id, window_minutes, role=role_new)
    base_rows = recent_telemetry(cam_id, baseline_minutes, role="primary")

    if len(new_rows) < 3 or len(base_rows) < 3:
        return RollbackCheck(False, {}, "chưa đủ telemetry để kết luận")

    b_count, n_count = _agg(base_rows, "people_count"), _agg(new_rows, "people_count")
    b_sw, n_sw = _agg(base_rows, "id_switch_per_min"), _agg(new_rows, "id_switch_per_min")
    b_fps, n_fps = _agg(base_rows, "fps"), _agg(new_rows, "fps")

    def pct(base: float, now: float) -> float:
        return abs(now - base) / base * 100 if base > 0 else 0.0

    count_drift = pct(b_count, n_count)
    sw_increase = ((n_sw - b_sw) / b_sw * 100) if b_sw > 0 else 0.0
    fps_drop = ((b_fps - n_fps) / b_fps * 100) if b_fps > 0 else 0.0

    signals = {
        "people_count": {"base": round(b_count, 2), "now": round(n_count, 2),
                         "drift_pct": round(count_drift, 2), "limit": s.rollback_count_drift_pct},
        "id_switch_per_min": {"base": round(b_sw, 2), "now": round(n_sw, 2),
                              "increase_pct": round(sw_increase, 2),
                              "limit": s.rollback_id_switch_increase_pct},
        "fps": {"base": round(b_fps, 2), "now": round(n_fps, 2),
                "drop_pct": round(fps_drop, 2), "limit": s.rollback_fps_drop_pct},
    }

    reasons = []
    if count_drift > s.rollback_count_drift_pct:
        reasons.append(f"people_count lệch {count_drift:.1f}%")
    if sw_increase > s.rollback_id_switch_increase_pct:
        reasons.append(f"id_switch tăng {sw_increase:.1f}%")
    if fps_drop > s.rollback_fps_drop_pct:
        reasons.append(f"FPS tụt {fps_drop:.1f}%")

    return RollbackCheck(bool(reasons), signals, "; ".join(reasons))
