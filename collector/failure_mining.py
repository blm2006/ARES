"""Nguồn B — failure mining. Nguồn DUY NHẤT bắt được lỗi bỏ sót người.

Nhắc lại vì sao nó quan trọng: cả 6 tiêu chí của v1 đều yêu cầu "đã có detection".
Người bị bỏ sót thì không sinh confidence thấp, không sinh ID nhảy — nên không
bao giờ vào dataset. Mà false negative là lỗi đắt nhất khi đếm người.

Module này chỉ chuyển FailureEvent (do realtime/tracker.py sinh ra) thành quyết
định lấy mẫu, kèm khử trùng lặp thời gian để một sự cố kéo dài không sinh 30 mẫu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common.schemas import BBox, FailureReason
from realtime.tracker import FailureEvent

# Ưu tiên trong nguồn B: bỏ sót người đắt hơn ID nhảy, ID nhảy đắt hơn track chết sớm.
REASON_PRIORITY = {
    FailureReason.TRACK_GAP: 0,
    FailureReason.ID_SWITCH: 1,
    FailureReason.SHORT_TRACK: 2,
}


@dataclass
class FailureSample:
    frame_idx: int
    reason: FailureReason
    track_ids: list[int]
    neighbor_boxes: list[BBox] = field(default_factory=list)
    detail: str = ""


class FailureMiner:
    def __init__(self, cooldown_frames: int = 30, max_per_burst: int = 3):
        self.cooldown_frames = cooldown_frames
        self.max_per_burst = max_per_burst
        self._last_emit: dict[tuple[FailureReason, int], int] = {}

    def select(self, events: list[FailureEvent]) -> list[FailureSample]:
        """Chọn ra sự kiện đáng lấy mẫu, có cooldown theo (reason, track_id)."""
        if not events:
            return []
        events = sorted(events, key=lambda e: REASON_PRIORITY.get(e.reason, 9))
        out: list[FailureSample] = []
        for e in events:
            if len(out) >= self.max_per_burst:
                break
            key_tid = e.track_ids[0] if e.track_ids else -1
            key = (e.reason, key_tid)
            last = self._last_emit.get(key)
            if last is not None and e.frame_idx - last < self.cooldown_frames:
                continue  # cùng một sự cố đang kéo dài, đã lấy mẫu rồi
            self._last_emit[key] = e.frame_idx
            out.append(
                FailureSample(
                    frame_idx=e.frame_idx,
                    reason=e.reason,
                    track_ids=e.track_ids,
                    neighbor_boxes=e.neighbor_boxes,
                    detail=e.detail,
                )
            )
        return out

    def gc(self, current_frame: int, keep: int = 5000) -> None:
        cutoff = current_frame - keep
        for k, v in list(self._last_emit.items()):
            if v < cutoff:
                self._last_emit.pop(k, None)
