"""Track history + phát hiện sự cố tracking.

Đây là nguồn dữ liệu cho FAILURE MINING (spec 2.1-B) — nguồn duy nhất bắt được
false negative, vì người bị bỏ sót không sinh confidence thấp và không sinh ID nhảy.

Cơ chế: phân tích TRỄ 1 FRAME. Muốn biết frame t có track gap hay không thì phải
thấy t+1 đã. Vì vậy analyze() trả sự kiện của frame t khi nhận frame t+1.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field

from common.schemas import BBox, FailureReason


@dataclass
class FailureEvent:
    frame_idx: int
    reason: FailureReason
    track_ids: list[int] = field(default_factory=list)
    neighbor_boxes: list[BBox] = field(default_factory=list)  # gợi ý cho pre-annotation
    detail: str = ""


@dataclass
class _Track:
    track_id: int
    first_frame: int
    last_frame: int
    last_box: BBox
    n_frames: int = 1
    finished: bool = False


class TrackHistory:
    """Giữ N frame gần nhất + trạng thái track để soi lỗi.

    Không giữ ảnh — chỉ toạ độ. Bộ nhớ không phụ thuộc độ phân giải.
    """

    def __init__(self, maxlen: int = 90, short_track_frames: int = 5, gap_iou: float = 0.3):
        self.maxlen = maxlen
        self.short_track_frames = short_track_frames
        self.gap_iou = gap_iou
        self.frames: OrderedDict[int, list[BBox]] = OrderedDict()
        self.tracks: dict[int, _Track] = {}
        self._order: deque[int] = deque()
        self._last_idx: int | None = None
        self.finished_lengths: list[int] = []

    # ---------- ghi ----------
    def add(self, frame_idx: int, boxes: list[BBox]) -> list[FailureEvent]:
        self.frames[frame_idx] = boxes
        self._order.append(frame_idx)
        while len(self._order) > self.maxlen:
            old = self._order.popleft()
            self.frames.pop(old, None)

        events: list[FailureEvent] = []
        current_ids = {b.track_id for b in boxes if b.track_id is not None}

        for b in boxes:
            if b.track_id is None:
                continue
            t = self.tracks.get(b.track_id)
            if t is None:
                self.tracks[b.track_id] = _Track(b.track_id, frame_idx, frame_idx, b)
            else:
                t.last_frame = frame_idx
                t.last_box = b
                t.n_frames += 1

        # Track biến mất quá lâu -> đóng lại, kiểm tra "chết sớm".
        for tid, t in list(self.tracks.items()):
            if t.finished:
                continue
            if tid not in current_ids and frame_idx - t.last_frame > self.short_track_frames * 2:
                t.finished = True
                self.finished_lengths.append(t.n_frames)
                if t.n_frames < self.short_track_frames:
                    events.append(
                        FailureEvent(
                            frame_idx=t.last_frame,
                            reason=FailureReason.SHORT_TRACK,
                            track_ids=[tid],
                            neighbor_boxes=[t.last_box],
                            detail=f"track {tid} chỉ sống {t.n_frames} frame",
                        )
                    )

        # ID switch: cùng vị trí, ID khác, giữa hai frame liền kề.
        if self._last_idx is not None and frame_idx - self._last_idx == 1:
            events.extend(self._detect_id_switch(self._last_idx, frame_idx))

        # Track gap: chỉ kết luận được cho frame t-1 khi đã có frame t.
        if self._last_idx is not None:
            events.extend(self._detect_track_gap(self._last_idx))

        self._last_idx = frame_idx
        self._gc_tracks(frame_idx)
        return events

    # ---------- soi lỗi ----------
    def _detect_id_switch(self, prev_idx: int, cur_idx: int) -> list[FailureEvent]:
        prev = self.frames.get(prev_idx, [])
        cur = self.frames.get(cur_idx, [])
        out: list[FailureEvent] = []
        for pb in prev:
            if pb.track_id is None:
                continue
            best, best_iou = None, 0.0
            for cb in cur:
                i = pb.iou(cb)
                if i > best_iou:
                    best, best_iou = cb, i
            if best is not None and best_iou > 0.7 and best.track_id != pb.track_id:
                out.append(
                    FailureEvent(
                        frame_idx=cur_idx,
                        reason=FailureReason.ID_SWITCH,
                        track_ids=[pb.track_id, best.track_id or -1],
                        neighbor_boxes=[pb, best],
                        detail=f"IoU {best_iou:.2f} nhưng ID {pb.track_id} -> {best.track_id}",
                    )
                )
        return out

    def _detect_track_gap(self, t: int) -> list[FailureEvent]:
        """Track có ở t-1 và t+1 nhưng mất ở t -> gần chắc chắn false negative."""
        prev_idx, next_idx = t - 1, t + 1
        if prev_idx not in self.frames or next_idx not in self.frames or t not in self.frames:
            return []

        prev_ids = {b.track_id: b for b in self.frames[prev_idx] if b.track_id is not None}
        next_ids = {b.track_id: b for b in self.frames[next_idx] if b.track_id is not None}
        cur_ids = {b.track_id for b in self.frames[t] if b.track_id is not None}

        missing = (prev_ids.keys() & next_ids.keys()) - cur_ids
        if not missing:
            return []

        neighbors: list[BBox] = []
        for tid in missing:
            neighbors.append(interpolate(prev_ids[tid], next_ids[tid], tid))
        return [
            FailureEvent(
                frame_idx=t,
                reason=FailureReason.TRACK_GAP,
                track_ids=sorted(missing),
                neighbor_boxes=neighbors,
                detail=f"{len(missing)} track biến mất 1 frame rồi quay lại",
            )
        ]

    def _gc_tracks(self, frame_idx: int) -> None:
        cutoff = frame_idx - self.maxlen * 2
        for tid, t in list(self.tracks.items()):
            if t.last_frame < cutoff:
                self.tracks.pop(tid, None)

    # ---------- đọc ----------
    def boxes_at(self, frame_idx: int) -> list[BBox]:
        return self.frames.get(frame_idx, [])

    def neighbor_suggestions(self, frame_idx: int) -> list[BBox]:
        """Bbox nội suy từ t-1/t+1 — gợi ý vùng model có thể đã bỏ sót."""
        prev = {b.track_id: b for b in self.frames.get(frame_idx - 1, []) if b.track_id is not None}
        nxt = {b.track_id: b for b in self.frames.get(frame_idx + 1, []) if b.track_id is not None}
        cur = {b.track_id for b in self.frames.get(frame_idx, []) if b.track_id is not None}
        return [interpolate(prev[t], nxt[t], t) for t in (prev.keys() & nxt.keys()) - cur]

    @property
    def avg_track_length(self) -> float:
        lens = self.finished_lengths[-500:]
        return sum(lens) / len(lens) if lens else 0.0

    @property
    def active_count(self) -> int:
        return sum(1 for t in self.tracks.values() if not t.finished)


def interpolate(a: BBox, b: BBox, track_id: int | None = None) -> BBox:
    """Nội suy tuyến tính giữa hai bbox liền kề. interpolated=True: CHỈ LÀ GỢI Ý."""
    return BBox(
        x1=(a.x1 + b.x1) / 2,
        y1=(a.y1 + b.y1) / 2,
        x2=(a.x2 + b.x2) / 2,
        y2=(a.y2 + b.y2) / 2,
        conf=0.0,
        cls=0,
        track_id=track_id,
        interpolated=True,
    )
