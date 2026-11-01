"""Payload đi qua queue và giữa các module.

RÀNG BUỘC (spec #4): QueueMessage KHÔNG BAO GIỜ chứa bytes ảnh.
`test_queue_payload.py` kiểm tra điều này bằng cách quét kiểu field.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Source(StrEnum):
    UNCERTAINTY = "uncertainty"
    FAILURE_MINING = "failure_mining"
    RANDOM = "random"


class FailureReason(StrEnum):
    TRACK_GAP = "track_gap"
    ID_SWITCH = "id_switch"
    SHORT_TRACK = "short_track"


class ReviewAction(StrEnum):
    APPROVE = "approve"
    FIX = "fix"
    REJECT = "reject"


class BBox(BaseModel):
    """Toạ độ pixel trên FULL FRAME GỐC (không phải toạ độ đã resize)."""

    model_config = ConfigDict(frozen=False)

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    cls: int = 0
    track_id: int | None = None
    interpolated: bool = False  # bbox nội suy từ t-1/t+1, chỉ là gợi ý

    @field_validator("x2")
    @classmethod
    def _x_order(cls, v, info):
        if "x1" in info.data and v < info.data["x1"]:
            raise ValueError("x2 < x1")
        return v

    def to_yolo(self, img_w: int, img_h: int) -> str:
        cx = (self.x1 + self.x2) / 2 / img_w
        cy = (self.y1 + self.y2) / 2 / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        return f"{self.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Detection(BaseModel):
    """Kết quả 1 frame ở luồng realtime (in-memory, không đi qua queue)."""

    cam_id: str
    frame_idx: int
    ts: datetime
    width: int
    height: int
    boxes: list[BBox] = Field(default_factory=list)
    inference_ms: float = 0.0


class QueueMessage(BaseModel):
    """Thứ duy nhất đi qua Redis Streams. Chỉ path + metadata."""

    model_config = ConfigDict(extra="forbid")

    msg_id: str
    frame_id: str
    path: str  # đường dẫn tương đối trong storage, KHÔNG phải bytes
    cam_id: str
    ts: datetime
    frame_idx: int
    width: int
    height: int
    source: Source
    failure_reason: FailureReason | None = None
    boxes: list[BBox] = Field(default_factory=list)  # detection lúc bắt mẫu
    track_ids: list[int] = Field(default_factory=list)
    neighbor_boxes: list[BBox] = Field(default_factory=list)  # từ t-1/t+1 cho nguồn B
    uncertainty_score: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class TrackState(BaseModel):
    track_id: int
    first_frame: int
    last_frame: int
    boxes: dict[int, BBox] = Field(default_factory=dict)  # frame_idx -> box

    @property
    def length(self) -> int:
        return self.last_frame - self.first_frame + 1
