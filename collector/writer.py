"""Frame writer + queue producer.

Hai ràng buộc cứng của module này:
  - Ảnh ghi xuống là FULL FRAME GỐC. Không resize, không crop, không normalize,
    không denoise. Mọi biến đổi là việc của dataloader lúc train (spec #3).
  - Queue chỉ nhận path + metadata. Bytes ảnh không bao giờ đi qua queue (spec #4).

Thứ tự bắt buộc: ghi ảnh -> ghi DB -> enqueue. Nếu đảo lại, worker sẽ nhận message
trỏ tới file chưa tồn tại.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

import numpy as np

from common.config import Settings, get_settings
from common.db import Frame as FrameRow
from common.db import session_scope
from common.queue import QueueBackend
from common.schemas import BBox, FailureReason, QueueMessage, Source
from common.storage import StorageBackend, frame_relpath

log = logging.getLogger(__name__)


def session_key(cam_id: str, ts: datetime) -> str:
    """Khoá session = camera + ngày. Đơn vị chia train/val/test (spec #5)."""
    return f"{cam_id}:{ts:%Y-%m-%d}"


class FrameWriter:
    def __init__(
        self,
        storage: StorageBackend,
        queue: QueueBackend,
        settings: Settings | None = None,
    ):
        self.storage = storage
        self.q = queue
        self.s = settings or get_settings()
        self.written = 0
        self.failed = 0

    def write(
        self,
        image: np.ndarray,
        cam_id: str,
        ts: datetime,
        frame_idx: int,
        source: Source,
        boxes: list[BBox],
        failure_reason: FailureReason | None = None,
        neighbor_boxes: list[BBox] | None = None,
        uncertainty_score: float | None = None,
    ) -> str | None:
        frame_id = uuid.uuid4().hex
        relpath = frame_relpath(cam_id, ts, self.s.image_format, frame_id)
        h, w = image.shape[:2]

        try:
            # 1. ẢNH GỐC xuống storage
            self.storage.write_image(relpath, image)
        except Exception:
            log.exception("ghi ảnh thất bại: %s", relpath)
            self.failed += 1
            return None

        try:
            # 2. metadata vào DB
            with session_scope() as s:
                s.add(
                    FrameRow(
                        id=frame_id,
                        path=relpath,
                        cam_id=cam_id,
                        session_key=session_key(cam_id, ts),
                        ts=ts.replace(tzinfo=None),
                        frame_idx=frame_idx,
                        width=w,
                        height=h,
                        source=source.value,
                        failure_reason=failure_reason.value if failure_reason else None,
                        uncertainty_score=uncertainty_score,
                    )
                )

            # 3. chỉ path + metadata lên queue
            msg = QueueMessage(
                msg_id=uuid.uuid4().hex,
                frame_id=frame_id,
                path=relpath,
                cam_id=cam_id,
                ts=ts,
                frame_idx=frame_idx,
                width=w,
                height=h,
                source=source,
                failure_reason=failure_reason,
                boxes=boxes,
                track_ids=[b.track_id for b in boxes if b.track_id is not None],
                neighbor_boxes=neighbor_boxes or [],
                uncertainty_score=uncertainty_score,
            )
            self.q.enqueue(msg)
        except Exception:
            log.exception("ghi metadata/queue thất bại, rollback ảnh: %s", relpath)
            self.storage.delete(relpath)
            self.failed += 1
            return None

        self.written += 1
        return frame_id
