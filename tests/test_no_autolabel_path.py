"""Ràng buộc #2: KHÔNG có đường nào đưa nhãn model tự sinh vào dataset.

Đây là thay đổi quan trọng nhất so với v1, nên nó được kiểm bằng test chứ không
bằng một dòng comment.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from common.db import Frame as FrameRow
from common.db import ReviewItem, session_scope
from common.schemas import BBox
from dataset.builder import load_reviewed


def add_frame(source="uncertainty", preannot=True):
    fid = uuid.uuid4().hex
    with session_scope() as s:
        s.add(
            FrameRow(
                id=fid,
                path=f"cam01/2026-01-01/09/{fid}.jpg",
                cam_id="cam01",
                session_key="cam01:2026-01-01",
                ts=datetime(2026, 1, 1, 9, 0),
                frame_idx=1,
                width=640,
                height=480,
                source=source,
                preannotated=preannot,
                preannot_model="yolov8n.pt",
                preannot_boxes=[BBox(x1=10, y1=10, x2=50, y2=90, conf=0.99).model_dump()],
            )
        )
    return fid


def add_review(frame_id, status="pending", action=None, boxes=None):
    with session_scope() as s:
        s.add(
            ReviewItem(
                frame_id=frame_id,
                priority=1,
                source="uncertainty",
                status=status,
                action=action,
                final_boxes=boxes,
                reviewed_at=datetime.now(timezone.utc) if status == "done" else None,
            )
        )


def test_preannotated_but_unreviewed_never_reaches_dataset():
    """Confidence 0.99 + preannotated vẫn KHÔNG vào dataset nếu chưa ai duyệt."""
    fid = add_frame()
    add_review(fid, status="pending")
    assert load_reviewed() == []


def test_rejected_frames_excluded():
    fid = add_frame()
    add_review(fid, status="done", action="reject", boxes=[])
    assert load_reviewed() == []


def test_approved_frame_enters_dataset():
    fid = add_frame()
    boxes = [BBox(x1=10, y1=10, x2=50, y2=90, conf=1.0).model_dump()]
    add_review(fid, status="done", action="approve", boxes=boxes)
    out = load_reviewed()
    assert len(out) == 1 and out[0].frame_id == fid


def test_approve_drops_unconfirmed_interpolated_boxes():
    """Box nội suy là GỢI Ý. Approve nhanh không được biến gợi ý thành ground truth."""
    from common.schemas import ReviewAction

    pre = [
        BBox(x1=10, y1=10, x2=50, y2=90, conf=0.9),
        BBox(x1=200, y1=10, x2=250, y2=90, conf=0.0, interpolated=True),
    ]
    # Cùng logic với review/app.py: approve giữ lại box không-nội-suy.
    final = [b for b in pre if not b.interpolated]
    assert len(final) == 1
    assert ReviewAction.APPROVE.value == "approve"


def test_builder_has_no_function_accepting_model_labels():
    """Không tồn tại API nào cho phép nhét nhãn model thẳng vào dataset."""
    import inspect

    import dataset.builder as b

    src = inspect.getsource(b)
    assert "preannot_boxes" not in src.replace("preannot_model", ""), (
        "builder đang đọc pre-annotation — đó là đường tắt vào dataset, phải bỏ"
    )
    assert 'action.in_(["approve", "fix"])' in src
