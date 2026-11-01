"""Ràng buộc #4: queue chỉ mang path + metadata, KHÔNG mang bytes ảnh."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from common.queue import _serialize, get_queue
from common.schemas import BBox, QueueMessage, Source
from common.storage import get_storage


def make_msg(**kw) -> QueueMessage:
    base = dict(
        msg_id=uuid.uuid4().hex,
        frame_id=uuid.uuid4().hex,
        path="cam01/2026-01-01/09/abc.jpg",
        cam_id="cam01",
        ts=datetime.now(timezone.utc),
        frame_idx=10,
        width=1920,
        height=1080,
        source=Source.UNCERTAINTY,
        boxes=[BBox(x1=1, y1=2, x2=30, y2=60, conf=0.5)],
    )
    base.update(kw)
    return QueueMessage(**base)


def test_schema_forbids_extra_fields():
    """Không ai nhét được field 'image' vào payload, kể cả vô tình."""
    with pytest.raises(Exception):
        QueueMessage(**{**make_msg().model_dump(), "image": "..."})


def test_payload_is_small():
    blob = _serialize(make_msg())
    assert len(blob) < 4000, "payload phình to bất thường"
    assert "jpg" in blob and "base64" not in blob


def test_serialize_rejects_oversized_payload():
    """Chốt chặn cứng: ai đó nhồi ảnh base64 vào extra sẽ bị chặn."""
    fake_image_b64 = "A" * 400_000
    msg = make_msg(extra={"thumbnail": fake_image_b64})
    with pytest.raises(ValueError, match="quá lớn"):
        _serialize(msg)


def test_roundtrip_through_persistent_queue(settings):
    q = get_queue(settings)
    msg = make_msg()
    q.enqueue(msg)
    assert q.depth() == 1

    got = q.consume(count=5)
    assert len(got) == 1
    entry_id, out = got[0]
    assert out.frame_id == msg.frame_id
    assert out.path == msg.path
    assert out.boxes[0].conf == 0.5
    q.ack(entry_id)
    assert q.depth() == 0


def test_queue_survives_restart(settings):
    """Điểm khác cốt lõi so với queue.Queue in-memory của v1."""
    import common.queue as qm

    q = get_queue(settings)
    q.enqueue(make_msg())
    q.enqueue(make_msg())
    qm.reset_queue()  # mô phỏng process chết

    q2 = get_queue(settings)
    assert q2.depth() == 2, "queue mất dữ liệu sau restart"


def test_image_goes_to_storage_not_queue(settings, img):
    """Đường đi đúng: ảnh -> storage; queue chỉ biết path."""
    storage = get_storage(settings)
    rel = "cam01/2026-01-01/09/test.jpg"
    storage.write_image(rel, img)

    q = get_queue(settings)
    q.enqueue(make_msg(path=rel))
    _eid, msg = q.consume(1)[0]

    assert isinstance(msg.path, str)
    loaded = storage.read_image(msg.path)
    assert loaded.shape == img.shape
    assert not any(isinstance(v, (bytes, bytearray, np.ndarray)) for v in msg.model_dump().values())
