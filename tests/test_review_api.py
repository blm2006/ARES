"""Review API: approve / fix / reject và cách nó ảnh hưởng dataset."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from common.db import Frame as FrameRow  # noqa: E402
from common.db import ReviewItem, session_scope  # noqa: E402
from common.schemas import BBox  # noqa: E402
from dataset.builder import load_reviewed  # noqa: E402
from review.app import create_app  # noqa: E402


@pytest.fixture
def client(settings):
    return TestClient(create_app())


def seed(source="failure_mining", priority=0, with_interpolated=True):
    fid = uuid.uuid4().hex
    boxes = [BBox(x1=10, y1=10, x2=60, y2=120, conf=0.8).model_dump()]
    if with_interpolated:
        boxes.append(BBox(x1=200, y1=10, x2=250, y2=120, conf=0.0, interpolated=True).model_dump())
    with session_scope() as s:
        s.add(
            FrameRow(
                id=fid, path=f"cam01/2026-01-01/09/{fid}.jpg", cam_id="cam01",
                session_key="cam01:2026-01-01", ts=datetime(2026, 1, 1, 9), frame_idx=1,
                width=640, height=480, source=source, preannotated=True,
                preannot_model="yolov8n.pt", preannot_boxes=boxes,
            )
        )
        s.add(ReviewItem(frame_id=fid, priority=priority, source=source,
                         status="pending", n_boxes_pre=len(boxes)))
    return fid


def test_priority_order_b_then_a_then_c(client):
    seed("random", 2)
    seed("uncertainty", 1)
    seed("failure_mining", 0)
    d = client.get("/api/next").json()
    assert d["source"] == "failure_mining"


def test_approve_keeps_only_confirmed_boxes(client):
    seed()
    item = client.get("/api/next").json()
    assert len(item["boxes"]) == 2  # 1 của model + 1 gợi ý nội suy

    r = client.post(f"/api/decide/{item['item_id']}", json={"action": "approve"}).json()
    assert r["n_boxes"] == 1, "approve nhanh không được biến gợi ý thành ground truth"
    assert r["edited"] is False

    out = load_reviewed()
    assert len(out) == 1 and len(out[0].boxes) == 1


def test_fix_confirms_interpolated_box(client):
    seed()
    item = client.get("/api/next").json()
    boxes = [dict(b) for b in item["boxes"]]  # người xác nhận cả hai
    r = client.post(f"/api/decide/{item['item_id']}",
                    json={"action": "fix", "boxes": boxes}).json()
    assert r["n_boxes"] == 2 and r["edited"] is True
    assert all(not b.interpolated for b in load_reviewed()[0].boxes)


def test_reject_keeps_frame_out_of_dataset(client):
    seed()
    item = client.get("/api/next").json()
    client.post(f"/api/decide/{item['item_id']}",
                json={"action": "reject", "reject_reason": "anh_hong_blur"})
    assert load_reviewed() == []


def test_double_decide_is_rejected(client):
    seed()
    item = client.get("/api/next").json()
    assert client.post(f"/api/decide/{item['item_id']}", json={"action": "approve"}).status_code == 200
    assert client.post(f"/api/decide/{item['item_id']}", json={"action": "approve"}).status_code == 409


def test_agreement_separated_by_source(client):
    seed("random", 2, with_interpolated=False)
    seed("failure_mining", 0)
    a = client.get("/api/next").json()
    client.post(f"/api/decide/{a['item_id']}", json={"action": "fix", "boxes": []})
    b = client.get("/api/next").json()
    client.post(f"/api/decide/{b['item_id']}", json={"action": "approve"})

    st = client.get("/api/stats").json()
    by = st["agreement"]["by_source"]
    assert by["failure_mining"]["rate"] == 0.0
    assert by["random"]["rate"] == 1.0
