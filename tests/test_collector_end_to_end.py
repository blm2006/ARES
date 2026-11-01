"""Phase 2 nghiệm thu: ảnh GỐC xuống đĩa, queue chỉ có metadata."""

from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np

from collector.service import Collector
from common.queue import get_queue
from common.schemas import BBox, FailureReason, Source
from common.storage import get_storage
from realtime.tracker import FailureEvent


def bx(x, tid=1, conf=0.5):
    return BBox(x1=x, y1=100, x2=x + 40, y2=240, conf=conf, track_id=tid)


def structured_image(h=480, w=640):
    """Ảnh có cấu trúc như cảnh thật. Nhiễu trắng thuần tuý không nén JPEG được,
    dùng nó để đo độ trung thực sẽ ra kết luận sai về codec chứ không về pipeline."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack(
        [(xx * 255 // w).astype(np.uint8),
         (yy * 255 // h).astype(np.uint8),
         np.full((h, w), 128, np.uint8)],
        axis=-1,
    )
    cv2.rectangle(base, (100, 120), (240, 380), (20, 200, 90), -1)
    cv2.circle(base, (450, 200), 70, (200, 40, 40), -1)
    return base


def test_original_frame_written_unmodified(settings):
    img = structured_image()
    c = Collector(settings)
    ts = datetime.now(timezone.utc)
    ev = [FailureEvent(frame_idx=5, reason=FailureReason.TRACK_GAP, track_ids=[1],
                       neighbor_boxes=[bx(100)])]
    ids = c.on_frame(img, "cam01", ts, 5, [bx(100)], failure_events=ev)
    assert len(ids) == 1

    storage = get_storage(settings)
    files = list(settings.images_dir.rglob("*.jpg"))
    assert len(files) == 1

    saved = cv2.imread(str(files[0]))
    assert saved.shape == img.shape, "ảnh bị resize — vi phạm ràng buộc #3"
    # JPEG q=95 có mất mát, nhưng phải rất gần bản gốc.
    assert np.mean(np.abs(saved.astype(int) - img.astype(int))) < 12


def test_queue_gets_path_only(settings, img):
    c = Collector(settings)
    ev = [FailureEvent(frame_idx=1, reason=FailureReason.ID_SWITCH, track_ids=[1, 2])]
    c.on_frame(img, "cam01", datetime.now(timezone.utc), 1, [bx(10)], failure_events=ev)

    q = get_queue(settings)
    assert q.depth() == 1
    _eid, msg = q.consume(1)[0]
    assert msg.source is Source.FAILURE_MINING
    assert msg.failure_reason is FailureReason.ID_SWITCH
    assert msg.path.endswith(".jpg")
    assert msg.width == img.shape[1] and msg.height == img.shape[0]


def test_storage_layout_is_cam_date_hour(settings, img):
    c = Collector(settings)
    ts = datetime(2026, 3, 14, 15, 9, tzinfo=timezone.utc)
    ev = [FailureEvent(frame_idx=1, reason=FailureReason.TRACK_GAP, track_ids=[1])]
    c.on_frame(img, "cam07", ts, 1, [bx(10)], failure_events=ev)

    files = list(settings.images_dir.rglob("*.jpg"))
    rel = files[0].relative_to(settings.images_dir).as_posix()
    assert rel.startswith("cam07/2026-03-14/15/")


def test_random_source_ignores_all_criteria(settings, img):
    """Nguồn C phải lấy được cả frame hoàn toàn rỗng — đó là điểm của nhóm đối chứng."""
    settings.random_sample_rate = 1.0
    c = Collector(settings)
    ids = c.on_frame(img, "cam01", datetime.now(timezone.utc), 1, [])
    assert len(ids) == 1
    _eid, msg = get_queue(settings).consume(1)[0]
    assert msg.source is Source.RANDOM
    assert msg.boxes == []


def test_db_row_has_session_key(settings, img):
    from common.db import Frame as FrameRow
    from common.db import session_scope

    settings.random_sample_rate = 1.0
    c = Collector(settings)
    c.on_frame(img, "cam02", datetime(2026, 5, 1, 8, tzinfo=timezone.utc), 1, [])
    with session_scope() as s:
        row = s.query(FrameRow).one()
    assert row.session_key == "cam02:2026-05-01"


def test_collector_never_raises_on_storage_failure(settings, img, monkeypatch):
    """Realtime không được chết vì learning. Lỗi ghi đĩa -> bỏ frame, không nổ."""
    c = Collector(settings)
    monkeypatch.setattr(
        c.writer.storage, "write_image", lambda *a, **k: (_ for _ in ()).throw(IOError("đĩa hỏng"))
    )
    settings.random_sample_rate = 1.0
    out = c.on_frame(img, "cam01", datetime.now(timezone.utc), 1, [])
    assert out == []
    assert c.writer.failed >= 1
