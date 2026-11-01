"""Hai kiểu chết nửa chừng mà bản trước không chịu được.

1. Worker chết SAU khi consume nhưng TRƯỚC khi ack -> message kẹt ở 'inflight'
   vĩnh viễn, không ai giao lại. Mất mẫu âm thầm, không lỗi nào nổ ra.
2. Message được giao lại cho frame đã vào review_queue -> INSERT lần hai vi phạm
   UNIQUE(frame_id) -> IntegrityError giết cả tiến trình worker giữa lô.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from collector.service import Collector
from common.db import ReviewItem, session_scope
from common.queue import SQLiteQueue, get_queue, reset_queue
from worker.consumer import Worker
from worker.embedding import HistogramEmbedder
from worker.pca import fit_pca
from tests.test_worker_end_to_end import scene


@pytest.fixture
def frozen_pca(settings):
    emb = HistogramEmbedder()
    vecs = emb.embed([scene(i) for i in range(120)])
    pca = fit_pca(vecs, settings.pca_version, emb.name, n_components=16, min_samples=100)
    pca.save(settings.pca_path)
    return pca


def collect(settings, n=8):
    settings.random_sample_rate = 1.0
    c = Collector(settings)
    base = datetime(2026, 2, 1, 10, tzinfo=timezone.utc)
    for i in range(n):
        c.on_frame(scene(i), "cam01", base + timedelta(minutes=i), i, [])
    return c


def test_inflight_messages_are_requeued_after_crash(settings):
    q = SQLiteQueue(settings.data_root / "crash.db")
    from common.schemas import QueueMessage, Source

    for i in range(5):
        q.enqueue(
            QueueMessage(
                msg_id=f"m{i}", frame_id=f"f{i}", path=f"p{i}.jpg", cam_id="cam01",
                ts=datetime(2026, 2, 1, tzinfo=timezone.utc), frame_idx=i,
                width=10, height=10, source=Source.RANDOM,
            )
        )
    got = q.consume(count=5)
    assert len(got) == 5          # đã giao, chưa ack
    assert q.consume(count=5) == []  # không còn gì 'ready'
    q.close()                      # ← worker chết ở đây

    again = SQLiteQueue(settings.data_root / "crash.db")
    assert len(again.consume(count=5)) == 5, "message dang dở phải được giao lại"
    again.close()


def test_redelivered_frame_does_not_kill_worker(settings, frozen_pca):
    collect(settings, 6)
    w = Worker(settings, detector=None, embedder_kind="hist")
    w.run(idle_timeout_s=0.2)
    with session_scope() as s:
        n_first = s.query(ReviewItem).count()
    assert n_first > 0

    # Giao lại đúng những frame đó (mô phỏng crash sau xử lý, trước ack).
    reset_queue()
    settings.random_sample_rate = 1.0
    from common.db import Frame as FrameRow
    from common.queue import get_queue as gq
    from common.schemas import QueueMessage, Source

    q = gq(settings)
    with session_scope() as s:
        rows = [(r.id, r.path, r.width, r.height) for r in s.query(FrameRow).all()]
    for fid, path, wdt, hgt in rows:
        q.enqueue(
            QueueMessage(
                msg_id=f"re-{fid}", frame_id=fid, path=path, cam_id="cam01",
                ts=datetime(2026, 2, 1, tzinfo=timezone.utc), frame_idx=0,
                width=wdt, height=hgt, source=Source.RANDOM,
            )
        )

    # Kịch bản thật: review_queue commit theo từng frame, còn faiss index chỉ
    # save mỗi 30 giây. Crash giữa chừng => index mất phần chưa save, review_queue
    # thì không. Frame được giao lại KHÔNG bị dedup chặn, đi thẳng tới _finalize.
    for f in settings.artifacts_dir.glob("faiss_*"):
        f.unlink()

    w2 = Worker(settings, detector=None, embedder_kind="hist")
    w2.run(idle_timeout_s=0.2)          # không được ném IntegrityError
    with session_scope() as s:
        assert s.query(ReviewItem).count() == n_first, "không được nhân bản review item"
