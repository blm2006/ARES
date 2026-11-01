"""Phase 3+4 nghiệm thu, chạy thật không cần GPU/ultralytics:

collector -> queue -> worker (embedding, PCA đã freeze, dedup) -> review_queue
-> người duyệt -> dataset builder -> split không leakage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from collector.service import Collector
from common.db import Frame as FrameRow
from common.db import ReviewItem, session_scope
from common.queue import get_queue
from common.schemas import BBox
from common.storage import get_storage
from worker.consumer import Worker
from worker.embedding import HistogramEmbedder
from worker.pca import fit_pca


def scene(seed: int, h=240, w=320):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack(
        [(xx * 255 // w).astype(np.uint8), (yy * 255 // h).astype(np.uint8),
         np.full((h, w), 100, np.uint8)], axis=-1
    )
    for _ in range(4):
        x, y = int(rng.integers(10, w - 60)), int(rng.integers(10, h - 60))
        cv2.rectangle(img, (x, y), (x + 50, y + 50),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


@pytest.fixture
def frozen_pca(settings):
    emb = HistogramEmbedder()
    vecs = emb.embed([scene(i) for i in range(120)])
    pca = fit_pca(vecs, settings.pca_version, emb.name, n_components=16, min_samples=100)
    pca.save(settings.pca_path)
    return pca


def collect(settings, n=12):
    settings.random_sample_rate = 1.0
    c = Collector(settings)
    base = datetime(2026, 2, 1, 10, tzinfo=timezone.utc)
    for i in range(n):
        c.on_frame(scene(i), "cam01", base + timedelta(minutes=i), i, [])
    return c


def test_worker_moves_frames_into_review_queue(settings, frozen_pca):
    collect(settings, 10)
    assert get_queue(settings).depth() == 10

    w = Worker(settings, detector=None, embedder_kind="hist")
    w.run(max_messages=10, idle_timeout_s=0.2)

    with session_scope() as s:
        n_review = s.query(ReviewItem).count()
        n_pre = s.query(FrameRow).filter(FrameRow.preannotated.is_(True)).count()
    assert n_review > 0
    assert n_pre == n_review
    assert get_queue(settings).depth() == 0


def test_identical_frames_are_deduped(settings, frozen_pca):
    """Ảnh giống hệt nhau chỉ được vào dataset một lần."""
    settings.random_sample_rate = 1.0
    settings.coreset_enabled = False
    c = Collector(settings)
    base = datetime(2026, 2, 1, 10, tzinfo=timezone.utc)
    img = scene(99)
    for i in range(6):
        c.on_frame(img, "cam01", base + timedelta(minutes=i), i, [])

    w = Worker(settings, detector=None, embedder_kind="hist")
    w.run(max_messages=6, idle_timeout_s=0.2)

    assert w.discarded >= 4, f"dedup bỏ sót: discarded={w.discarded}"
    with session_scope() as s:
        assert s.query(ReviewItem).count() <= 2


def test_dedup_counter_signals_saturation(settings, frozen_pca):
    from worker.dedup import saturation_report

    settings.random_sample_rate = 1.0
    settings.coreset_enabled = False
    c = Collector(settings)
    base = datetime(2026, 2, 1, 10, tzinfo=timezone.utc)
    for i in range(6):
        c.on_frame(scene(42), "cam01", base + timedelta(minutes=i), i, [])
    Worker(settings, detector=None, embedder_kind="hist").run(max_messages=6, idle_timeout_s=0.2)

    rep = saturation_report()
    assert rep and rep[0]["discard_rate"] > 0.5, rep


def test_full_loop_to_dataset(settings, frozen_pca):
    """Duyệt tay xong thì dataset build được, và không có leakage."""
    from dataset.builder import DatasetBuilder
    from dataset.split import assert_no_leakage, split_sessions

    settings.random_sample_rate = 1.0
    settings.coreset_enabled = False
    c = Collector(settings)
    base = datetime(2026, 2, 1, 10, tzinfo=timezone.utc)
    for day in range(6):  # 6 session khác nhau -> split mới có ý nghĩa
        for i in range(3):
            c.on_frame(scene(day * 10 + i), "cam01", base + timedelta(days=day, minutes=i), i, [])

    Worker(settings, detector=None, embedder_kind="hist").run(max_messages=100, idle_timeout_s=0.2)

    # Người duyệt: approve tất cả với 1 box.
    with session_scope() as s:
        for item in s.query(ReviewItem).filter(ReviewItem.status == "pending").all():
            item.status = "done"
            item.action = "approve"
            item.final_boxes = [BBox(x1=10, y1=10, x2=60, y2=120, conf=1.0).model_dump()]
            item.n_boxes_final = 1
            item.edited = False
            item.reviewer = "test"
            item.reviewed_at = datetime.now(timezone.utc)

    version, root, man = DatasetBuilder(settings, link_mode="copy").build(notes="test")
    total = sum(man.counts.values())
    assert total > 0
    assert (root / "data.yaml").exists()
    assert (root / "manifest.json").exists()

    split = split_sessions(
        man.sessions["train"] + man.sessions["val"] + man.sessions["test"],
        settings.split_tuple,
        man.sessions["golden"],
    )
    assert_no_leakage(split)

    # Label phải là toạ độ YOLO chuẩn hoá trên ảnh GỐC.
    labels = list((root / "labels").rglob("*.txt"))
    assert labels
    parts = labels[0].read_text().split()
    assert len(parts) == 5 and parts[0] == "0"
    assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_agreement_rate_computed(settings, frozen_pca):
    from review.agreement import current_agreement, summary

    collect(settings, 6)
    Worker(settings, detector=None, embedder_kind="hist").run(max_messages=6, idle_timeout_s=0.2)

    with session_scope() as s:
        items = s.query(ReviewItem).all()
        for i, item in enumerate(items):
            item.status = "done"
            item.action = "approve" if i % 2 == 0 else "fix"
            item.edited = i % 2 == 1
            item.reviewed_at = datetime.now(timezone.utc)
            item.final_boxes = []

    r = current_agreement(days=7)
    assert 0.0 < r <= 1.0
    assert "by_source" in summary(30)
