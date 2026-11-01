#!/usr/bin/env python3
"""Đẩy lại vào queue những frame đã ghi xuống đĩa nhưng KHÔNG còn message nào.

Vì sao có frame mồ côi: message queue và bản ghi DB là hai thứ tách rời. Đổi
QUEUE_BACKEND (redis <-> sqlite), xoá queue.db, hoặc worker ack rồi chết trước khi
ghi kết quả — đều để lại frame nằm trên đĩa mà Phase 3 không bao giờ nhìn thấy.
Chúng không hỏng, chỉ là vô hình.

Script này tìm frame: embedded=0, chưa có trong review_queue, chưa bị xoá, ẢNH CÒN
TRÊN ĐĨA, và KHÔNG có message nào đang chờ — rồi enqueue lại.

    python -m scripts.requeue_orphans            # xem có bao nhiêu, không đổi gì
    python -m scripts.requeue_orphans --apply    # đẩy vào queue thật
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import uuid

from sqlalchemy import select

from common.config import get_settings
from common.db import Frame, ReviewItem, init_db, session_scope
from common.queue import get_queue
from common.schemas import FailureReason, QueueMessage, Source
from common.storage import get_storage

logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
log = logging.getLogger("requeue")


def frame_ids_in_queue(settings) -> set[str]:
    """Frame đã có message đang chờ — enqueue lại là xử lý hai lần vô ích."""
    if settings.queue_backend != "sqlite":
        log.warning("Backend %s: không đọc được nội dung queue, bỏ qua bước lọc trùng.",
                    settings.queue_backend)
        return set()
    p = settings.data_root / "queue.db"
    if not p.exists():
        return set()
    out: set[str] = set()
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10)
    try:
        for (payload,) in con.execute("SELECT payload FROM q WHERE state IN ('ready','inflight')"):
            try:
                out.add(json.loads(payload)["frame_id"])
            except Exception:  # noqa: BLE001
                continue
    finally:
        con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="requeue_orphans")
    ap.add_argument("--apply", action="store_true", help="thật sự enqueue (mặc định chỉ xem)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    s = get_settings()
    init_db()
    storage = get_storage(s)
    queued = frame_ids_in_queue(s)

    with session_scope() as sess:
        reviewed = set(sess.execute(select(ReviewItem.frame_id)).scalars().all())
        rows = sess.execute(
            select(Frame).where(Frame.embedded.is_(False), Frame.deleted_at.is_(None))
            .order_by(Frame.ts)
        ).scalars().all()
        cand = [
            {
                "id": r.id, "path": r.path, "cam_id": r.cam_id, "ts": r.ts,
                "frame_idx": r.frame_idx, "width": r.width, "height": r.height,
                "source": r.source, "failure_reason": r.failure_reason,
                "uncertainty_score": r.uncertainty_score, "session_key": r.session_key,
            }
            for r in rows
            if r.id not in reviewed and r.id not in queued
        ]

    alive = [c for c in cand if storage.exists(c["path"])]
    missing = len(cand) - len(alive)
    if args.limit:
        alive = alive[: args.limit]

    by_session: dict[str, int] = {}
    for c in alive:
        by_session[c["session_key"]] = by_session.get(c["session_key"], 0) + 1

    print(f"\nFrame mồ côi tìm được : {len(alive):,}")
    if missing:
        print(f"  (bỏ qua {missing:,} frame vì ảnh không còn trên đĩa)")
    if queued:
        print(f"  (bỏ qua {len(queued):,} frame đã có message đang chờ)")
    if by_session:
        print("\n  Theo session (cam + ngày):")
        for k in sorted(by_session):
            print(f"    {k}  {by_session[k]:,}")
    print()

    if not alive:
        print("Không có gì để đẩy lại.")
        return 0
    if not args.apply:
        print("Chạy lại với --apply để thật sự đẩy vào queue.")
        return 0

    q = get_queue(s)
    n = 0
    for c in alive:
        try:
            q.enqueue(
                QueueMessage(
                    msg_id=uuid.uuid4().hex,
                    frame_id=c["id"],
                    path=c["path"],
                    cam_id=c["cam_id"],
                    ts=c["ts"],
                    frame_idx=c["frame_idx"],
                    width=c["width"],
                    height=c["height"],
                    source=Source(c["source"]),
                    failure_reason=FailureReason(c["failure_reason"]) if c["failure_reason"] else None,
                    boxes=[],  # detection gốc không lưu lại; pre-annotation sẽ chạy model
                    uncertainty_score=c["uncertainty_score"],
                )
            )
            n += 1
        except Exception:  # noqa: BLE001
            log.exception("enqueue lỗi, bỏ frame %s", c["id"])

    print(f"Đã đẩy {n:,} frame vào queue. Chạy Phase 3 để xử lý:")
    print("    python scripts/startup.py --phase 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
