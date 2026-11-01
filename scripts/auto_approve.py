"""Tự động DUYỆT hàng loạt (pseudo-label): lấy pre-annotation của model làm nhãn.

Dùng để BOOTSTRAP nhanh khi không muốn bấm tay từng frame trong Phase 4.
    python -m scripts.auto_approve --min-conf 0.5
    python -m scripts.auto_approve --min-conf 0.4 --limit 800 --empty skip

Lưu ý: chất lượng nhãn = chất lượng model gốc. Đặt --min-conf cao để chỉ lấy box
chắc chắn. Đây KHÔNG thay được review tay cho dữ liệu cần độ chính xác cao — nhưng
rất hợp để dựng dataset thử và train một bản đầu, rồi chỉ soi tay các frame khó.
"""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import func, select

from common.db import Frame as FrameRow
from common.db import ReviewItem, audit, init_db, session_scope, utcnow
from common.schemas import BBox

log = logging.getLogger("auto_approve")


def main() -> None:
    ap = argparse.ArgumentParser(prog="auto_approve")
    ap.add_argument("--min-conf", type=float, default=0.4,
                    help="chỉ giữ box có conf >= ngưỡng này làm nhãn (mặc định 0.4)")
    ap.add_argument("--limit", type=int, default=None, help="số item tối đa xử lý")
    ap.add_argument("--empty", choices=["skip", "approve", "reject"], default="skip",
                    help="frame không còn box nào sau lọc: skip (để chờ), approve (nhãn rỗng = mẫu âm), reject")
    ap.add_argument("--reviewer", default="auto")
    args = ap.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    init_db()

    done = empty_skipped = 0
    with session_scope() as sess:
        q = (select(ReviewItem)
             .where(ReviewItem.status == "pending")
             .order_by(ReviewItem.priority, ReviewItem.created_at))
        if args.limit:
            q = q.limit(args.limit)
        for item in sess.execute(q).scalars().all():
            frame = sess.get(FrameRow, item.frame_id)
            pre = [BBox(**b) for b in (frame.preannot_boxes or [])]
            final = [b for b in pre if not b.interpolated and b.conf >= args.min_conf]

            if not final and args.empty == "skip":
                empty_skipped += 1
                continue

            item.status = "done"
            item.reviewer = args.reviewer
            item.reviewed_at = utcnow()
            if not final and args.empty == "reject":
                item.action = "reject"
                item.reject_reason = "khac"
                item.final_boxes = []
                item.n_boxes_final = 0
                item.edited = True
            else:
                item.action = "approve"
                item.final_boxes = [b.model_dump() for b in final]
                item.n_boxes_final = len(final)
                item.edited = False
            done += 1

        pending = sess.execute(
            select(func.count()).select_from(ReviewItem).where(ReviewItem.status == "pending")
        ).scalar_one()

    audit(args.reviewer, "auto_approve", "*", {"done": done, "min_conf": args.min_conf})
    log.info("Auto-duyệt %d item (bỏ qua %d frame rỗng). Còn chờ: %d", done, empty_skipped, pending)


if __name__ == "__main__":
    main()
