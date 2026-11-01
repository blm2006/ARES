"""Governance: TTL xoá ảnh + báo cáo audit.

Ảnh người là dữ liệu cá nhân (NĐ 13/2023/NĐ-CP). Phần này thường bị bỏ và chặn
cả dự án đúng lúc đưa vào vận hành.

Chạy hằng ngày:  python -m scripts.governance ttl
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import timedelta

from sqlalchemy import func, select

from common.config import get_settings
from common.db import AuditLog, Frame as FrameRow, init_db, session_scope, utcnow
from common.storage import get_storage

log = logging.getLogger(__name__)


def ttl_cleanup(dry_run: bool = False) -> dict:
    s = get_settings()
    storage = get_storage(s)
    cutoff = utcnow().replace(tzinfo=None) - timedelta(days=s.image_ttl_days)

    with session_scope() as sess:
        rows = sess.execute(
            select(FrameRow).where(FrameRow.ts < cutoff, FrameRow.deleted_at.is_(None))
        ).scalars().all()
        targets = [(r.id, r.path) for r in rows]

    deleted = 0
    for fid, path in targets:
        if dry_run:
            continue
        try:
            storage.delete(path)
        except Exception:
            log.exception("xoá thất bại %s", path)
            continue
        with session_scope() as sess:
            r = sess.get(FrameRow, fid)
            if r:
                r.deleted_at = utcnow()
        deleted += 1

    # Lưu ý: label trong dataset version đã build KHÔNG bị xoá theo — chúng là
    # toạ độ, không phải ảnh. Nếu chính sách yêu cầu xoá cả dataset đã build thì
    # phải xoá theo dataset version, và điều đó làm mọi so sánh mAP cũ mất nghĩa.
    return {
        "cutoff": cutoff.isoformat(),
        "ttl_days": s.image_ttl_days,
        "candidates": len(targets),
        "deleted": deleted,
        "dry_run": dry_run,
    }


def audit_report(days: int = 7) -> dict:
    since = utcnow() - timedelta(days=days)
    with session_scope() as s:
        rows = s.execute(
            select(AuditLog.action, AuditLog.actor, func.count())
            .where(AuditLog.ts >= since)
            .group_by(AuditLog.action, AuditLog.actor)
        ).all()
    return {"days": days, "events": [{"action": a, "actor": b, "n": int(c)} for a, b, c in rows]}


def main() -> None:
    ap = argparse.ArgumentParser(prog="governance")
    ap.add_argument("cmd", choices=["ttl", "audit"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    init_db()
    if args.cmd == "ttl":
        print(json.dumps(ttl_cleanup(args.dry_run), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(audit_report(args.days), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
