"""Agreement rate = % pre-annotation được approve mà KHÔNG sửa.

Đây là thước đo thật cho "model có đang tốt lên trên dữ liệu thật hay không",
độc lập với test set. mAP trên test set có thể đẹp vì leakage; agreement rate thì
được tính trên đúng dòng dữ liệu đang chảy vào, do người thật bấm.

Cách đọc:
  - Tính riêng theo source. Agreement trên nguồn random ≈ chất lượng model tổng thể.
    Agreement trên nguồn failure_mining thấp là BÌNH THƯỜNG — đó là các ca khó.
    Trộn chung hai con số này sẽ ra một số vô nghĩa.
  - Ngưỡng nới lỏng review (~97%) chỉ xét trên nguồn random, và chỉ cho lớp đã ổn định.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, select

from common.db import ReviewItem, session_scope, utcnow


@dataclass
class AgreementRow:
    day: str
    source: str
    n_reviewed: int
    n_approved_unedited: int

    @property
    def rate(self) -> float:
        return self.n_approved_unedited / self.n_reviewed if self.n_reviewed else 0.0


def agreement_by_day(days: int = 30) -> list[AgreementRow]:
    since = utcnow() - timedelta(days=days)
    with session_scope() as s:
        rows = s.execute(
            select(
                func.strftime("%Y-%m-%d", ReviewItem.reviewed_at).label("day"),
                ReviewItem.source,
                func.count().label("n"),
                func.sum(
                    (ReviewItem.action == "approve").cast(__import__("sqlalchemy").Integer)
                ).label("n_ok"),
            )
            .where(ReviewItem.status == "done", ReviewItem.reviewed_at >= since)
            .group_by("day", ReviewItem.source)
            .order_by("day")
        ).all()
    return [AgreementRow(r[0], r[1], int(r[2] or 0), int(r[3] or 0)) for r in rows]


def current_agreement(source: str | None = None, days: int = 7) -> float:
    since = utcnow() - timedelta(days=days)
    with session_scope() as s:
        q = select(
            func.count().label("n"),
            func.sum((ReviewItem.action == "approve").cast(__import__("sqlalchemy").Integer)),
        ).where(ReviewItem.status == "done", ReviewItem.reviewed_at >= since)
        if source:
            q = q.where(ReviewItem.source == source)
        n, n_ok = s.execute(q).one()
    n = int(n or 0)
    return (int(n_ok or 0) / n) if n else 0.0


def summary(days: int = 30) -> dict:
    rows = agreement_by_day(days)
    by_source: dict[str, dict] = {}
    for r in rows:
        b = by_source.setdefault(r.source, {"n": 0, "ok": 0})
        b["n"] += r.n_reviewed
        b["ok"] += r.n_approved_unedited
    return {
        "by_source": {
            k: {"n": v["n"], "rate": round(v["ok"] / v["n"], 4) if v["n"] else 0.0}
            for k, v in by_source.items()
        },
        "overall": round(current_agreement(days=days), 4),
        "random_only": round(current_agreement("random", days=days), 4),
        "daily": [
            {"day": r.day, "source": r.source, "n": r.n_reviewed, "rate": round(r.rate, 4)}
            for r in rows
        ],
    }
