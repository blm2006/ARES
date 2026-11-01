"""Khi nào thì train. KHÔNG train theo lịch cứng khi không có dữ liệu mới.

Ba điều kiện (OR):
  1. >= N mẫu mới đã DUYỆT (không phải mẫu mới thu được — mẫu chưa duyệt vô giá trị)
  2. rolling mAP tụt > X%
  3. agreement rate tụt > Y%  -> dấu hiệu drift, model đang lệch khỏi dữ liệu thật
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, select

from common.config import Settings, get_settings
from common.db import ModelVersion, ReviewItem, session_scope, utcnow
from review.agreement import current_agreement


@dataclass
class TriggerDecision:
    should_train: bool
    reasons: list[str]
    n_new_approved: int
    agreement_now: float
    agreement_prev: float
    rolling_map_now: float | None
    rolling_map_prev: float | None


def count_new_approved(since) -> int:
    with session_scope() as s:
        return int(
            s.execute(
                select(func.count())
                .select_from(ReviewItem)
                .where(
                    ReviewItem.status == "done",
                    ReviewItem.action.in_(["approve", "fix"]),
                    ReviewItem.reviewed_at >= since,
                )
            ).scalar_one()
        )


def last_trained_at():
    with session_scope() as s:
        m = (
            s.execute(select(ModelVersion).order_by(ModelVersion.created_at.desc()))
            .scalars()
            .first()
        )
        return (m.created_at if m else None), (m.metrics if m else {})


def evaluate(settings: Settings | None = None) -> TriggerDecision:
    s = settings or get_settings()
    trained_at, last_metrics = last_trained_at()
    since = trained_at or (utcnow() - timedelta(days=365))

    n_new = count_new_approved(since)
    agree_now = current_agreement(days=7)
    agree_prev = current_agreement(days=28)  # cửa sổ dài hơn làm mốc so sánh
    map_prev = (last_metrics or {}).get("rolling", {}).get("map50")
    map_now = None  # điền bởi validate.py khi có model production đang chạy eval định kỳ

    reasons: list[str] = []
    if n_new >= s.train_trigger_min_new:
        reasons.append(f"{n_new} mẫu mới đã duyệt >= {s.train_trigger_min_new}")
    if (
        agree_prev > 0
        and (agree_prev - agree_now) / agree_prev * 100 > s.train_trigger_agreement_drop_pct
    ):
        reasons.append(
            f"agreement tụt {agree_prev:.3f} -> {agree_now:.3f} (> {s.train_trigger_agreement_drop_pct}%)"
        )
    if map_prev and map_now and (map_prev - map_now) / map_prev * 100 > s.train_trigger_map_drop_pct:
        reasons.append(f"rolling mAP tụt {map_prev:.3f} -> {map_now:.3f}")

    return TriggerDecision(
        should_train=bool(reasons),
        reasons=reasons,
        n_new_approved=n_new,
        agreement_now=agree_now,
        agreement_prev=agree_prev,
        rolling_map_now=map_now,
        rolling_map_prev=map_prev,
    )
