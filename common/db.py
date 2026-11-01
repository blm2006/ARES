"""Metadata store. SQLite mặc định (1 máy), đổi DB_URL sang Postgres là chạy nhiều máy.

Nguyên tắc: ảnh nằm ở storage, DB chỉ giữ path + metadata + trạng thái review.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from common.config import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Frame(Base):
    """Một frame đã được frame_writer ghi xuống storage."""

    __tablename__ = "frames"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    path: Mapped[str] = mapped_column(String(512), unique=True)
    cam_id: Mapped[str] = mapped_column(String(64), index=True)
    session_key: Mapped[str] = mapped_column(String(80), index=True)  # cam_id + ngày -> dùng để split
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    frame_idx: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(24), index=True)
    failure_reason: Mapped[str | None] = mapped_column(String(24), nullable=True)
    uncertainty_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # worker
    embedded: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    pca_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dedup_discarded: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    dedup_neighbor_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    dedup_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    in_faiss: Mapped[bool] = mapped_column(Boolean, default=False)

    # pre-annotation
    preannotated: Mapped[bool] = mapped_column(Boolean, default=False)
    preannot_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    preannot_boxes: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # governance
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    review: Mapped["ReviewItem | None"] = relationship(back_populates="frame", uselist=False)


Index("ix_frames_cam_ts", Frame.cam_id, Frame.ts)


class ReviewItem(Base):
    """Hàng chờ duyệt. Đây là CỬA DUY NHẤT vào dataset."""

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    frame_id: Mapped[str] = mapped_column(ForeignKey("frames.id"), unique=True, index=True)
    priority: Mapped[int] = mapped_column(Integer, index=True)  # nhỏ hơn = ưu tiên hơn
    source: Mapped[str] = mapped_column(String(24), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    action: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reject_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    final_boxes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    n_boxes_pre: Mapped[int] = mapped_column(Integer, default=0)
    n_boxes_final: Mapped[int] = mapped_column(Integer, default=0)
    edited: Mapped[bool] = mapped_column(Boolean, default=False)  # -> agreement_rate
    reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    frame: Mapped[Frame] = relationship(back_populates="review")


class SampleBudget(Base):
    """Quota ngày theo source — bảo đảm không nguồn nào ăn hết băng thông review."""

    __tablename__ = "sample_budget"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    day: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD
    source: Mapped[str] = mapped_column(String(24), index=True)
    cam_id: Mapped[str] = mapped_column(String(64), index=True, default="*")
    hour: Mapped[int] = mapped_column(Integer, default=-1)  # -1 = tổng ngày
    used: Mapped[int] = mapped_column(Integer, default=0)


Index("ix_budget_key", SampleBudget.day, SampleBudget.source, SampleBudget.cam_id, SampleBudget.hour, unique=True)


class Telemetry(Base):
    """Metric hạ tầng của luồng realtime — TÁCH KHỎI dataset pipeline (spec 1.7)."""

    __tablename__ = "telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    cam_id: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    role: Mapped[str] = mapped_column(String(16), default="primary")  # primary | shadow | canary
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    inference_ms: Mapped[float] = mapped_column(Float, default=0.0)
    frame_drop_rate: Mapped[float] = mapped_column(Float, default=0.0)
    queue_depth: Mapped[int] = mapped_column(Integer, default=0)
    gpu_util: Mapped[float] = mapped_column(Float, default=0.0)
    id_switch_per_min: Mapped[float] = mapped_column(Float, default=0.0)
    avg_track_length: Mapped[float] = mapped_column(Float, default=0.0)
    people_count: Mapped[float] = mapped_column(Float, default=0.0)


class DedupStat(Base):
    """Counter dedup — tăng nhanh = dataset bão hoà."""

    __tablename__ = "dedup_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    day: Mapped[str] = mapped_column(String(10), index=True)
    source: Mapped[str] = mapped_column(String(24), index=True)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[int] = mapped_column(Integer, default=0)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    path: Mapped[str] = mapped_column(String(512))
    manifest: Mapped[dict] = mapped_column(JSON)
    n_train: Mapped[int] = mapped_column(Integer, default=0)
    n_val: Mapped[int] = mapped_column(Integer, default=0)
    n_test: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    weights_path: Mapped[str] = mapped_column(String(512))
    base_checkpoint: Mapped[str] = mapped_column(String(512))
    dataset_version: Mapped[str] = mapped_column(String(32))
    pca_version: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[dict] = mapped_column(JSON)
    stage: Mapped[str] = mapped_column(String(16), default="registered", index=True)
    # registered | shadow | canary | production | rolled_back
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str] = mapped_column(String(512))
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        url = get_settings().database_url
        kwargs = {"future": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(_engine, "connect")
            def _wal(dbapi_conn, _):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()

    return _engine


def get_sessionmaker():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    sm = get_sessionmaker()
    s = sm()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_db() -> None:
    get_settings().ensure_dirs()
    Base.metadata.create_all(get_engine())


def audit(actor: str, action: str, target: str, meta: dict | None = None) -> None:
    if not get_settings().audit_log_enabled:
        return
    with session_scope() as s:
        s.add(AuditLog(actor=actor, action=action, target=target, meta=meta))
