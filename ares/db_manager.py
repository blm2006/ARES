"""Lớp lưu trữ: SQLite mặc định (đổi sang PostgreSQL bằng ARES_DB_URL).

3 bảng theo spec + 1 bảng `system_events` (ghi ngoại lệ fail-safe, lệnh điều hòa, kết quả retrain —
spec yêu cầu "log an exception" nhưng không nói log ở đâu).

Khác spec có chủ ý (đều là để chạy đúng khi có hơn một phòng / cần thao tác duyệt ảnh):
  * telemetry_logs: khóa chính (timestamp, room_id) thay vì chỉ timestamp.
  * habit_profiles: thêm room_id vào khóa chính và cột n_obs (số ngày quan sát) để biết có đủ lịch sử chưa.
  * active_learning_queue: thêm final_bboxes (bbox sau khi người sửa), reviewed_at, reject_reason, trained_in,
    width/height (kích thước ảnh gốc, để đổi bbox pixel <-> nhãn YOLO mà không phải giải mã lại ảnh).

Mọi ghi từ luồng nóng (vision, điều khiển) đi qua hàng đợi + 1 luồng ghi riêng, không chặn.
Thời gian lưu là giờ địa phương không tzinfo (đúng như một cột DATETIME thường dùng).
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from statistics import mean, median
from typing import Callable, Iterator

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, LargeBinary, String, Text, Time, create_engine, event, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from ares.clock import Clock
from ares.control_engine import HabitInfo
from ares.state import Event

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class TelemetryLog(Base):
    __tablename__ = "telemetry_logs"

    timestamp: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    room_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    occ_total: Mapped[int | None] = mapped_column(Integer, nullable=True)  # tổng số người đã làm mượt; NULL = camera mất (khác với 0 người)
    occ_zones: Mapped[list] = mapped_column(JSON, default=list)  # ví dụ [10, 12, 6, 4]
    temp_in: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_out: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity: Mapped[float | None] = mapped_column(Float, nullable=True)
    lux_window: Mapped[float | None] = mapped_column(Float, nullable=True)
    lux_zones: Mapped[list] = mapped_column(JSON, default=list)  # ví dụ [115, 120, 95, 40]
    ac_state: Mapped[dict] = mapped_column(JSON, default=dict)  # {"power": 1, "temp": 24, "fan": 3}
    led_pwm: Mapped[list] = mapped_column(JSON, default=list)  # [45, 50, 80, 15]


class HabitProfile(Base):
    __tablename__ = "habit_profiles"

    day_of_week: Mapped[int] = mapped_column(Integer, primary_key=True)  # 2 (Thứ Hai) .. 8 (Chủ nhật)
    time_slot: Mapped[time] = mapped_column(Time, primary_key=True)  # bước 15 phút
    room_id: Mapped[str] = mapped_column(String(20), primary_key=True, default="")
    occ_probability: Mapped[float] = mapped_column(Float, default=0.0)  # 0..1
    expected_headcount: Mapped[int] = mapped_column(Integer, default=0)
    avg_cooling_rate: Mapped[float | None] = mapped_column(Float, nullable=True)  # R_cool, °C/phút
    n_obs: Mapped[int] = mapped_column(Integer, default=0)


class ActiveLearningItem(Base):
    __tablename__ = "active_learning_queue"

    image_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    img_path: Mapped[str] = mapped_column(String(255))
    pred_bboxes: Mapped[list] = mapped_column(JSON, default=list)  # [{"class":0,"bbox":[x1,y1,x2,y2],"conf":0.42}]
    # CANDIDATE (chờ lọc PCA lần 1) | PENDING (chờ người duyệt) | APPROVED | REJECTED | TRAINED | DUPLICATE (bị PCA lọc trùng)
    status: Mapped[str] = mapped_column(String(20), index=True, default="PENDING")
    room_id: Mapped[str] = mapped_column(String(20), default="")
    final_bboxes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reject_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    trained_in: Mapped[str | None] = mapped_column(String(32), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    camera_id: Mapped[str | None] = mapped_column(String(30), nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)  # vector thô float16 (độc lập phiên bản PCA)
    dup_of: Mapped[str | None] = mapped_column(String(50), nullable=True)  # bị lọc trùng vì giống ảnh này
    dup_dist: Mapped[float | None] = mapped_column(Float, nullable=True)
    edited: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # người đã sửa bbox (mẫu khó -> luôn đưa vào train)
    train_pick: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # lọc PCA lần 2: True=đưa vào train, False=thừa


class SystemEvent(Base):
    __tablename__ = "system_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    room_id: Mapped[str] = mapped_column(String(20), default="")
    kind: Mapped[str] = mapped_column(String(40), index=True)
    severity: Mapped[str] = mapped_column(String(10), default="info")
    message: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[dict] = mapped_column(JSON, default=dict)


class DBManager:
    """Cũng là `HabitProvider` cho Control Engine (get / set_cooling_rate / mean_cooling_rate)."""

    def __init__(self, url: str, room_id: str, clock: Clock, habit_slot_min: int = 15,
                 habit_weeks: int = 8, habit_min_people: float = 2.0, start_writer: bool = True):
        self.url = url
        self.room_id = room_id
        self.clock = clock
        self.slot_min = habit_slot_min
        self.weeks = habit_weeks
        self.min_people = habit_min_people
        kwargs: dict = {"future": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_conn, _):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()
        self._sm = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self.engine)
        self._migrate_columns()

        self._q: queue.Queue[Callable[[Session], None]] = queue.Queue(maxsize=20000)
        self._stop = threading.Event()
        self.dropped = 0
        self.write_errors = 0
        self._thread: threading.Thread | None = None
        if start_writer:
            self.start()

    def _migrate_columns(self) -> None:
        """create_all không thêm cột vào bảng đã có: thêm các cột mới của bảng học chủ động nếu thiếu (DB cũ)."""
        from sqlalchemy import inspect, text

        insp = inspect(self.engine)
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have or col.primary_key:
                    continue
                ddl = col.type.compile(dialect=self.engine.dialect)
                with self.engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl}'))

    # ---------------------------------------------------------------- phiên
    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._sm()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ---------------------------------------------------------------- luồng ghi
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not (self._stop.is_set() and self._q.empty()):
            try:
                first = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [first]
            while len(batch) < 100:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            try:
                with self.session() as s:
                    for fn in batch:
                        fn(s)
            except Exception:
                # Một lệnh hỏng không được kéo cả mẻ: thử lại từng lệnh.
                for fn in batch:
                    try:
                        with self.session() as s:
                            fn(s)
                    except Exception:
                        self.write_errors += 1
                        log.exception("Ghi DB lỗi (bỏ qua bản ghi này)")
            finally:
                for _ in batch:
                    self._q.task_done()

    def submit(self, fn: Callable[[Session], None]) -> None:
        """Không bao giờ chặn: hàng đợi đầy thì bỏ bản ghi và đếm."""
        try:
            self._q.put_nowait(fn)
        except queue.Full:
            self.dropped += 1

    def flush(self, timeout: float = 10.0) -> bool:
        """Chờ ghi xong (dùng khi tắt máy và trong test)."""
        done = threading.Event()

        def waiter():
            self._q.join()
            done.set()

        threading.Thread(target=waiter, daemon=True).start()
        return done.wait(timeout)

    def close(self) -> None:
        self.flush(5)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.engine.dispose()

    # ---------------------------------------------------------------- telemetry
    def log_telemetry(self, row: dict) -> None:
        def _w(s: Session) -> None:
            s.merge(TelemetryLog(room_id=self.room_id, **row))
        self.submit(_w)

    def telemetry_range(self, start: datetime, end: datetime, bucket_s: int = 300) -> list[dict]:
        with self.session() as s:
            rows = s.execute(
                select(TelemetryLog).where(TelemetryLog.room_id == self.room_id,
                                           TelemetryLog.timestamp >= start, TelemetryLog.timestamp <= end)
                .order_by(TelemetryLog.timestamp)
            ).scalars().all()
        epoch = datetime(1970, 1, 1)  # phép tính trên datetime không tzinfo, không phụ thuộc múi giờ máy
        buckets: dict[int, list[TelemetryLog]] = defaultdict(list)
        for r in rows:
            buckets[int((r.timestamp - epoch).total_seconds() // bucket_s)].append(r)

        def avg(vals):
            v = [x for x in vals if x is not None]
            return round(sum(v) / len(v), 2) if v else None

        def avg_vec(vecs):
            vecs = [v for v in vecs if v]
            if not vecs:
                return []
            n = len(vecs[0])
            return [avg([v[i] for v in vecs if len(v) > i]) for i in range(n)]

        out = []
        for b in sorted(buckets):
            rs = buckets[b]
            out.append({
                "t": (epoch + timedelta(seconds=b * bucket_s)).isoformat(),
                "occ_total": avg([r.occ_total for r in rs]),
                "occ_zones": avg_vec([r.occ_zones for r in rs]),
                "temp_in": avg([r.temp_in for r in rs]),
                "temp_out": avg([r.temp_out for r in rs]),
                "humidity": avg([r.humidity for r in rs]),
                "lux_window": avg([r.lux_window for r in rs]),
                "lux_zones": avg_vec([r.lux_zones for r in rs]),
                "led_pwm": avg_vec([r.led_pwm for r in rs]),
                "ac_power": avg([(r.ac_state or {}).get("power", 0) for r in rs]),
                "ac_temp": avg([(r.ac_state or {}).get("temp") if (r.ac_state or {}).get("power") else None for r in rs]),
            })
        return out

    # ---------------------------------------------------------------- sự kiện
    def log_event(self, e: Event) -> None:
        ts = datetime.fromtimestamp(e.t, tz=self.clock.tz).replace(tzinfo=None)

        def _w(s: Session) -> None:
            s.add(SystemEvent(timestamp=ts, room_id=self.room_id, kind=e.kind, severity=e.severity,
                              message=e.message, data=_jsonable(e.data)))
        self.submit(_w)

    def recent_events(self, limit: int = 50, kinds: list[str] | None = None) -> list[dict]:
        with self.session() as s:
            q = select(SystemEvent).where(SystemEvent.room_id == self.room_id)
            if kinds:
                q = q.where(SystemEvent.kind.in_(kinds))
            rows = s.execute(q.order_by(SystemEvent.id.desc()).limit(limit)).scalars().all()
        return [{"t": r.timestamp.isoformat(), "kind": r.kind, "severity": r.severity, "message": r.message,
                 "data": r.data} for r in rows]

    def last_event_time(self, kind: str) -> datetime | None:
        with self.session() as s:
            return s.execute(select(func.max(SystemEvent.timestamp)).where(
                SystemEvent.room_id == self.room_id, SystemEvent.kind == kind)).scalar_one_or_none()

    # ---------------------------------------------------------------- thói quen (habit_profiles)
    def _slot(self, dt: datetime | time) -> time:
        m = (dt.minute // self.slot_min) * self.slot_min
        return time(dt.hour, m)

    def get(self, dow: int, slot: time) -> HabitInfo | None:
        with self.session() as s:
            r = s.get(HabitProfile, (dow, slot, self.room_id))
            if r is None:
                return None
            return HabitInfo(r.occ_probability, r.expected_headcount, r.avg_cooling_rate, r.n_obs)

    def set_cooling_rate(self, dow: int, slot: time, rate: float) -> None:
        with self.session() as s:
            r = s.get(HabitProfile, (dow, slot, self.room_id))
            if r is None:
                r = HabitProfile(day_of_week=dow, time_slot=slot, room_id=self.room_id, occ_probability=0.0,
                                 expected_headcount=0, n_obs=0)
                s.add(r)
            r.avg_cooling_rate = float(rate)

    def mean_cooling_rate(self) -> float | None:
        with self.session() as s:
            v = s.execute(select(func.avg(HabitProfile.avg_cooling_rate)).where(
                HabitProfile.room_id == self.room_id, HabitProfile.avg_cooling_rate.is_not(None))).scalar_one_or_none()
        return float(v) if v else None

    def aggregate_habits(self, now: datetime | None = None) -> dict:
        """Chạy lúc nửa đêm: gom telemetry `habit_weeks` tuần gần nhất thành xác suất có người theo (thứ, slot 15')."""
        now = now or self.clock.naive_now()
        since = now - timedelta(weeks=self.weeks)
        with self.session() as s:
            rows = s.execute(
                select(TelemetryLog.timestamp, TelemetryLog.occ_total)
                .where(TelemetryLog.room_id == self.room_id, TelemetryLog.timestamp >= since,
                       TelemetryLog.occ_total.is_not(None))  # NULL = không có dữ liệu, không được tính là phòng trống
            ).all()
        per: dict[tuple[int, time], dict[date, list[int]]] = defaultdict(lambda: defaultdict(list))
        for ts, occ in rows:
            per[(ts.isoweekday() + 1, self._slot(ts))][ts.date()].append(occ)
        min_samples = max(1, self.slot_min // 2)  # >= nửa số phút của slot có dữ liệu mới tính là "đã quan sát"
        n_upd = 0
        with self.session() as s:
            for (dow, slot), days in per.items():
                means = [mean(v) for v in days.values() if len(v) >= min_samples]
                if not means:
                    continue
                occ_days = [m for m in means if m >= self.min_people]
                prob = len(occ_days) / len(means)
                headcount = int(round(median(occ_days))) if occ_days else 0
                r = s.get(HabitProfile, (dow, slot, self.room_id))
                if r is None:
                    r = HabitProfile(day_of_week=dow, time_slot=slot, room_id=self.room_id)
                    s.add(r)
                r.occ_probability = round(prob, 4)
                r.expected_headcount = headcount
                r.n_obs = len(means)  # avg_cooling_rate được giữ nguyên
                n_upd += 1
        return {"slots_updated": n_upd, "rows_used": len(rows)}

    def habit_matrix(self) -> list[dict]:
        with self.session() as s:
            rows = s.execute(select(HabitProfile).where(HabitProfile.room_id == self.room_id)).scalars().all()
        return [{"dow": r.day_of_week, "slot": r.time_slot.strftime("%H:%M"), "p": r.occ_probability,
                 "headcount": r.expected_headcount, "r_cool": r.avg_cooling_rate, "n_obs": r.n_obs} for r in rows]

    # ---------------------------------------------------------------- dọn dẹp
    def purge_old_telemetry(self, days: int = 400) -> int:
        cutoff = self.clock.naive_now() - timedelta(days=days)
        with self.session() as s:
            res = s.execute(TelemetryLog.__table__.delete().where(TelemetryLog.timestamp < cutoff))
            return res.rowcount or 0


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple, dict)):
            try:
                import json
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)
        else:
            out[k] = str(v)
    return out
