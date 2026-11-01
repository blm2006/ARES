"""Queue có persistence. Mặc định Redis Streams (spec #1).

SQLite backend có mặt để test chạy được trên máy không có Redis — nó cũng
persistent, nên không vi phạm ràng buộc "mất hết khi restart" của v1.
Payload chỉ là JSON metadata: enqueue() từ chối bất cứ thứ gì có bytes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from common.config import Settings, get_settings
from common.schemas import QueueMessage

log = logging.getLogger(__name__)


class QueueFull(RuntimeError):
    pass


class QueueBackend(ABC):
    @abstractmethod
    def enqueue(self, msg: QueueMessage) -> str: ...

    @abstractmethod
    def consume(self, count: int = 1, block_ms: int = 2000) -> list[tuple[str, QueueMessage]]: ...

    @abstractmethod
    def ack(self, entry_id: str) -> None: ...

    @abstractmethod
    def depth(self) -> int: ...

    @abstractmethod
    def close(self) -> None: ...


def _serialize(msg: QueueMessage) -> str:
    payload = msg.model_dump(mode="json")
    blob = json.dumps(payload, ensure_ascii=False)
    # Chốt chặn cứng: không bao giờ để ảnh lọt vào queue.
    if len(blob) > 256_000:
        raise ValueError(
            f"queue payload {len(blob)} bytes — quá lớn, nghi có ảnh trong payload. "
            "Queue chỉ mang path + metadata."
        )
    return blob


class RedisStreamQueue(QueueBackend):
    def __init__(self, url: str, stream: str, group: str, maxlen: int):
        import redis  # import trễ để test không cần redis-py

        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.stream = stream
        self.group = group
        self.maxlen = maxlen
        self.consumer = f"c-{uuid.uuid4().hex[:8]}"
        self._ensure_group()

    def _ensure_group(self) -> None:
        import redis

        try:
            self.r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def enqueue(self, msg: QueueMessage) -> str:
        return self.r.xadd(
            self.stream,
            {"payload": _serialize(msg)},
            maxlen=self.maxlen,
            approximate=True,
        )

    def consume(self, count: int = 1, block_ms: int = 2000) -> list[tuple[str, QueueMessage]]:
        resp = self.r.xreadgroup(
            self.group, self.consumer, {self.stream: ">"}, count=count, block=block_ms
        )
        out: list[tuple[str, QueueMessage]] = []
        for _stream, entries in resp or []:
            for entry_id, fields in entries:
                try:
                    out.append((entry_id, QueueMessage.model_validate_json(fields["payload"])))
                except Exception:
                    log.exception("payload hỏng, ack bỏ qua: %s", entry_id)
                    self.ack(entry_id)
        return out

    def ack(self, entry_id: str) -> None:
        self.r.xack(self.stream, self.group, entry_id)

    def depth(self) -> int:
        try:
            info = self.r.xpending(self.stream, self.group)
            pending = info["pending"] if isinstance(info, dict) else 0
        except Exception:
            pending = 0
        # backlog chưa giao + đang xử lý dở
        try:
            groups = self.r.xinfo_groups(self.stream)
            lag = 0
            for g in groups:
                if g["name"] == self.group:
                    lag = int(g.get("lag") or 0)
        except Exception:
            lag = 0
        return int(pending) + int(lag)

    def close(self) -> None:
        try:
            self.r.close()
        except Exception:
            pass


class SQLiteQueue(QueueBackend):
    """Persistent queue không cần service ngoài. Dùng cho test và máy 1 node nhỏ."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS q (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      payload TEXT NOT NULL,
      state TEXT NOT NULL DEFAULT 'ready',
      created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_q_state ON q(state, id);
    """

    def __init__(self, path: Path, maxlen: int = 200_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.maxlen = maxlen
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self._DDL)
        # consume() đánh dấu 'inflight', ack() mới xoá. Worker chết giữa chừng thì
        # những dòng đó KẸT MÃI MÃI: không ai ack, cũng không ai trả lại 'ready'.
        # Queue này chỉ có một consumer local, nên trả lại lúc mở là đúng và an toàn.
        n = self._conn.execute("UPDATE q SET state='ready' WHERE state='inflight'").rowcount
        self._conn.commit()
        if n:
            log.warning("trả %d message đang dang dở về hàng đợi (worker lần trước chết giữa chừng)", n)

    def enqueue(self, msg: QueueMessage) -> str:
        import time

        blob = _serialize(msg)
        with self._lock:
            if self.depth() >= self.maxlen:
                raise QueueFull(f"queue đầy ({self.maxlen})")
            cur = self._conn.execute(
                "INSERT INTO q(payload, created_at) VALUES (?,?)", (blob, time.time())
            )
            self._conn.commit()
            return str(cur.lastrowid)

    def consume(self, count: int = 1, block_ms: int = 0) -> list[tuple[str, QueueMessage]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, payload FROM q WHERE state='ready' ORDER BY id LIMIT ?", (count,)
            ).fetchall()
            if rows:
                self._conn.executemany(
                    "UPDATE q SET state='inflight' WHERE id=?", [(r[0],) for r in rows]
                )
                self._conn.commit()
        return [(str(r[0]), QueueMessage.model_validate_json(r[1])) for r in rows]

    def ack(self, entry_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM q WHERE id=?", (int(entry_id),))
            self._conn.commit()

    def depth(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM q WHERE state IN ('ready','inflight')"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_queue: QueueBackend | None = None


def get_queue(settings: Settings | None = None) -> QueueBackend:
    global _queue
    if _queue is None:
        s = settings or get_settings()
        if s.queue_backend == "redis":
            _queue = RedisStreamQueue(s.redis_url, s.queue_stream, s.queue_group, s.queue_maxlen)
        else:
            _queue = SQLiteQueue(s.data_root / "queue.db", s.queue_maxlen)
    return _queue


def reset_queue() -> None:
    """Chỉ dùng trong test."""
    global _queue
    if _queue is not None:
        _queue.close()
    _queue = None
