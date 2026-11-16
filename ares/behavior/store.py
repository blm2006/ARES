"""Lưu sự kiện hành vi + clip bằng chứng.

    <storage_root>/behavior/behavior.db                  chỉ mục sự kiện (sqlite, thư viện chuẩn)
    <storage_root>/behavior/events/<YYYYMMDD>/<id>/
        meta.json        thông tin sự kiện
        snapshot.jpg     ảnh lúc phát hiện (có khung xương)
        frames/0001.jpg  các khung của clip (dashboard phát lại bằng chuỗi ảnh — chạy được mọi trình duyệt)
        clip.mp4         cùng clip dạng video để tải về (H.264 nếu máy có, không thì mp4v)
        keypoints.npz    chuỗi khung xương của (các) người liên quan — dùng làm dữ liệu train khi đã duyệt Đúng/Sai
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)



class EventStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.root / "behavior.db"), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("""CREATE TABLE IF NOT EXISTS events(
                id TEXT PRIMARY KEY, t REAL, ts TEXT, cam_id TEXT, cam_name TEXT, kind TEXT, label TEXT,
                severity TEXT, message TEXT, track_ids TEXT, score REAL, source TEXT,
                status TEXT DEFAULT 'new', verdict TEXT DEFAULT '', dir TEXT,
                n_frames INTEGER DEFAULT 0, clip_ready INTEGER DEFAULT 0, has_mp4 INTEGER DEFAULT 0)""")
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_ev_t ON events(t)")
            cols = {r[1] for r in self._db.execute("PRAGMA table_info(events)").fetchall()}
            if "uplinked" not in cols:  # nâng cấp DB cũ: 0 = chưa gửi server, 1 = đã gửi, 3 = server từ chối
                self._db.execute("ALTER TABLE events ADD COLUMN uplinked INTEGER DEFAULT 0")
            self._db.commit()

    def add(self, row: dict) -> None:
        with self._lock:
            self._db.execute(f"INSERT INTO events({','.join(row)}) VALUES({','.join('?' * len(row))})", list(row.values()))
            self._db.commit()

    def update(self, eid: str, **kw) -> None:
        if not kw:
            return
        with self._lock:
            self._db.execute(f"UPDATE events SET {','.join(f'{k}=?' for k in kw)} WHERE id=?", [*kw.values(), eid])
            self._db.commit()

    def get(self, eid: str) -> dict:
        with self._lock:
            r = self._db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
        if r is None:
            raise KeyError(eid)
        return self._row(r)

    def list(self, limit: int = 100, status: str = "", kind: str = "", cam_id: str = "") -> list[dict]:
        q, args = "SELECT * FROM events WHERE 1=1", []
        for col, v in (("status", status), ("kind", kind), ("cam_id", cam_id)):
            if v:
                q += f" AND {col}=?"
                args.append(v)
        q += " ORDER BY t DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [self._row(r) for r in rows]

    def counts(self) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) FROM events GROUP BY status").fetchall()
        return {r[0]: r[1] for r in rows}

    def ack_all(self) -> int:
        with self._lock:
            n = self._db.execute("UPDATE events SET status='ack' WHERE status='new'").rowcount
            self._db.commit()
        return n

    def reviewed(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM events WHERE verdict IN ('true','false')").fetchall()
        return [self._row(r) for r in rows]

    def pending_uplink(self, since: float, limit: int = 5) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM events WHERE uplinked=0 AND t>=? ORDER BY t LIMIT ?",
                                    (since, int(limit))).fetchall()
        return [self._row(r) for r in rows]

    def count_pending_uplink(self, since: float) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM events WHERE uplinked=0 AND t>=?", (since,)).fetchone()[0]

    def cleanup(self, keep_days: int) -> int:
        if keep_days <= 0:
            return 0
        cut = time.time() - keep_days * 86400
        with self._lock:
            rows = self._db.execute("SELECT id, dir FROM events WHERE t < ? AND verdict = ''", (cut,)).fetchall()
            for r in rows:
                shutil.rmtree(self.root / r["dir"], ignore_errors=True)
            self._db.execute("DELETE FROM events WHERE t < ? AND verdict = ''", (cut,))
            self._db.commit()
        return len(rows)

    @staticmethod
    def _row(r) -> dict:
        d = {k: r[k] for k in r.keys()}
        d["track_ids"] = json.loads(d["track_ids"] or "[]")
        return d

    def close(self) -> None:
        with self._lock:
            self._db.close()


class ClipRecorder:
    """Giữ vài giây ảnh gần nhất cho mỗi camera; khi có sự kiện thì gom thêm vài giây sau rồi ghi đĩa ở luồng nền."""

    def __init__(self, cfg, store: EventStore):
        self.cfg, self.store = cfg, store
        self.buffers: dict[str, deque] = {}
        self.jobs: list[dict] = []
        self._last_push: dict[str, float] = {}
        self._q: queue.Queue = queue.Queue(maxsize=32)
        self._th = threading.Thread(target=self._writer, name="ares-behavior-clip", daemon=True)
        self._th.start()

    def push(self, cam_id: str, t: float, image: np.ndarray) -> None:
        cfg = self.cfg
        if t - self._last_push.get(cam_id, -1e9) < 1.0 / cfg.behavior_clip_fps - 1e-3:
            return
        self._last_push[cam_id] = t
        buf = self.buffers.setdefault(cam_id, deque())
        buf.append((t, image))
        while buf and t - buf[0][0] > cfg.behavior_clip_pre_s:
            buf.popleft()
        for job in self.jobs:
            if job["cam_id"] == cam_id and t <= job["end"]:
                job["frames"].append((t, image))
        done = [j for j in self.jobs if (j["cam_id"] == cam_id and t >= j["end"]) or t > j["end"] + 3.0]
        for j in done:
            self.jobs.remove(j)
            self._enqueue(j)

    def start(self, eid: str, cam_id: str, t: float, out_dir: Path, snapshot: np.ndarray | None, kp_payload: dict) -> None:
        frames = list(self.buffers.get(cam_id, ()))
        job = {"id": eid, "cam_id": cam_id, "t": t, "end": t + self.cfg.behavior_clip_post_s, "frames": frames,
               "dir": out_dir, "snapshot": snapshot, "kp": kp_payload}
        if self.cfg.behavior_clip_post_s <= 0:
            self._enqueue(job)
        else:
            self.jobs.append(job)

    def flush(self) -> None:
        for j in list(self.jobs):
            self.jobs.remove(j)
            self._enqueue(j)

    def _enqueue(self, job: dict) -> None:
        try:
            self._q.put_nowait(job)
        except queue.Full:
            log.warning("Hàng đợi ghi clip đầy — bỏ clip của sự kiện %s", job["id"])

    def _writer(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            try:
                self._write(job)
            except Exception:  # noqa: BLE001
                log.exception("Ghi clip sự kiện %s lỗi", job["id"])

    def _write(self, job: dict) -> None:
        import cv2

        d: Path = job["dir"]
        (d / "frames").mkdir(parents=True, exist_ok=True)
        if job["snapshot"] is not None:
            cv2.imwrite(str(d / "snapshot.jpg"), job["snapshot"], [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames = job["frames"]
        for i, (_, img) in enumerate(frames, start=1):
            cv2.imwrite(str(d / "frames" / f"{i:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        has_mp4 = False
        if frames:
            span = frames[-1][0] - frames[0][0]
            fps = max(1.0, min(30.0, (len(frames) - 1) / span)) if span > 0 else self.cfg.behavior_clip_fps
            h, w = frames[0][1].shape[:2]
            for fourcc in ("avc1", "mp4v"):
                vw = cv2.VideoWriter(str(d / "clip.mp4"), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                if vw.isOpened():
                    for _, img in frames:
                        vw.write(img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h)))
                    vw.release()
                    has_mp4 = True
                    break
                vw.release()
        kp = job["kp"]
        if kp:
            np.savez_compressed(d / "keypoints.npz", **kp)
        meta_p = d / "meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        meta.update({"n_frames": len(frames), "clip_t0": frames[0][0] if frames else None,
                     "clip_t1": frames[-1][0] if frames else None})
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.update(job["id"], n_frames=len(frames), clip_ready=1, has_mp4=int(has_mp4))

    def close(self) -> None:
        self.flush()
        self._q.put(None)
        self._th.join(timeout=10)


def event_dir_name(t: float, tz) -> str:
    return datetime.fromtimestamp(t, tz).strftime("%Y%m%d")

