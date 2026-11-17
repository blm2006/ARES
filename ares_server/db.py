"""Lưu trữ của server trung tâm (sqlite thư viện chuẩn — đủ cho vài chục địa điểm; lớn hơn thì chuyển PostgreSQL).

Bảng:
  sites    địa điểm khách hàng (trường, toà nhà, phòng khám...) + khoá thiết bị (chỉ lưu băm SHA-256)
  edges    máy Edge tại địa điểm (1 địa điểm có thể nhiều máy), lần cuối liên lạc, phiên bản, trạng thái AI
  cameras  camera của từng Edge: online, số người hiện tại
  events   sự kiện hành vi gửi lên (té ngã, nằm bất động, ngồi gục, đánh nhau) + ảnh/clip nếu Edge cho phép gửi
  metrics  chuỗi số liệu theo heartbeat (số người, đèn/điều hòa nếu có tích hợp) để vẽ biểu đồ
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sites(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, address TEXT DEFAULT '', note TEXT DEFAULT '',
  key_hash TEXT NOT NULL UNIQUE, key_hint TEXT NOT NULL, created REAL NOT NULL, active INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS edges(
  site_id INTEGER NOT NULL, edge_id TEXT NOT NULL, name TEXT DEFAULT '', version TEXT DEFAULT '', ip TEXT DEFAULT '',
  first_seen REAL, last_seen REAL, info TEXT DEFAULT '{}', PRIMARY KEY(site_id, edge_id));
CREATE TABLE IF NOT EXISTS cameras(
  site_id INTEGER NOT NULL, edge_id TEXT NOT NULL, cam_id TEXT NOT NULL, name TEXT DEFAULT '', online INTEGER DEFAULT 0,
  n_people INTEGER DEFAULT 0, fps REAL DEFAULT 0, behavior INTEGER DEFAULT 0, last_seen REAL,
  PRIMARY KEY(site_id, edge_id, cam_id));
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, site_id INTEGER NOT NULL, edge_id TEXT NOT NULL, edge_event_id TEXT NOT NULL,
  t REAL NOT NULL, ts TEXT, cam_id TEXT, cam_name TEXT, kind TEXT, label TEXT, severity TEXT, message TEXT,
  source TEXT, score REAL, status TEXT DEFAULT 'new', verdict TEXT DEFAULT '', snapshot TEXT DEFAULT '',
  clip TEXT DEFAULT '', frames TEXT DEFAULT '', n_frames INTEGER DEFAULT 0, received REAL, synced INTEGER DEFAULT 1, handled_by TEXT DEFAULT '', handled_at REAL,
  UNIQUE(site_id, edge_id, edge_event_id));
CREATE INDEX IF NOT EXISTS ix_events_t ON events(t);
CREATE TABLE IF NOT EXISTS metrics(
  site_id INTEGER NOT NULL, edge_id TEXT NOT NULL, t REAL NOT NULL, people INTEGER, cams_online INTEGER,
  cams_total INTEGER, extra TEXT DEFAULT '{}');
CREATE INDEX IF NOT EXISTS ix_metrics ON metrics(site_id, t);
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE COLLATE NOCASE, name TEXT DEFAULT '',
  pw_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'viewer', status TEXT NOT NULL DEFAULT 'active',
  created REAL NOT NULL, last_login REAL, note TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS client_cams(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, site_id INTEGER NOT NULL, edge_id TEXT NOT NULL,
  cam_id TEXT NOT NULL, name TEXT NOT NULL, token TEXT NOT NULL UNIQUE, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS video_jobs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, name TEXT NOT NULL, ext TEXT NOT NULL, size INTEGER DEFAULT 0,
  status TEXT DEFAULT 'queued', progress REAL DEFAULT 0, message TEXT DEFAULT '', created REAL NOT NULL, started REAL,
  updated REAL, finished REAL, site_id INTEGER, edge_id TEXT, result TEXT DEFAULT '', n_frames INTEGER DEFAULT 0,
  has_mp4 INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS devices(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, place_id INTEGER, name TEXT NOT NULL, kind TEXT NOT NULL,
  token TEXT NOT NULL UNIQUE, created REAL NOT NULL, last_seen REAL, last_data TEXT DEFAULT '', desired TEXT DEFAULT '',
  via TEXT DEFAULT '', ip TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS sensor_readings(device_id INTEGER NOT NULL, t REAL NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_sensor_readings ON sensor_readings(device_id, t);
CREATE TABLE IF NOT EXISTS places(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, name TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS cam_places(
  site_id INTEGER NOT NULL, edge_id TEXT NOT NULL, cam_id TEXT NOT NULL, place_id INTEGER NOT NULL,
  PRIMARY KEY(site_id, edge_id, cam_id));
CREATE TABLE IF NOT EXISTS cam_metrics(
  site_id INTEGER NOT NULL, edge_id TEXT NOT NULL, cam_id TEXT NOT NULL, t REAL NOT NULL, people INTEGER);
CREATE INDEX IF NOT EXISTS ix_cam_metrics ON cam_metrics(site_id, edge_id, cam_id, t);
CREATE TABLE IF NOT EXISTS commands(
  id INTEGER PRIMARY KEY AUTOINCREMENT, site_id INTEGER NOT NULL, edge_id TEXT NOT NULL, cmd TEXT NOT NULL,
  created REAL NOT NULL, created_by TEXT DEFAULT '', delivered REAL, done REAL, ok INTEGER, message TEXT DEFAULT '');
"""


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


PBKDF2_ITER = 240_000
USERNAME_RE = r"^[A-Za-z0-9._\-]{3,32}$"
ROLES = ("admin", "viewer")  # admin: toàn quyền; viewer: xem + xử lý cảnh báo, không vào Thiết lập
STATUSES = ("active", "pending", "disabled")  # pending = tự đăng ký, chờ quản trị viên duyệt


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, PBKDF2_ITER)
    return f"pbkdf2_sha256${PBKDF2_ITER}${salt.hex()}${dk.hex()}"


def check_password(pw: str, stored: str) -> bool:
    import hmac

    try:
        algo, it, salt, dk = stored.split("$")
        assert algo == "pbkdf2_sha256"
        got = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), bytes.fromhex(salt), int(it))
        return hmac.compare_digest(got.hex(), dk)
    except Exception:  # noqa: BLE001
        return False


def password_problem(pw: str) -> str | None:
    if len(pw) < 8:
        return "Mật khẩu phải có ít nhất 8 ký tự"
    if len(pw) > 128:
        return "Mật khẩu quá dài"
    if pw.isdigit() or pw.lower() in ("12345678", "password", "matkhau1", "admin123"):
        return "Mật khẩu quá dễ đoán"
    return None


class ServerDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        with self._lock:
            self._db.executescript(SCHEMA)
            self._migrate()
            self._db.commit()

    # cột thêm ở các bản sau: DB cũ được nâng cấp tại chỗ, không mất dữ liệu
    MIGRATIONS = {"events": {"frames": "TEXT DEFAULT ''", "n_frames": "INTEGER DEFAULT 0", "synced": "INTEGER DEFAULT 1"},
                  "client_cams": {"place_id": "INTEGER"}, "places": {"ppe": "TEXT DEFAULT ''"},
                  "video_jobs": {"ppe": "TEXT DEFAULT ''"}}

    def _migrate(self) -> None:
        for table, cols in self.MIGRATIONS.items():
            have = {r[1] for r in self._db.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, decl in cols.items():
                if col not in have:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    # ------------------------------------------------------------------ tiện ích
    def _q(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _x(self, sql: str, args=()) -> int:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            # INSERT -> id dòng mới; UPDATE/DELETE -> số dòng bị ảnh hưởng (lastrowid của kết nối là của INSERT trước đó)
            return cur.lastrowid if sql.lstrip().upper().startswith("INSERT") else cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ địa điểm + khoá
    def create_site(self, name: str, address: str = "", note: str = "") -> tuple[int, str]:
        """Trả (id, khoá thiết bị). Khoá chỉ hiện MỘT lần — server chỉ giữ bản băm."""
        key = "ares_" + secrets.token_urlsafe(24)
        sid = self._x("INSERT INTO sites(name,address,note,key_hash,key_hint,created) VALUES(?,?,?,?,?,?)",
                      (name.strip(), address.strip(), note.strip(), hash_key(key), key[:9] + "…" + key[-4:], time.time()))
        return sid, key

    def rotate_key(self, site_id: int) -> str:
        key = "ares_" + secrets.token_urlsafe(24)
        if not self._x("UPDATE sites SET key_hash=?, key_hint=? WHERE id=?", (hash_key(key), key[:9] + "…" + key[-4:], site_id)):
            raise KeyError(site_id)
        return key

    def site_by_key(self, key: str) -> dict | None:
        rows = self._q("SELECT * FROM sites WHERE key_hash=? AND active=1", (hash_key(key),))
        return rows[0] if rows else None

    def sites(self) -> list[dict]:
        return self._q("SELECT id,name,address,note,key_hint,created,active FROM sites ORDER BY name")

    def site(self, site_id: int) -> dict:
        rows = self._q("SELECT id,name,address,note,key_hint,created,active FROM sites WHERE id=?", (site_id,))
        if not rows:
            raise KeyError(site_id)
        return rows[0]

    def update_site(self, site_id: int, **kw) -> None:
        allowed = {k: v for k, v in kw.items() if k in ("name", "address", "note", "active")}
        if allowed:
            self._x(f"UPDATE sites SET {','.join(f'{k}=?' for k in allowed)} WHERE id=?", [*allowed.values(), site_id])

    # ------------------------------------------------------------------ heartbeat
    def heartbeat(self, site_id: int, edge_id: str, payload: dict, ip: str) -> None:
        now = time.time()
        edge = payload.get("edge", {})
        cams = payload.get("cameras", [])
        with self._lock:
            self._db.execute(
                """INSERT INTO edges(site_id,edge_id,name,version,ip,first_seen,last_seen,info) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(site_id,edge_id) DO UPDATE SET name=excluded.name, version=excluded.version,
                   ip=excluded.ip, last_seen=excluded.last_seen, info=excluded.info""",
                (site_id, edge_id, str(edge.get("name", ""))[:80], str(edge.get("version", ""))[:40], ip, now, now,
                 json.dumps(payload, ensure_ascii=False)[:200_000]))
            for c in cams[:64]:
                self._db.execute(
                    """INSERT INTO cameras(site_id,edge_id,cam_id,name,online,n_people,fps,behavior,last_seen)
                       VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(site_id,edge_id,cam_id) DO UPDATE SET name=excluded.name,
                       online=excluded.online, n_people=excluded.n_people, fps=excluded.fps, behavior=excluded.behavior,
                       last_seen=excluded.last_seen""",
                    (site_id, edge_id, str(c.get("id", ""))[:20], str(c.get("name", ""))[:80], int(bool(c.get("online"))),
                     int(c.get("n_people") or 0), float(c.get("fps") or 0), int(bool(c.get("behavior"))), now))
            for c in cams[:64]:
                if c.get("online"):
                    self._db.execute("INSERT INTO cam_metrics(site_id,edge_id,cam_id,t,people) VALUES(?,?,?,?,?)",
                                     (site_id, edge_id, str(c.get("id", ""))[:20], now, int(c.get("n_people") or 0)))
            people = sum(int(c.get("n_people") or 0) for c in cams if c.get("online"))
            self._db.execute("INSERT INTO metrics(site_id,edge_id,t,people,cams_online,cams_total,extra) VALUES(?,?,?,?,?,?,?)",
                             (site_id, edge_id, now, people, sum(1 for c in cams if c.get("online")), len(cams),
                              json.dumps(payload.get("integrations", {}), ensure_ascii=False)[:20_000]))
            self._db.commit()

    # ------------------------------------------------------------------ sự kiện
    def add_event(self, site_id: int, edge_id: str, ev: dict) -> tuple[int, bool]:
        """Trả (id, mới?). Gửi lại cùng sự kiện (mạng chập chờn) không tạo bản trùng."""
        rows = self._q("SELECT id FROM events WHERE site_id=? AND edge_id=? AND edge_event_id=?",
                       (site_id, edge_id, ev["id"]))
        if rows:
            return rows[0]["id"], False
        eid = self._x(
            """INSERT INTO events(site_id,edge_id,edge_event_id,t,ts,cam_id,cam_name,kind,label,severity,message,source,score,
               verdict,received) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (site_id, edge_id, ev["id"], float(ev["t"]), ev.get("ts", ""), ev.get("cam_id", ""), ev.get("cam_name", ""),
             ev.get("kind", ""), ev.get("label", ""), ev.get("severity", "warn"), ev.get("message", ""),
             ev.get("source", ""), float(ev.get("score") or 0), ev.get("verdict", "") if ev.get("verdict") in ("true", "false") else "",
             time.time()))
        return eid, True

    def set_event_media(self, eid: int, snapshot: str = "", clip: str = "", frames: str = "", n_frames: int = 0) -> None:
        kw = {k: v for k, v in (("snapshot", snapshot), ("clip", clip), ("frames", frames), ("n_frames", n_frames)) if v}
        if kw:
            self._x(f"UPDATE events SET {','.join(f'{k}=?' for k in kw)} WHERE id=?", [*kw.values(), eid])

    def event(self, eid: int) -> dict:
        rows = self._q("""SELECT e.*, s.name AS site_name FROM events e JOIN sites s ON s.id=e.site_id WHERE e.id=?""", (eid,))
        if not rows:
            raise KeyError(eid)
        return rows[0]

    @staticmethod
    def _scope_sql(scope) -> tuple[str, list]:
        """scope = None (quản trị: tất cả) hoặc danh sách (site_id, edge_id, cam_id) người dùng được xem."""
        if scope is None:
            return "", []
        if not scope:
            return " AND 0", []
        parts, args = [], []
        for sid, eid, cid in scope:
            parts.append("(e.site_id=? AND e.edge_id=? AND e.cam_id=?)")
            args += [sid, eid, cid]
        return " AND (" + " OR ".join(parts) + ")", args

    def events(self, limit: int = 100, status: str = "", site_id: int | None = None, kind: str = "",
               since: float | None = None, scope=None) -> list[dict]:
        q = "SELECT e.*, s.name AS site_name FROM events e JOIN sites s ON s.id=e.site_id WHERE 1=1"
        args: list = []
        sq, sa = self._scope_sql(scope)
        q += sq
        args += sa
        if status:
            q += " AND e.status=?"
            args.append(status)
        if site_id:
            q += " AND e.site_id=?"
            args.append(site_id)
        if kind:
            q += " AND e.kind=?"
            args.append(kind)
        if since is not None:
            q += " AND e.t>=?"
            args.append(since)
        q += " ORDER BY e.t DESC LIMIT ?"
        args.append(int(limit))
        return self._q(q, args)

    def handle_event(self, eid: int, user: str, verdict: str | None = None) -> None:
        if verdict is None:
            n = self._x("UPDATE events SET status='ack', handled_by=?, handled_at=?, synced=0 WHERE id=?",
                        (user, time.time(), eid))
        else:
            n = self._x("UPDATE events SET status='ack', verdict=?, handled_by=?, handled_at=?, synced=0 WHERE id=?",
                        (verdict, user, time.time(), eid))
        if not n:
            raise KeyError(eid)

    def ack_all(self, user: str, scope=None) -> int:
        sq, sa = self._scope_sql(scope)
        return self._x("UPDATE events SET status='ack', handled_by=?, handled_at=?, synced=0 WHERE status='new' "
                       "AND id IN (SELECT e.id FROM events e WHERE 1=1" + sq + ")", [user, time.time(), *sa])

    def pop_updates(self, site_id: int, edge_id: str, limit: int = 200) -> list[dict]:
        """Đánh giá/đã xử lý làm trên dashboard server -> trả về cho đúng máy Edge (để Edge train lại bằng dữ liệu đó)."""
        with self._lock:
            rows = self._db.execute("SELECT id, edge_event_id, status, verdict FROM events WHERE site_id=? AND edge_id=? "
                                    "AND synced=0 LIMIT ?", (site_id, edge_id, limit)).fetchall()
            if rows:
                self._db.execute(f"UPDATE events SET synced=1 WHERE id IN ({','.join('?' * len(rows))})", [r["id"] for r in rows])
                self._db.commit()
        return [{"id": r["edge_event_id"], "status": r["status"], "verdict": r["verdict"]} for r in rows]

    # ------------------------------------------------------------------ tổng quan
    def edges(self, site_id: int | None = None) -> list[dict]:
        q = "SELECT site_id,edge_id,name,version,ip,first_seen,last_seen,info FROM edges"
        rows = self._q(q + (" WHERE site_id=?" if site_id else "") + " ORDER BY site_id, edge_id", (site_id,) if site_id else ())
        for r in rows:
            r["info"] = json.loads(r["info"] or "{}")
        return rows

    def cameras(self, site_id: int | None = None) -> list[dict]:
        q = "SELECT * FROM cameras" + (" WHERE site_id=?" if site_id else "") + " ORDER BY site_id, edge_id, cam_id"
        return self._q(q, (site_id,) if site_id else ())

    def series(self, site_id: int | None, since: float, bucket_s: int = 900) -> list[dict]:
        """Số người trung bình theo khung thời gian (cộng các Edge của cùng địa điểm)."""
        q = """SELECT CAST(t / ? AS INTEGER) * ? AS b, edge_id, AVG(people) AS p FROM metrics WHERE t >= ?"""
        args: list = [bucket_s, bucket_s, since]
        if site_id:
            q += " AND site_id=?"
            args.append(site_id)
        q += " GROUP BY b, site_id, edge_id ORDER BY b"
        agg: dict[int, float] = {}
        for r in self._q(q, args):
            agg[r["b"]] = agg.get(r["b"], 0.0) + float(r["p"] or 0)
        return [{"t": b, "people": round(v, 1)} for b, v in sorted(agg.items())]

    def counts_since(self, since: float, scope=None) -> dict:
        sq, sa = self._scope_sql(scope)
        rows = self._q("SELECT e.kind, COUNT(*) AS n FROM events e WHERE e.t>=?" + sq + " GROUP BY e.kind", [since, *sa])
        return {r["kind"]: r["n"] for r in rows}

    def cam_series(self, cams, since: float, bucket_s: int = 300) -> list[dict]:
        agg: dict[int, float] = {}
        for sid, eid, cid in cams:
            for r in self._q("SELECT CAST(t / ? AS INTEGER) * ? AS b, AVG(people) AS p FROM cam_metrics WHERE site_id=? "
                             "AND edge_id=? AND cam_id=? AND t>=? GROUP BY b", (bucket_s, bucket_s, sid, eid, cid, since)):
                agg[r["b"]] = agg.get(r["b"], 0.0) + float(r["p"] or 0)
        return [{"t": b, "people": round(v, 1)} for b, v in sorted(agg.items())]

    # ------------------------------------------------------------------ mạch cảm biến (ESP32/ESP8266 qua Wi-Fi, Arduino qua USB)
    def add_device(self, user_id: int, place_id: int, name: str, kind: str) -> dict:
        token = secrets.token_urlsafe(18)
        did = self._x("INSERT INTO devices(user_id,place_id,name,kind,token,created) VALUES(?,?,?,?,?,?)",
                      (user_id, place_id, name.strip()[:60], kind, token, time.time()))
        return self.device(did)

    @staticmethod
    def _dev_row(d: dict) -> dict:
        d["last_data"] = json.loads(d["last_data"]) if d.get("last_data") else {}
        d["desired"] = json.loads(d["desired"]) if d.get("desired") else {}
        return d

    def device(self, did: int) -> dict | None:
        rows = self._q("SELECT * FROM devices WHERE id=?", (did,))
        return self._dev_row(rows[0]) if rows else None

    def device_by_token(self, token: str) -> dict | None:
        if not token:
            return None
        rows = self._q("SELECT * FROM devices WHERE token=?", (token,))
        return self._dev_row(rows[0]) if rows else None

    def devices(self, user_id: int | None = None) -> list[dict]:
        rows = self._q("SELECT * FROM devices" + (" WHERE user_id=?" if user_id else "") + " ORDER BY id",
                       ((user_id,) if user_id else ()))
        return [self._dev_row(r) for r in rows]

    def device_data(self, did: int, data: dict, via: str, ip: str, history_every_s: float = 30.0) -> None:
        """Lưu số đo mới nhất; lịch sử (biểu đồ) chỉ lưu tối đa 1 bản / history_every_s giây cho đỡ phình DB."""
        now = time.time()
        with self._lock:
            self._db.execute("UPDATE devices SET last_seen=?, last_data=?, via=?, ip=? WHERE id=?",
                             (now, json.dumps(data), via, ip[:60], did))
            r = self._db.execute("SELECT MAX(t) FROM sensor_readings WHERE device_id=?", (did,)).fetchone()
            if r[0] is None or now - r[0] >= history_every_s:
                self._db.execute("INSERT INTO sensor_readings(device_id,t,data) VALUES(?,?,?)", (did, now, json.dumps(data)))
            self._db.commit()

    def device_series(self, did: int, since: float) -> list[dict]:
        return [{"t": r["t"], **json.loads(r["data"])} for r in
                self._q("SELECT t, data FROM sensor_readings WHERE device_id=? AND t>=? ORDER BY t", (did, since))]

    def set_device(self, did: int, **kw) -> None:
        if "desired" in kw and not isinstance(kw["desired"], str):
            kw["desired"] = json.dumps(kw["desired"])
        cols = ",".join(f"{k}=?" for k in kw)
        if not self._x(f"UPDATE devices SET {cols} WHERE id=?", (*kw.values(), did)):
            raise KeyError(did)

    def delete_device(self, did: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM sensor_readings WHERE device_id=?", (did,))
            self._db.execute("DELETE FROM devices WHERE id=?", (did,))
            self._db.commit()

    # ------------------------------------------------------------------ thử video (người dùng tải lên, máy AI xử lý)
    def add_video_job(self, user_id: int, name: str, ext: str, size: int, ppe: dict | None = None,
                      status: str = "queued") -> int:
        """status='uploading': file đang tải lên dở — máy AI CHƯA được nhận (chỉ nhận job 'queued')."""
        return self._x("INSERT INTO video_jobs(user_id,name,ext,size,created,updated,ppe,status,message) VALUES(?,?,?,?,?,?,?,?,?)",
                       (user_id, name[:120], ext, size, time.time(), time.time(), json.dumps(ppe) if ppe else "", status,
                        "Đang tải lên" if status == "uploading" else ""))

    def fail_stale_uploads(self, older_s: float = 0.0) -> int:
        """Video tải lên dở mà server tắt / mất mạng giữa chừng: báo lỗi rõ ràng thay vì treo mãi ở 'đang tải lên'."""
        with self._lock:
            n = self._db.execute("UPDATE video_jobs SET status='error', message='Tải lên bị ngắt giữa chừng — tải lại video', "
                                 "finished=? WHERE status='uploading' AND updated <= ?", (time.time(), time.time() - older_s)).rowcount
            self._db.commit()
        return n

    def video_job(self, jid: int) -> dict | None:
        rows = self._q("SELECT * FROM video_jobs WHERE id=?", (jid,))
        if not rows:
            return None
        j = rows[0]
        j["result"] = json.loads(j["result"]) if j["result"] else None
        return j

    def video_jobs(self, user_id: int | None = None, limit: int = 50) -> list[dict]:
        q = "SELECT * FROM video_jobs"
        rows = self._q(q + (" WHERE user_id=?" if user_id else "") + " ORDER BY id DESC LIMIT ?",
                       ((user_id, limit) if user_id else (limit,)))
        for j in rows:  # danh sách chỉ cần tóm tắt, không kèm cả chuỗi sự kiện
            r = json.loads(j.pop("result") or "null") or {}
            j["duration"], j["n_events"], j["people_max"] = r.get("duration"), len(r.get("events") or []), r.get("people_max")
        return rows

    def claim_video_job(self, site_id: int, edge_id: str, stale_s: float = 600) -> dict | None:
        """Máy AI nhận 1 video đang chờ. Job 'running' mà quá lâu không báo tiến độ (máy AI tắt giữa chừng) thì cho nhận lại."""
        now = time.time()
        with self._lock:
            self._db.execute("UPDATE video_jobs SET status='queued', message='Máy AI ngắt giữa chừng — xử lý lại' "
                             "WHERE status='running' AND updated < ?", (now - stale_s,))
            r = self._db.execute("SELECT id FROM video_jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
            if r is None:
                self._db.commit()
                return None
            self._db.execute("UPDATE video_jobs SET status='running', progress=0, started=?, updated=?, site_id=?, edge_id=?, "
                             "message='Đang xử lý' WHERE id=?", (now, now, site_id, edge_id, r[0]))
            self._db.commit()
        return self.video_job(r[0])

    def update_video_job(self, jid: int, **kw) -> None:
        kw["updated"] = time.time()
        if "result" in kw and not isinstance(kw["result"], str):
            kw["result"] = json.dumps(kw["result"], ensure_ascii=False)
        cols = ",".join(f"{k}=?" for k in kw)
        self._x(f"UPDATE video_jobs SET {cols} WHERE id=?", (*kw.values(), jid))

    def delete_video_job(self, jid: int) -> None:
        self._x("DELETE FROM video_jobs WHERE id=?", (jid,))

    # ------------------------------------------------------------------ địa điểm (mỗi người dùng 1 hoặc nhiều)
    def places(self, user_id: int | None = None) -> list[dict]:
        if user_id is None:
            return self._q("SELECT * FROM places ORDER BY user_id, id")
        return self._q("SELECT * FROM places WHERE user_id=? ORDER BY id", (user_id,))

    def place(self, pid: int) -> dict | None:
        rows = self._q("SELECT * FROM places WHERE id=?", (pid,))
        return rows[0] if rows else None

    def create_place(self, user_id: int, name: str) -> int:
        name = name.strip()[:60]
        if not name:
            raise ValueError("Nhập tên địa điểm")
        if self.user(user_id) is None:
            raise KeyError(user_id)
        return self._x("INSERT INTO places(user_id,name,created) VALUES(?,?,?)", (user_id, name, time.time()))

    def default_place(self, user_id: int) -> int:
        """Địa điểm đầu tiên của người dùng; chưa có thì tạo, đặt theo tên người đó."""
        rows = self._q("SELECT id FROM places WHERE user_id=? ORDER BY id LIMIT 1", (user_id,))
        if rows:
            return rows[0]["id"]
        u = self.user(user_id)
        return self.create_place(user_id, (u or {}).get("name") or (u or {}).get("username") or "Địa điểm")

    def set_place_ppe(self, pid: int, rules: dict) -> None:
        """Luật đồ bảo hộ của địa điểm: {"required": [...], "forbidden": [...]} (rỗng = không kiểm)."""
        on = rules.get("required") or rules.get("forbidden")
        if not self._x("UPDATE places SET ppe=? WHERE id=?", (json.dumps(rules) if on else "", pid)):
            raise KeyError(pid)

    def rename_place(self, pid: int, name: str) -> None:
        name = name.strip()[:60]
        if not name:
            raise ValueError("Nhập tên địa điểm")
        if not self._x("UPDATE places SET name=? WHERE id=?", (name, pid)):
            raise KeyError(pid)

    def delete_place(self, pid: int) -> None:
        p = self.place(pid)
        if p is None:
            raise KeyError(pid)
        others = [x for x in self.places(p["user_id"]) if x["id"] != pid]
        if not others:
            raise ValueError("Mỗi người dùng cần ít nhất một địa điểm")
        to = others[0]["id"]  # camera của địa điểm bị xoá chuyển sang địa điểm khác của cùng người dùng
        with self._lock:
            self._db.execute("UPDATE client_cams SET place_id=? WHERE place_id=?", (to, pid))
            self._db.execute("UPDATE cam_places SET place_id=? WHERE place_id=?", (to, pid))
            self._db.execute("UPDATE devices SET place_id=? WHERE place_id=?", (to, pid))
            self._db.execute("DELETE FROM places WHERE id=?", (pid,))
            self._db.commit()

    def cam_places(self) -> list[dict]:
        return self._q("SELECT * FROM cam_places")

    def assign_cam(self, site_id: int, edge_id: str, cam_id: str, place_id: int) -> None:
        if self.place(place_id) is None:
            raise KeyError(place_id)
        rows = self._q("SELECT id FROM client_cams WHERE site_id=? AND edge_id=? AND cam_id=?", (site_id, edge_id, cam_id))
        if rows:
            self._x("UPDATE client_cams SET place_id=? WHERE id=?", (place_id, rows[0]["id"]))
        else:
            self._x("INSERT INTO cam_places(site_id,edge_id,cam_id,place_id) VALUES(?,?,?,?) ON CONFLICT(site_id,edge_id,cam_id) "
                    "DO UPDATE SET place_id=excluded.place_id", (site_id, edge_id, cam_id, place_id))

    def set_user_name(self, uid: int, name: str) -> None:
        if not self._x("UPDATE users SET name=? WHERE id=?", (name.strip()[:80], uid)):
            raise KeyError(uid)

    # ------------------------------------------------------------------ camera của máy khác (người dùng app)
    def add_client_cam(self, user_id: int, site_id: int, edge_id: str, name: str, place_id: int | None = None) -> dict:
        token = secrets.token_urlsafe(18)
        place_id = place_id or self.default_place(user_id)
        cid = self._x("INSERT INTO client_cams(user_id,site_id,edge_id,cam_id,name,token,created,place_id) VALUES(?,?,?,?,?,?,?,?)",
                      (user_id, site_id, edge_id, "pending", name.strip()[:60], token, time.time(), place_id))
        self._x("UPDATE client_cams SET cam_id=? WHERE id=?", (f"k{cid}", cid))
        return self.client_cam(cid)

    def client_cam(self, cid: int) -> dict | None:
        rows = self._q("SELECT * FROM client_cams WHERE id=?", (cid,))
        return rows[0] if rows else None

    def client_cam_by_token(self, token: str) -> dict | None:
        rows = self._q("SELECT * FROM client_cams WHERE token=?", (token,))
        return rows[0] if rows else None

    def client_cams(self, user_id: int | None = None) -> list[dict]:
        if user_id is None:
            return self._q("SELECT * FROM client_cams ORDER BY user_id, id")
        return self._q("SELECT * FROM client_cams WHERE user_id=? ORDER BY id", (user_id,))

    def delete_client_cam(self, cid: int) -> None:
        self._x("DELETE FROM client_cams WHERE id=?", (cid,))

    # ------------------------------------------------------------------ tài khoản dashboard
    _USER_COLS = "id, username, name, role, status, created, last_login, note"

    def count_users(self, status: str | None = None) -> int:
        q = "SELECT COUNT(*) AS n FROM users" + (" WHERE status=?" if status else "")
        return self._q(q, (status,) if status else ())[0]["n"]

    def create_user(self, username: str, password: str, name: str = "", role: str = "viewer",
                    status: str = "active", note: str = "") -> int:
        import re

        username = username.strip()
        if not re.match(USERNAME_RE, username):
            raise ValueError("Tên đăng nhập 3–32 ký tự, chỉ gồm chữ không dấu, số, dấu chấm, gạch dưới, gạch ngang")
        if (p := password_problem(password)) is not None:
            raise ValueError(p)
        if role not in ROLES or status not in STATUSES:
            raise ValueError("Vai trò/trạng thái không hợp lệ")
        if self._q("SELECT id FROM users WHERE username=?", (username,)):
            raise ValueError("Tên đăng nhập đã có người dùng")
        uid = self._x("INSERT INTO users(username,name,pw_hash,role,status,created,note) VALUES(?,?,?,?,?,?,?)",
                      (username, name.strip()[:80], hash_password(password), role, status, time.time(), note[:200]))
        self.default_place(uid)  # mỗi tài khoản có sẵn 1 địa điểm mang tên người đó
        return uid

    def authenticate(self, username: str, password: str) -> tuple[dict | None, str]:
        """-> (user, '') nếu đúng; (None, lý do) nếu sai/chờ duyệt/bị khoá."""
        rows = self._q("SELECT * FROM users WHERE username=?", (username.strip(),))
        if not rows or not check_password(password, rows[0]["pw_hash"]):
            hash_password("x" * 8) if not rows else None  # giữ thời gian trả lời gần như nhau
            return None, "Sai tên đăng nhập hoặc mật khẩu"
        u = rows[0]
        if u["status"] == "pending":
            return None, "Tài khoản đang chờ quản trị viên duyệt"
        if u["status"] == "disabled":
            return None, "Tài khoản đã bị khoá — liên hệ quản trị viên"
        self._x("UPDATE users SET last_login=? WHERE id=?", (time.time(), u["id"]))
        return {k: u[k] for k in ("id", "username", "name", "role", "status")}, ""

    def user(self, uid: int) -> dict | None:
        rows = self._q(f"SELECT {self._USER_COLS} FROM users WHERE id=?", (uid,))
        return rows[0] if rows else None

    def users(self) -> list[dict]:
        return self._q(f"SELECT {self._USER_COLS} FROM users ORDER BY status='pending' DESC, role, username")

    def update_user(self, uid: int, **kw) -> None:
        u = self.user(uid)
        if u is None:
            raise KeyError(uid)
        kw = {k: v for k, v in kw.items() if k in ("name", "role", "status") and v is not None}
        if kw.get("role", "admin") not in ROLES or kw.get("status", "active") not in STATUSES:
            raise ValueError("Vai trò/trạng thái không hợp lệ")
        losing_admin = u["role"] == "admin" and u["status"] == "active" and \
            (kw.get("role", "admin") != "admin" or kw.get("status", "active") != "active")
        if losing_admin and self._q("SELECT COUNT(*) AS n FROM users WHERE role='admin' AND status='active'")[0]["n"] <= 1:
            raise ValueError("Phải còn ít nhất một quản trị viên đang hoạt động")
        if kw:
            self._x(f"UPDATE users SET {','.join(f'{k}=?' for k in kw)} WHERE id=?", [*kw.values(), uid])

    def set_password(self, uid: int, password: str) -> None:
        if (p := password_problem(password)) is not None:
            raise ValueError(p)
        if not self._x("UPDATE users SET pw_hash=? WHERE id=?", (hash_password(password), uid)):
            raise KeyError(uid)

    def delete_user(self, uid: int) -> None:
        u = self.user(uid)
        if u is None:
            raise KeyError(uid)
        if u["role"] == "admin" and u["status"] == "active" and \
                self._q("SELECT COUNT(*) AS n FROM users WHERE role='admin' AND status='active'")[0]["n"] <= 1:
            raise ValueError("Không xoá được quản trị viên cuối cùng")
        self._x("DELETE FROM users WHERE id=?", (uid,))
        with self._lock:
            pids = [r[0] for r in self._db.execute("SELECT id FROM places WHERE user_id=?", (uid,)).fetchall()]
            self._db.execute("DELETE FROM places WHERE user_id=?", (uid,))
            for pid in pids:
                self._db.execute("DELETE FROM cam_places WHERE place_id=?", (pid,))
            for (did,) in self._db.execute("SELECT id FROM devices WHERE user_id=?", (uid,)).fetchall():
                self._db.execute("DELETE FROM sensor_readings WHERE device_id=?", (did,))
            self._db.execute("DELETE FROM devices WHERE user_id=?", (uid,))
            self._db.commit()

    # ------------------------------------------------------------------ lệnh gửi xuống Edge (thêm/gỡ camera)
    def add_command(self, site_id: int, edge_id: str, cmd: dict, user: str) -> int:
        return self._x("INSERT INTO commands(site_id,edge_id,cmd,created,created_by) VALUES(?,?,?,?,?)",
                       (site_id, edge_id, json.dumps(cmd, ensure_ascii=False), time.time(), user))

    def pop_commands(self, site_id: int, edge_id: str) -> list[dict]:
        """Lệnh chưa giao -> trả cho Edge ở heartbeat; quá 10 phút chưa có kết quả thì giao lại."""
        now = time.time()
        with self._lock:
            rows = self._db.execute("SELECT id, cmd FROM commands WHERE site_id=? AND edge_id=? AND done IS NULL "
                                    "AND (delivered IS NULL OR delivered < ?) ORDER BY id LIMIT 10",
                                    (site_id, edge_id, now - 600)).fetchall()
            if rows:
                self._db.execute(f"UPDATE commands SET delivered=? WHERE id IN ({','.join('?' * len(rows))})",
                                 [now, *[r["id"] for r in rows]])
                self._db.commit()
        return [{"id": r["id"], **json.loads(r["cmd"])} for r in rows]

    def save_results(self, site_id: int, edge_id: str, results: list) -> None:
        with self._lock:
            for r in results[:50]:
                if isinstance(r, dict) and isinstance(r.get("id"), int):
                    self._db.execute("UPDATE commands SET done=?, ok=?, message=? WHERE id=? AND site_id=? AND edge_id=?",
                                     (time.time(), int(bool(r.get("ok"))), str(r.get("message", ""))[:300], r["id"],
                                      site_id, edge_id))
            self._db.commit()

    def commands(self, site_id: int, limit: int = 20) -> list[dict]:
        rows = self._q("SELECT * FROM commands WHERE site_id=? ORDER BY id DESC LIMIT ?", (site_id, limit))
        for r in rows:
            r["cmd"] = json.loads(r["cmd"])
        return rows

    def cleanup(self, media_root: Path, media_keep_days: int, metrics_keep_days: int) -> int:
        now = time.time()
        with self._lock:
            self._db.execute("DELETE FROM metrics WHERE t < ?", (now - metrics_keep_days * 86400,))
            self._db.execute("DELETE FROM cam_metrics WHERE t < ?", (now - metrics_keep_days * 86400,))
            self._db.execute("DELETE FROM sensor_readings WHERE t < ?", (now - metrics_keep_days * 86400,))
            n = 0
            if media_keep_days > 0:
                rows = self._db.execute("SELECT id, snapshot, clip, frames FROM events WHERE t < ? AND verdict='' "
                                        "AND (snapshot!='' OR clip!='' OR frames!='')", (now - media_keep_days * 86400,)).fetchall()
                for r in rows:
                    for rel in (r["snapshot"], r["clip"], r["frames"]):
                        if rel:
                            (media_root / rel).unlink(missing_ok=True)
                    self._db.execute("UPDATE events SET snapshot='', clip='', frames='', n_frames=0 WHERE id=?", (r["id"],))
                    n += 1
            self._db.commit()
        return n
