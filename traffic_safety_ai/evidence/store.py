"""CSDL SQLite (WAL): vi phạm, lần chạy, đếm lưu lượng, tốc độ, nhật ký truy cập.

Một file `traffic.db` trong thư mục dữ liệu; an toàn đa luồng (khoá + check_same_thread=False).
Không lưu biển số của xe KHÔNG vi phạm (quyền riêng tư theo Luật BVDLCN 2025).
"""

from __future__ import annotations

import json
import sqlite3
import time
import threading
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from traffic_safety_ai.lpr.vn_plate import normalize_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
  id TEXT PRIMARY KEY,
  rule_code TEXT NOT NULL, rule_name TEXT, variant TEXT, severity TEXT,
  channel TEXT, camera_id TEXT, camera_name TEXT, location TEXT,
  occurred_at TEXT, frame_idx INTEGER, video_t REAL,
  vehicle_cls TEXT, vehicle_label TEXT,
  plate_text TEXT, plate_display TEXT, plate_conf REAL, plate_color TEXT, plate_color_label TEXT,
  plate_valid INTEGER, province TEXT, plate_reads INTEGER, plate_candidates TEXT,
  behavior TEXT, legal_ref TEXT, fine TEXT, points TEXT, legal_verified INTEGER,
  ai_confidence REAL, measurements TEXT, notes TEXT,
  status TEXT DEFAULT 'cho_xac_minh', reviewer TEXT, review_note TEXT, reviewed_at TEXT,
  evidence_dir TEXT, files TEXT, run_id TEXT, source TEXT, created_at TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_v_plate ON violations(plate_text);
CREATE INDEX IF NOT EXISTS idx_v_time ON violations(occurred_at);
CREATE INDEX IF NOT EXISTS idx_v_rule ON violations(rule_code);
CREATE INDEX IF NOT EXISTS idx_v_status ON violations(status);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, source TEXT, camera_id TEXT, camera_name TEXT, location TEXT, channel TEXT,
  started_at TEXT, ended_at TEXT, frames INTEGER, fps REAL, config_hash TEXT, models TEXT,
  rules TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS counts (
  run_id TEXT, camera_id TEXT, line_id TEXT, line_name TEXT, bucket_start TEXT, cls TEXT, direction TEXT,
  n INTEGER, PRIMARY KEY (run_id, line_id, bucket_start, cls, direction)
);
CREATE TABLE IF NOT EXISTS speeds (
  run_id TEXT, camera_id TEXT, track_id INTEGER, cls TEXT, at TEXT, speed_kmh REAL, method TEXT
);
CREATE TABLE IF NOT EXISTS hazard_points (
  run_id TEXT, camera_id TEXT, rule_code TEXT, at TEXT, x REAL, y REAL
);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, actor TEXT, action TEXT, target TEXT, detail TEXT, ip TEXT
);
"""

JSON_FIELDS = ("measurements", "notes", "files", "plate_candidates")



def evidence_folder(rec: dict[str, Any], evidence_root: Path) -> Path | None:
    """Thư mục bằng chứng của hồ sơ, kể cả khi dữ liệu đã được CHUYỂN sang máy/thư mục khác.

    Đường dẫn lưu trong CSDL là đường dẫn lúc ghi (vd chạy trong máy ảo Linux rồi mở trên Windows) → không còn
    đúng thì tìm lại theo cấu trúc chuẩn <evidence>/<ngày>/<mã hồ sơ>.
    """
    vid = str(rec.get("id") or "")
    if not vid or "/" in vid or "\\" in vid or vid.startswith("."):
        return None
    d = rec.get("evidence_dir")
    if d:
        p = Path(str(d))
        if p.is_dir() and p.name == vid:       # chỉ tin đường dẫn trong CSDL khi nó đúng là thư mục của hồ sơ này
            return p
    day = str(rec.get("occurred_at") or "")[:10]
    cand = evidence_root / day / vid
    if day and cand.is_dir():
        return cand
    hits = sorted(evidence_root.glob(f"*/{vid}"))
    return hits[0] if hits else None

class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            for attempt in range(40):       # tiến trình khác (dashboard/Studio cũ) có thể đang giữ khoá trong tích tắc
                try:
                    self.conn.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError:
                    if attempt == 39:
                        break               # CSDL đã ở chế độ WAL từ trước thì vẫn dùng bình thường
                    time.sleep(0.25)
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------------ chung
    def _exec(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, tuple(args))
            self.conn.commit()
            return cur

    def _rows(self, sql: str, args: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(sql, tuple(args)).fetchall()
        return [self._decode(dict(r)) for r in rows]

    @staticmethod
    def _decode(r: dict[str, Any]) -> dict[str, Any]:
        for k in JSON_FIELDS:
            if k in r and isinstance(r[k], str) and r[k]:
                try:
                    r[k] = json.loads(r[k])
                except json.JSONDecodeError:
                    pass
        return r

    # ------------------------------------------------------------------ lần chạy
    def start_run(self, run: dict[str, Any]) -> str:
        cols = ",".join(run)
        self._exec(f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({','.join('?' * len(run))})",
                   [json.dumps(v, ensure_ascii=False) if isinstance(v, dict | list) else v for v in run.values()])
        return str(run["id"])

    def end_run(self, run_id: str, frames: int, fps: float, notes: str = "") -> None:
        self._exec("UPDATE runs SET ended_at=?, frames=?, fps=?, notes=? WHERE id=?",
                   (datetime.now().isoformat(timespec="seconds"), frames, round(fps, 2), notes, run_id))

    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))

    # ------------------------------------------------------------------ vi phạm
    def next_violation_id(self, when: datetime, camera_id: str) -> str:
        cam = "".join(ch for ch in camera_id.upper() if ch.isalnum())[:10] or "CAM"
        prefix = f"VP{when:%Y%m%d}-{cam}-"
        with self._lock:
            row = self.conn.execute("SELECT id FROM violations WHERE id LIKE ? ORDER BY id DESC LIMIT 1",
                                    (prefix + "%",)).fetchone()
        n = int(row[0].rsplit("-", 1)[-1]) + 1 if row else 1
        return f"{prefix}{n:05d}"

    def insert_violation(self, rec: dict[str, Any]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        rec = {**rec, "created_at": rec.get("created_at", now), "updated_at": now}
        vals = [json.dumps(v, ensure_ascii=False) if k in JSON_FIELDS and not isinstance(v, str) else v
                for k, v in rec.items()]
        self._exec(f"INSERT OR REPLACE INTO violations ({','.join(rec)}) VALUES ({','.join('?' * len(rec))})", vals)

    def update_violation(self, vid: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = datetime.now().isoformat(timespec="seconds")
        sets = ",".join(f"{k}=?" for k in fields)
        vals = [json.dumps(v, ensure_ascii=False) if k in JSON_FIELDS and not isinstance(v, str) else v
                for k, v in fields.items()]
        self._exec(f"UPDATE violations SET {sets} WHERE id=?", [*vals, vid])

    def get_violation(self, vid: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM violations WHERE id=?", (vid,))
        return rows[0] if rows else None

    @staticmethod
    def _violation_where(rule: str | None = None, status: str | None = None, severity: str | None = None,
                         camera: str | None = None, day: str | None = None, plate: str | None = None,
                         vehicle: str | None = None, day_from: str | None = None, day_to: str | None = None,
                         run: str | None = None, q: str | None = None) -> tuple[str, list[Any]]:
        where: list[str] = []
        args: list[Any] = []
        for col, val in (("rule_code", rule), ("status", status), ("severity", severity), ("camera_id", camera),
                         ("vehicle_cls", vehicle), ("run_id", run)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        if day:
            where.append("substr(occurred_at,1,10)=?")
            args.append(day)
        if day_from:
            where.append("substr(occurred_at,1,10)>=?")
            args.append(day_from)
        if day_to:
            where.append("substr(occurred_at,1,10)<=?")
            args.append(day_to)
        if plate:
            where.append("plate_text LIKE ?")
            args.append(f"%{normalize_text(plate)}%")
        if q:                                           # ô tìm kiếm chung: mã hồ sơ hoặc biển số
            where.append("(id LIKE ? OR plate_text LIKE ?)")
            args += [f"%{q.strip()}%", f"%{normalize_text(q) or q.strip()}%"]
        return ((" WHERE " + " AND ".join(where)) if where else ""), args

    def list_violations(self, rule: str | None = None, status: str | None = None, severity: str | None = None,
                        camera: str | None = None, day: str | None = None, plate: str | None = None,
                        limit: int = 100, offset: int = 0, **more: str | None) -> tuple[list[dict[str, Any]], int]:
        w, args = self._violation_where(rule, status, severity, camera, day, plate, **more)
        with self._lock:
            total = self.conn.execute(f"SELECT COUNT(*) FROM violations{w}", args).fetchone()[0]
        rows = self._rows(f"SELECT * FROM violations{w} ORDER BY occurred_at DESC, id DESC LIMIT ? OFFSET ?",
                          [*args, limit, offset])
        return rows, int(total)

    def violation_facets(self, **filters: str | None) -> dict[str, Any]:
        """Số hồ sơ theo trạng thái (theo bộ lọc hiện tại, trừ chính trạng thái) + các giá trị để dựng ô lọc."""
        filters.pop("status", None)
        w, args = self._violation_where(**filters)
        with self._lock:
            by_status = dict(self.conn.execute(f"SELECT status, COUNT(*) FROM violations{w} GROUP BY status", args)
                             .fetchall())
            cams = self.conn.execute("SELECT camera_id, MAX(camera_name), MAX(location) FROM violations "
                                     "GROUP BY camera_id ORDER BY camera_id").fetchall()
            vehicles = [r[0] for r in self.conn.execute("SELECT DISTINCT vehicle_cls FROM violations "
                                                        "WHERE vehicle_cls IS NOT NULL AND vehicle_cls<>''")]
            rules = self.conn.execute("SELECT rule_code, MAX(rule_name) FROM violations GROUP BY rule_code "
                                      "ORDER BY rule_code").fetchall()
            days = self.conn.execute("SELECT MIN(substr(occurred_at,1,10)), MAX(substr(occurred_at,1,10)) "
                                     "FROM violations").fetchone()
        return {"by_status": {k or "cho_xac_minh": int(v) for k, v in by_status.items()},
                "cameras": [{"id": c[0], "name": c[1], "location": c[2]} for c in cams if c[0]],
                "vehicles": vehicles, "rules": [{"code": r[0], "name": r[1]} for r in rules],
                "first_day": days[0] if days else None, "last_day": days[1] if days else None}

    def review(self, vid: str, action: str, reviewer: str, note: str = "", plate: str | None = None) -> bool:
        status = {"approve": "da_xac_minh", "reject": "loai_bo", "reset": "cho_xac_minh"}.get(action)
        if status is None or self.get_violation(vid) is None:
            return False
        fields: dict[str, Any] = {"status": status, "reviewer": reviewer, "review_note": note,
                                  "reviewed_at": datetime.now().isoformat(timespec="seconds")}
        if plate:
            from traffic_safety_ai.lpr.vn_plate import parse_plate

            rec = self.get_violation(vid) or {}
            p = parse_plate(plate, rec.get("vehicle_cls"))
            fields["plate_text"] = p.text if p else normalize_text(plate)
            fields["plate_display"] = p.display if p else plate.strip().upper()
            fields["plate_valid"] = int(bool(p and p.valid))
            fields["plate_conf"] = 1.0
            if p is not None:
                from traffic_safety_ai.lpr.vn_plate import province_label

                fields["province"] = province_label(p.province_code)
        self.update_violation(vid, **fields)
        return True

    def lookup_plate(self, plate: str, limit: int = 50) -> list[dict[str, Any]]:
        q = normalize_text(plate)
        if len(q) < 4:
            return []
        return self._rows("SELECT * FROM violations WHERE plate_text LIKE ? AND status!='loai_bo' "
                          "ORDER BY occurred_at DESC LIMIT ?", (f"%{q}%", limit))

    def stats(self, day: str | None = None, camera: str | None = None) -> dict[str, Any]:
        where, args = ["status!='loai_bo'"], []
        if day:
            where.append("substr(occurred_at,1,10)=?")
            args.append(day)
        if camera:
            where.append("camera_id=?")
            args.append(camera)
        w = " WHERE " + " AND ".join(where)
        q = lambda s: self._rows(s, args)  # noqa: E731
        return {
            "by_rule": q(f"SELECT rule_code, rule_name, severity, COUNT(*) n FROM violations{w} GROUP BY rule_code, severity ORDER BY rule_code"),
            "by_hour": q(f"SELECT substr(occurred_at,12,2) hour, COUNT(*) n FROM violations{w} GROUP BY hour ORDER BY hour"),
            "by_vehicle": q(f"SELECT vehicle_cls, COUNT(*) n FROM violations{w} GROUP BY vehicle_cls ORDER BY n DESC"),
            "by_status": self._rows("SELECT status, COUNT(*) n FROM violations GROUP BY status"),
            "total": sum(r["n"] for r in q(f"SELECT COUNT(*) n FROM violations{w}")),
        }

    # ------------------------------------------------------------------ lưu lượng / tốc độ / điểm nóng
    def add_counts(self, run_id: str, camera_id: str, rows: list[tuple[str, str, str, str, str, int]]) -> None:
        with self._lock:
            for line_id, line_name, bucket, cls, direction, n in rows:
                self.conn.execute(
                    "INSERT INTO counts VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(run_id,line_id,bucket_start,cls,direction) "
                    "DO UPDATE SET n=n+excluded.n", (run_id, camera_id, line_id, line_name, bucket, cls, direction, n))
            self.conn.commit()

    def counts(self, camera_id: str | None = None, day: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        where, args = [], []
        for col, val in (("camera_id", camera_id), ("run_id", run_id)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        if day:
            where.append("substr(bucket_start,1,10)=?")
            args.append(day)
        w = (" WHERE " + " AND ".join(where)) if where else ""
        return self._rows(f"SELECT line_id, line_name, bucket_start, cls, direction, SUM(n) n FROM counts{w} "
                          "GROUP BY line_id, bucket_start, cls, direction ORDER BY bucket_start", args)

    def add_speed(self, run_id: str, camera_id: str, track_id: int, cls: str, at: str, speed_kmh: float,
                  method: str) -> None:
        self._exec("INSERT INTO speeds VALUES (?,?,?,?,?,?,?)",
                   (run_id, camera_id, track_id, cls, at, round(speed_kmh, 2), method))

    def speeds(self, camera_id: str | None = None, day: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        where, args = [], []
        for col, val in (("camera_id", camera_id), ("run_id", run_id)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        if day:
            where.append("substr(at,1,10)=?")
            args.append(day)
        w = (" WHERE " + " AND ".join(where)) if where else ""
        return self._rows(f"SELECT * FROM speeds{w}", args)

    def add_hazard_point(self, run_id: str, camera_id: str, rule_code: str, at: str, x: float, y: float) -> None:
        self._exec("INSERT INTO hazard_points VALUES (?,?,?,?,?,?)", (run_id, camera_id, rule_code, at, x, y))

    def hazard_points(self, camera_id: str | None = None, run_id: str | None = None,
                      day: str | None = None) -> list[dict[str, Any]]:
        where, args = [], []
        for col, val in (("camera_id", camera_id), ("run_id", run_id)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        if day:
            where.append("substr(at,1,10)=?")
            args.append(day)
        w = (" WHERE " + " AND ".join(where)) if where else ""
        return self._rows(f"SELECT * FROM hazard_points{w}", args)

    # ------------------------------------------------------------------ nhật ký / lưu trữ
    def audit(self, actor: str, action: str, target: str = "", detail: str = "", ip: str = "") -> None:
        self._exec("INSERT INTO audit_log (at, actor, action, target, detail, ip) VALUES (?,?,?,?,?,?)",
                   (datetime.now().isoformat(timespec="seconds"), actor, action, target, detail, ip))

    def audit_log(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))

    def expired(self, retention_days: int) -> list[dict[str, Any]]:
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat(timespec="seconds")
        return self._rows("SELECT id, evidence_dir FROM violations WHERE occurred_at < ?", (cutoff,))

    def delete_violation(self, vid: str) -> None:
        self._exec("DELETE FROM violations WHERE id=?", (vid,))
