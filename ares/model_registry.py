"""Sổ đăng ký model: ai là 'production', các bản ứng viên, lịch sử, quay lui.

Lưu ở models/registry.json. Ghi nguyên tử (tệp tạm + replace) để mất điện giữa chừng
không làm hỏng sổ. Vision đọc `production()` mỗi nhịp và tự nạp lại khi đổi bản.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

BASE = "base"


@dataclass
class ModelVersion:
    version: str
    path: str
    parent: str
    created_at: str
    trigger: str = ""
    status: str = "candidate"  # candidate | production | rejected | archived
    n_new: int = 0
    n_total: int = 0
    n_replay: int = 0
    metrics: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    notes: str = ""


class ModelRegistry:
    def __init__(self, models_dir: Path, base_weights: Path):
        self.dir = Path(models_dir)
        self.base_weights = Path(base_weights)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "registry.json"
        self._lock = threading.RLock()
        self._mtime = -1.0
        self._data: dict = {"production": BASE, "versions": [], "history": [BASE]}
        self._load()

    # ---------------------------------------------------------------- I/O
    def _load(self) -> None:
        with self._lock:
            if not self.file.exists():
                return
            try:
                m = self.file.stat().st_mtime
                if m == self._mtime:
                    return
                self._data = json.loads(self.file.read_text(encoding="utf-8"))
                self._mtime = m
                if "history" not in self._data:  # registry.json tạo bởi bản trước khi có lịch sử
                    cur = self._data.get("production", BASE)
                    self._data["history"] = [BASE] if cur == BASE else [BASE, cur]
            except Exception:
                log.exception("Không đọc được registry.json — giữ bản trong bộ nhớ")

    def _save(self) -> None:
        tmp = self.file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.file)
        self._mtime = self.file.stat().st_mtime

    # ---------------------------------------------------------------- truy vấn
    def production(self) -> tuple[str, Path]:
        """(version, đường dẫn weights) đang chạy. Nếu file production biến mất thì lùi về base."""
        self._load()
        with self._lock:
            v = self._data.get("production", BASE)
            if v == BASE:
                return BASE, self.base_weights
            for mv in self._data["versions"]:
                if mv["version"] == v and Path(mv["path"]).exists():
                    return v, Path(mv["path"])
            log.error("Model production %s mất file — lùi về base", v)
            return BASE, self.base_weights

    def versions(self) -> list[dict]:
        self._load()
        with self._lock:
            out = [dict(v) for v in self._data["versions"]]
        base = {"version": BASE, "path": str(self.base_weights), "parent": "", "created_at": "", "status": "archived",
                "trigger": "", "n_new": 0, "n_total": 0, "n_replay": 0, "metrics": {}, "gate": {}, "notes": "Weights gốc"}
        prod = self.production()[0]
        for v in out:
            if v["version"] == prod:
                v["status"] = "production"
        if prod == BASE:
            base["status"] = "production"
        return [base, *out][::-1]

    def get(self, version: str) -> dict | None:
        for v in self.versions():
            if v["version"] == version:
                return v
        return None

    def next_version(self) -> str:
        self._load()
        with self._lock:
            n = max([int(v["version"][1:]) for v in self._data["versions"] if v["version"][1:].isdigit()] or [0])
        return f"v{n + 1:04d}"

    # ---------------------------------------------------------------- thay đổi
    def add(self, weights_src: Path, parent: str, trigger: str, n_new: int, n_total: int, n_replay: int,
            metrics: dict, gate: dict, status: str = "candidate", notes: str = "") -> ModelVersion:
        with self._lock:
            self._load()
            ver = self.next_version()
            dst = self.dir / f"{ver}.pt"
            if Path(weights_src).resolve() != dst.resolve():
                dst.write_bytes(Path(weights_src).read_bytes())
            mv = ModelVersion(ver, str(dst), parent, datetime.now().isoformat(timespec="seconds"), trigger, status,
                              n_new, n_total, n_replay, metrics, gate, notes)
            self._data["versions"].append(asdict(mv))
            self._save()
            return mv

    def promote(self, version: str) -> None:
        with self._lock:
            self._load()
            self._set_production(version, record=True)

    def _set_production(self, version: str, record: bool) -> None:
        if version != BASE:
            row = next((v for v in self._data["versions"] if v["version"] == version), None)
            if row is None:
                raise KeyError(version)
            if not Path(row["path"]).exists():
                raise FileNotFoundError(row["path"])
        old = self._data.get("production", BASE)
        for v in self._data["versions"]:
            if v["version"] == old and v["version"] != version:
                v["status"] = "archived"
            if v["version"] == version:
                v["status"] = "production"
        self._data["production"] = version
        hist = self._data.setdefault("history", [BASE])
        if record and (not hist or hist[-1] != version):
            hist.append(version)
        self._save()
        log.info("Model production: %s -> %s", old, version)

    def rollback(self) -> str | None:
        """Quay lui về phiên bản production liền trước (theo lịch sử promote). Gọi lại thì lùi tiếp. None nếu hết lịch sử."""
        with self._lock:
            self._load()
            hist = self._data.setdefault("history", [BASE])
            while len(hist) >= 2:
                hist.pop()
                prev = hist[-1]
                try:
                    self._set_production(prev, record=False)
                    return prev
                except (KeyError, FileNotFoundError):
                    log.warning("Bỏ qua %s khi quay lui (không còn trong registry/mất file)", prev)
            return None

    def set_status(self, version: str, status: str) -> None:
        with self._lock:
            self._load()
            for v in self._data["versions"]:
                if v["version"] == version:
                    v["status"] = status
            self._save()

    def previous_production(self) -> str | None:
        """Phiên bản mà rollback() sẽ quay về (None nếu không còn gì để lùi)."""
        with self._lock:
            self._load()
            hist = self._data.get("history", [BASE])
            return hist[-2] if len(hist) >= 2 else None
