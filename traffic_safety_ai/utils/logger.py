"""Nhật ký sự kiện theo lần chạy: events.jsonl + events.csv (mở bằng Excel), an toàn đa luồng."""

from __future__ import annotations

import csv
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any

from traffic_safety_ai.core.types import ViolationEvent

FIELDS = ["vid", "rule_code", "rule_name", "severity", "variant", "wall_time", "frame_idx", "t", "vehicle_cls",
          "primary_track_id", "track_ids", "confidence", "legal_ref", "measurements"]


class EventLog:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.jsonl = self.folder / "events.jsonl"
        self.csv = self.folder / "events.csv"
        if not self.csv.exists():
            with open(self.csv, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(FIELDS)

    def write(self, ev: ViolationEvent, vid: str = "") -> None:
        row: dict[str, Any] = {
            "vid": vid, "rule_code": ev.rule_code, "rule_name": ev.rule_name, "severity": ev.severity,
            "variant": ev.variant, "wall_time": ev.wall_time.isoformat(timespec="seconds"), "frame_idx": ev.frame_idx,
            "t": round(ev.t, 3), "vehicle_cls": ev.vehicle_cls, "primary_track_id": ev.primary_track_id,
            "track_ids": ev.track_ids, "confidence": round(ev.confidence, 3), "legal_ref": ev.legal.get("ref"),
            "measurements": ev.measurements,
        }
        with self._lock:
            with open(self.jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            with open(self.csv, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([json.dumps(row[k], ensure_ascii=False, default=str)
                                        if isinstance(row[k], dict | list) else row[k] for k in FIELDS])


def setup_logging(verbose: bool = False, log_file: str | Path | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")   # console Windows: in tiếng Việt không lỗi
        except Exception:
            pass
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
                        handlers=handlers, force=True)
    for noisy in ("ultralytics", "PIL", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
