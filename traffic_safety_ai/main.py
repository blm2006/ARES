"""Điểm chạy tương thích với cấu trúc gốc của spec (traffic_safety_ai/main.py).

Tương đương: python -m traffic_safety_ai <lệnh> ...  — xem: python -m traffic_safety_ai --help
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traffic_safety_ai.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
