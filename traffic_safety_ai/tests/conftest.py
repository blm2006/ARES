"""Test chạy offline, không cần GPU/ultralytics/video thật.

Chạy:  python -m pytest traffic_safety_ai/tests -q      (từ thư mục chứa gói)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setenv("TSA_DATA_DIR", str(d))
    return d
