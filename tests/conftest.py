"""Test chạy hoàn toàn offline: không Redis, không GPU, không ultralytics.

Mỗi test có data_root riêng trong tmp_path, queue backend sqlite, storage local.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "var"))
    monkeypatch.setenv("QUEUE_BACKEND", "sqlite")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("DB_URL", f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    monkeypatch.setenv("CAMERAS", "cam01=0,cam02=0")

    import common.config as config

    # Test phải độc lập với .env của máy đang chạy. Nếu không, ai sửa
    # QUOTA_UNCERTAINTY trong .env là test quota đỏ ngay, dù code không sai.
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    config.get_settings.cache_clear()

    import common.db as db
    import common.queue as q
    import common.storage as st

    db._engine = None
    db._SessionLocal = None
    st.reset_storage()
    q.reset_queue()

    db.init_db()
    yield
    q.reset_queue()


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def settings():
    from common.config import get_settings

    return get_settings()
