"""Tạo schema + thư mục. Chạy một lần trước khi làm gì khác."""

from __future__ import annotations

from common.config import get_settings
from common.db import init_db


def main() -> None:
    s = get_settings()
    init_db()
    print("OK")
    print("  data_root :", s.data_root)
    print("  database  :", s.database_url)
    print("  images    :", s.images_dir)
    print("  queue     :", s.queue_backend, s.redis_url if s.queue_backend == "redis" else "")
    print("  pca       :", s.pca_path, "(not yet - run scripts.fit_pca)" if not s.pca_path.exists() else "")


if __name__ == "__main__":
    main()
