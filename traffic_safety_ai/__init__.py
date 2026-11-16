"""Traffic Safety AI — giám sát an toàn giao thông khu vực trường học, chạy local.

Gói độc lập với yolo-al/ARES (không import chéo), có thể chép sang thư mục khác.
Chạy:  python -m traffic_safety_ai --help
"""

from __future__ import annotations

from pathlib import Path

__version__ = "1.0.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_ROOT / "configs"
MODELS_DIR = PACKAGE_ROOT / "models"
