"""Nạp model đồ bảo hộ (dùng chung cho camera trực tiếp và Thử video) + lưu luật theo camera do server gửi về."""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def weights_path(cfg) -> Path:
    from ares.config import REPO_ROOT

    p = Path(cfg.ppe_weights).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def load_detector(cfg):
    """-> (PPEDetector | None, thông tin trạng thái cho dashboard)."""
    path = weights_path(cfg)
    if not path.exists():
        return None, {"ready": False, "error": "", "weights": path.name,
                      "hint": "Chưa cài model đồ bảo hộ — chạy CAI_AI_BAO_HO.bat trên máy chủ."}
    try:
        from ares.behavior.pose import resolve_device
        from ares.ppe.detector import PPEDetector

        det = PPEDetector(path, resolve_device(cfg.behavior_device or cfg.device), cfg.ppe_imgsz, cfg.ppe_conf, cfg.half)
        return det, {"ready": True, "error": "", **det.describe()}
    except Exception as e:  # noqa: BLE001
        log.exception("Không nạp được model đồ bảo hộ %s", path)
        return None, {"ready": False, "error": f"{type(e).__name__}: {e}"[:200], "weights": path.name}


def rules_file(cfg) -> Path:
    return Path(cfg.behavior_dir) / "ppe_rules.json"


def load_rules(cfg) -> dict:
    from ares.ppe import clean_rules

    try:
        d = json.loads(rules_file(cfg).read_text(encoding="utf-8"))
        return {str(k): clean_rules(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_rules(cfg, rules: dict) -> None:
    f = rules_file(cfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")
