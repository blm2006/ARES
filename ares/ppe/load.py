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


def extra_paths(cfg) -> list[Path]:
    from ares.config import REPO_ROOT

    out = []
    for x in str(getattr(cfg, "ppe_extra_weights", "") or "").split(","):
        x = x.strip()
        if x:
            p = Path(x).expanduser()
            out.append(p if p.is_absolute() else REPO_ROOT / p)
    return out


def load_detector(cfg):
    """-> (PPEDetector | MultiPPEDetector | None, thông tin trạng thái cho dashboard).
    Nạp model chính (ARES_PPE_WEIGHTS) + các model thêm (ARES_PPE_EXTRA_WEIGHTS) đã có file; ≥ 2 model thì chạy cùng lúc."""
    path = weights_path(cfg)
    paths = [p for p in [path, *extra_paths(cfg)] if p.exists()]
    if not paths:
        return None, {"ready": False, "error": "", "weights": path.name,
                      "hint": "Chưa cài model đồ bảo hộ — chạy CAI_AI_BAO_HO.bat trên máy chủ."}
    from ares.behavior.pose import resolve_device
    from ares.ppe.detector import MultiPPEDetector, PPEDetector

    dets, errs = [], []
    for p in dict.fromkeys(paths):
        try:
            dets.append(PPEDetector(p, resolve_device(cfg.behavior_device or cfg.device), cfg.ppe_imgsz, cfg.ppe_conf, cfg.half))
        except Exception as e:  # noqa: BLE001
            log.exception("Không nạp được model đồ bảo hộ %s", p)
            errs.append(f"{p.name}: {type(e).__name__}: {e}"[:200])
    if not dets:
        return None, {"ready": False, "error": "; ".join(errs)[:300], "weights": path.name}
    det = dets[0] if len(dets) == 1 else MultiPPEDetector(dets)
    return det, {"ready": True, "error": "; ".join(errs)[:300], **det.describe()}


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
