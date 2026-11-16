"""Tra cứu căn cứ pháp lý tham khảo cho từng mã tình huống (đọc configs/rules_catalog.yaml).

Để dữ liệu pháp lý ở file YAML (không viết cứng trong code) → khi nghị định thay đổi chỉ cần sửa file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from traffic_safety_ai import CONFIG_DIR


@dataclass
class LegalRef:
    behavior: str
    ref: str
    fine: str
    points: str | None
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LegalCatalog:
    def __init__(self, path: str | Path | None = None):
        p = Path(path) if path else CONFIG_DIR / "rules_catalog.yaml"
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        self.version = str(data.get("phien_ban", ""))
        self.source = str(data.get("van_ban", ""))
        self.warning = str(data.get("canh_bao", ""))
        self.rules: dict[str, dict[str, Any]] = data.get("rules", {}) or {}

    def codes(self) -> list[str]:
        return list(self.rules)

    def name(self, code: str, default: str = "") -> str:
        return str(self.rules.get(code, {}).get("ten", default or code))

    def name_en(self, code: str) -> str:
        return str(self.rules.get(code, {}).get("ten_en", code))

    def category(self, code: str, default: str = "vi_pham") -> str:
        return str(self.rules.get(code, {}).get("nhom", default))

    def lookup(self, code: str, vehicle_cls: str | None, variant: str | None = None) -> LegalRef | None:
        entries = self.rules.get(code, {}).get("hanh_vi", []) or []
        for e in entries:
            cond = e.get("khi", {}) or {}
            classes = cond.get("loai_xe")
            if classes and vehicle_cls not in classes:
                continue
            want = cond.get("bien_the")
            if want is not None and str(want) != str(variant):
                continue
            return LegalRef(
                behavior=str(e.get("mo_ta", "")),
                ref=str(e.get("can_cu", "")),
                fine=str(e.get("muc_phat", "")),
                points=e.get("tru_diem"),
                verified=bool(e.get("da_doi_chieu", False)),
            )
        return None
