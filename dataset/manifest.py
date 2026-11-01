"""Manifest của một dataset version.

Ghi đủ để về sau trả lời được: dataset này sinh ra từ đâu, model nào tạo
pre-annotation, ai duyệt, phân bố nguồn A/B/C ra sao, PCA version nào.

Không có manifest thì 3 tháng nữa không ai biết vì sao mAP đổi.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class Manifest:
    version: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_rev: str = field(default_factory=_git_rev)
    pca_version: str = ""
    preannot_models: dict[str, int] = field(default_factory=dict)
    reviewers: dict[str, int] = field(default_factory=dict)
    source_distribution: dict[str, int] = field(default_factory=dict)
    action_distribution: dict[str, int] = field(default_factory=dict)
    reject_reasons: dict[str, int] = field(default_factory=dict)
    cameras: dict[str, int] = field(default_factory=dict)
    sessions: dict[str, list[str]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    golden_frozen_at: str | None = None
    notes: str = ""

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def load(path: Path) -> "Manifest":
        return Manifest(**json.loads(Path(path).read_text(encoding="utf-8")))

    def diff(self, other: "Manifest") -> dict:
        """So hai version để biết cái gì đã đổi — thứ DVC/git-lfs làm tốt hơn,
        nhưng ở mức manifest thì đủ để trả lời câu hỏi thường gặp nhất."""
        out = {}
        for k in ("counts", "source_distribution", "action_distribution", "cameras"):
            a, b = getattr(self, k), getattr(other, k)
            keys = set(a) | set(b)
            out[k] = {kk: (b.get(kk, 0) - a.get(kk, 0)) for kk in sorted(keys)}
        return out
