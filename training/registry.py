"""Model registry — model + dataset version + pca version + metrics đầy đủ.

Rollback được về bất kỳ điểm nào. Đây là phần nhiều dự án bỏ qua rồi đến lúc cần
thì không biết model đang chạy được train từ dữ liệu nào.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from common.config import Settings, get_settings
from common.db import ModelVersion, session_scope

STAGES = ("registered", "shadow", "canary", "production", "rolled_back")


@dataclass
class RegisteredModel:
    version: str
    weights_path: str
    dataset_version: str
    pca_version: str
    metrics: dict
    stage: str


class Registry:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.root = self.s.registry_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def next_version(self) -> str:
        with session_scope() as s:
            rows = s.execute(select(ModelVersion.version)).scalars().all()
        nums = [int(v[1:]) for v in rows if v.startswith("v") and v[1:].isdigit()]
        return f"v{max(nums) + 1 if nums else 1:03d}"

    def register(
        self,
        weights: Path,
        dataset_version: str,
        metrics: dict,
        base_checkpoint: str,
        notes: str = "",
        version: str | None = None,
    ) -> RegisteredModel:
        version = version or self.next_version()
        # Mỗi model một FOLDER RIÊNG, tự đủ để sao lưu: model.pt + meta.json + data.yaml.
        vdir = self.root / version
        vdir.mkdir(parents=True, exist_ok=True)
        dst = vdir / "model.pt"
        shutil.copy2(weights, dst)
        (vdir / "meta.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "dataset_version": dataset_version,
                    "pca_version": self.s.pca_version,
                    "base_checkpoint": base_checkpoint,
                    "metrics": metrics,
                    "notes": notes,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # Kèm bản sao data.yaml của dataset để folder này tự đủ (biết được train trên lớp gì).
        try:
            dy = self.s.datasets_dir / dataset_version / "data.yaml"
            if dy.exists():
                shutil.copy2(dy, vdir / "data.yaml")
        except Exception:
            pass
        with session_scope() as s:
            s.add(
                ModelVersion(
                    version=version,
                    weights_path=str(dst),
                    base_checkpoint=base_checkpoint,
                    dataset_version=dataset_version,
                    pca_version=self.s.pca_version,
                    metrics=metrics,
                    stage="registered",
                    notes=notes,
                )
            )
        return RegisteredModel(version, str(dst), dataset_version, self.s.pca_version, metrics, "registered")

    def set_stage(self, version: str, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"stage phải thuộc {STAGES}")
        with session_scope() as s:
            m = s.execute(select(ModelVersion).where(ModelVersion.version == version)).scalar_one()
            if stage == "production":
                # Chỉ một model production tại một thời điểm.
                for other in s.execute(
                    select(ModelVersion).where(ModelVersion.stage == "production")
                ).scalars():
                    other.stage = "registered"
            m.stage = stage

    def get(self, version: str) -> ModelVersion:
        with session_scope() as s:
            return s.execute(select(ModelVersion).where(ModelVersion.version == version)).scalar_one()

    def current(self, stage: str = "production") -> ModelVersion | None:
        with session_scope() as s:
            return (
                s.execute(
                    select(ModelVersion)
                    .where(ModelVersion.stage == stage)
                    .order_by(ModelVersion.created_at.desc())
                )
                .scalars()
                .first()
            )

    def history(self, limit: int = 20) -> list[ModelVersion]:
        with session_scope() as s:
            return list(
                s.execute(
                    select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit)
                ).scalars()
            )

    def rollback_to(self, version: str, reason: str) -> ModelVersion:
        with session_scope() as s:
            cur = (
                s.execute(select(ModelVersion).where(ModelVersion.stage == "production"))
                .scalars()
                .first()
            )
            if cur is not None:
                cur.stage = "rolled_back"
                cur.notes = f"{cur.notes or ''}\nrollback: {reason}".strip()
            target = s.execute(select(ModelVersion).where(ModelVersion.version == version)).scalar_one()
            target.stage = "production"
            return target
