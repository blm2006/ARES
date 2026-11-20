"""Nạp + chạy model YOLO của các mục AI bổ sung (cháy/khói, hút thuốc)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ares.hazards import HAZARDS

log = logging.getLogger(__name__)


@dataclass
class HazardDet:
    cls: str  # tên lớp của model (fire, smoke, cigarette)
    label: str  # nhãn vẽ lên hình (LỬA, KHÓI, THUỐC LÁ)
    box: tuple  # x1, y1, x2, y2 pixel ảnh gốc
    conf: float


def weights_for(cfg, name: str) -> Path:
    """Model đã train lại từ đánh giá (nếu có) được ưu tiên hơn model tải về."""
    from ares.config import REPO_ROOT

    tuned = Path(cfg.behavior_dir) / "hazards" / name / "current.pt"
    if tuned.exists():
        return tuned
    p = Path(HAZARDS[name]["weights"])
    return p if p.is_absolute() else REPO_ROOT / p


def imgsz_for(cfg, name: str) -> int:
    return max(int(cfg.hazard_imgsz), int(HAZARDS[name].get("imgsz", 0)))


def wanted(cfg) -> list[str]:
    raw = str(getattr(cfg, "hazards", "auto") or "").strip().lower()
    if raw in ("", "off", "none", "false", "0"):
        return []
    if raw in ("auto", "all"):
        return list(HAZARDS)
    return [x.strip() for x in raw.split(",") if x.strip() in HAZARDS]


class HazardDetector:
    def __init__(self, name: str, weights: str | Path, device: str = "cpu", imgsz: int = 640, conf: float = 0.35,
                 half: bool = False):
        from ultralytics import YOLO

        self.name, self.weights = name, str(weights)
        self.device = device
        self.model = YOLO(self.weights)
        if getattr(self.model, "task", "detect") != "detect":
            raise ValueError(f"{weights} không phải model nhận diện vật thể")
        names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        labels = HAZARDS[name]["labels"]
        self.class_names = {int(i): str(n) for i, n in names.items()}
        self._keep = {i: n.lower() for i, n in self.class_names.items() if n.lower() in labels}
        if not self._keep:
            raise ValueError(f"{weights}: không có lớp nào trong {list(labels)} (model có {list(names.values())[:10]})")
        self.labels = labels
        self.imgsz, self.conf = imgsz, conf
        self.half = bool(half) and str(device).startswith("cuda")
        if self.device != "cpu":
            try:
                self.infer([np.zeros((64, 64, 3), np.uint8)])
            except RuntimeError as e:
                log.warning("Model %s lỗi trên %s (%s) — chuyển CPU", name, self.device, str(e).splitlines()[0][:120])
                self.device, self.half = "cpu", False

    def infer(self, images: list[np.ndarray]) -> list[list[HazardDet]]:
        kw = {"half": True} if self.half else {}
        res = self.model.predict(images, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                 classes=list(self._keep), verbose=False, **kw)
        out = []
        for r in res:
            dets = []
            if r.boxes is not None and len(r.boxes):
                for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                    cls = self._keep.get(int(k))
                    if cls:
                        dets.append(HazardDet(cls, self.labels[cls], tuple(float(v) for v in b), float(c)))
            out.append(dets)
        return out

    def describe(self) -> dict:
        tuned = "hazards" in Path(self.weights).parts
        return {"name": HAZARDS[self.name]["name"], "weights": Path(self.weights).name if not tuned else
                f"{self.name}/current.pt (đã train lại)", "device": self.device, "tuned": tuned,
                "classes": sorted(set(self._keep.values())), "license": HAZARDS[self.name]["license"]}


def load_hazards(cfg) -> tuple[dict, dict]:
    """-> ({mục: HazardDetector}, {mục: thông tin trạng thái}). Thiếu file / lỗi thì bỏ qua mục đó."""
    from ares.behavior.pose import resolve_device

    dets, info = {}, {}
    for name in wanted(cfg):
        p = weights_for(cfg, name)
        if not p.exists():
            info[name] = {"ready": False, "name": HAZARDS[name]["name"], "hint": "Chưa cài — chạy CAI_AI_BAO_HO.bat"}
            continue
        try:
            d = HazardDetector(name, p, resolve_device(cfg.behavior_device or cfg.device), imgsz_for(cfg, name),
                               cfg.hazard_conf, cfg.half)
            dets[name] = d
            info[name] = {"ready": True, **d.describe()}
        except Exception as e:  # noqa: BLE001
            log.exception("Không nạp được model %s", p)
            info[name] = {"ready": False, "name": HAZARDS[name]["name"], "error": f"{type(e).__name__}: {e}"[:200]}
    return dets, info
