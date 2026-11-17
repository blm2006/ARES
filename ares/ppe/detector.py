"""YOLO phát hiện đồ bảo hộ (Ultralytics). Không phụ thuộc bộ lớp cụ thể: tên lớp của model được ánh xạ về mã
món chuẩn (ITEMS) theo từ khoá, nên dùng được model SH17, Construction-PPE hoặc model tự train."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PPEDet:
    item: str  # mã món trong ITEMS
    worn: bool  # True = thấy món đó · False = model có lớp "không có ..." (no_helmet, NO-Mask...)
    box: tuple  # x1, y1, x2, y2 theo pixel ảnh gốc
    conf: float


# (từ khoá trong tên lớp đã bỏ dấu cách/gạch, mã món). Thứ tự quan trọng: "earmuf" phải xét trước "ear" (bộ phận),
# "faceguard"/"facemask" trước "face".
_KEYS = [
    ("earmuf", "ear"), ("earprotect", "ear"), ("earplug", "ear"), ("hearing", "ear"),
    ("faceguard", "goggles"), ("faceshield", "goggles"),  # kính che mặt cũng bảo vệ mắt
    ("mask", "mask"), ("respirator", "mask"),
    ("goggle", "goggles"), ("glass", "goggles"), ("eyewear", "goggles"),
    ("hardhat", "helmet"), ("helmet", "helmet"),
    ("vest", "vest"),
    ("suit", "coat"), ("coverall", "coat"), ("labcoat", "coat"), ("apron", "coat"), ("gown", "coat"),
    ("glove", "gloves"),
    ("boot", "boots"), ("shoe", "boots"),
]
_NEG = re.compile(r"^(no|non|without|missing)(?=[a-z])")


def map_class(name: str) -> tuple[str, bool] | None:
    """Tên lớp của model -> (mã món, đang đeo?) hoặc None nếu không phải đồ bảo hộ (Person, Head, Ear, Hands, Tools...)."""
    n = re.sub(r"[^a-z]", "", str(name).lower())
    neg = bool(_NEG.match(n)) and n not in ("none", "nose")
    if neg:
        n = _NEG.sub("", n, count=1)
    for key, item in _KEYS:
        if key in n:
            return item, not neg
    return None


class PPEDetector:
    def __init__(self, weights: str | Path, device: str = "auto", imgsz: int = 640, conf: float = 0.3, half: bool = True):
        from ultralytics import YOLO

        from ares.behavior.pose import resolve_device

        self.weights = str(weights)
        self.device = resolve_device(device)
        self.model = YOLO(self.weights)
        if getattr(self.model, "task", "detect") != "detect":
            raise ValueError(f"{weights} không phải model nhận diện vật thể (task={self.model.task}).")
        names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.class_map = {int(i): map_class(n) for i, n in names.items()}
        self.class_names = {int(i): str(n) for i, n in names.items()}
        self.items = sorted({m[0] for m in self.class_map.values() if m and m[1]})  # các món model nhận ra được
        if not self.items:
            raise ValueError(f"{weights}: không lớp nào là đồ bảo hộ (các lớp: {list(names.values())[:20]}).")
        self._classes = [i for i, m in self.class_map.items() if m]
        self.imgsz, self.conf = imgsz, conf
        self.half = bool(half) and self.device.startswith("cuda")
        self.fallback_reason = ""
        if self.device != "cpu":
            try:
                self.infer([np.zeros((64, 64, 3), np.uint8)])
            except RuntimeError as e:
                self._to_cpu(e)

    def _to_cpu(self, e: Exception) -> None:
        self.fallback_reason = str(e).splitlines()[0][:160]
        log.warning("YOLO đồ bảo hộ lỗi trên %s (%s) — chuyển sang CPU.", self.device, self.fallback_reason)
        self.device, self.half = "cpu", False

    def infer(self, images: list[np.ndarray]) -> list[list[PPEDet]]:
        kw = {"half": True} if self.half else {}
        args = dict(imgsz=self.imgsz, conf=self.conf, classes=self._classes, verbose=False)
        try:
            res = self.model.predict(images, device=self.device, **args, **kw)
        except RuntimeError as e:
            if self.device == "cpu" or "CUDA" not in str(e):
                raise
            self._to_cpu(e)
            res = self.model.predict(images, device="cpu", **args)
        out = []
        for r in res:
            dets: list[PPEDet] = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                for b, c, k in zip(xyxy, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                    m = self.class_map.get(int(k))
                    if m:
                        dets.append(PPEDet(m[0], m[1], tuple(float(v) for v in b), float(c)))
            out.append(dets)
        return out

    def describe(self) -> dict:
        return {"weights": Path(self.weights).name, "device": self.device, "items": self.items,
                "classes": {n: (self.class_map[i][0] if self.class_map[i] else None) for i, n in self.class_names.items()}}
