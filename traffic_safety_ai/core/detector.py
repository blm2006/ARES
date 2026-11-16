"""Bọc Ultralytics YOLO + quản lý "năng lực" (capability) của bộ model.

Thực tế quan trọng: KHÔNG có model công khai nào có đủ các lớp spec gốc liệt kê
(helmet/no_helmet/phone/cargo_box/turn_signal...). Vì vậy:
  * model nền (COCO) lo person/bicycle/car/motorcycle/bus/truck/traffic light/cell phone;
  * các model chuyên biệt (mũ bảo hiểm, biển số, ký tự biển số, hàng cồng kềnh, phân vùng làn) là tuỳ chọn,
    chạy trên ẢNH CẮT của đối tượng đã track (hai tầng) → bắt được vật nhỏ ở góc cao tốt hơn chạy cả khung;
  * rule nào thiếu năng lực sẽ tự TẮT và báo lý do — không bao giờ "đoán bừa" ra vi phạm.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from traffic_safety_ai.core.config import ModelsConfig, ModelSpec, resolve_weights_path
from traffic_safety_ai.core.types import Detection

log = logging.getLogger(__name__)

COCO_BASE_MAP = {
    "person": "person", "bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle", "bus": "bus",
    "truck": "truck", "traffic light": "traffic_light", "cell phone": "phone",
}


@dataclass
class ModelOutput:
    boxes: list[Detection] = field(default_factory=list)
    keypoints: np.ndarray | None = None          # (N, K, 3) cùng thứ tự với boxes
    masks_xy: list[np.ndarray] | None = None     # đa giác (toạ độ ảnh) cùng thứ tự với boxes
    probs: dict[str, float] | None = None        # model phân loại


def pick_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def precision_kwargs(want_half: bool, device: str) -> dict[str, Any]:
    """Ultralytics ≥ 8.4 dùng `quantize=16`, bản cũ dùng `half=True`."""
    if not want_half or not str(device).startswith("cuda"):
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
    except Exception:
        return {"half": True}


class ModelLoadError(RuntimeError):
    pass


class YoloModel:
    def __init__(self, name: str, spec: ModelSpec, device: str = "auto", half: bool = True):
        self.name = name
        self.spec = spec
        self.device = pick_device(device)
        self._prec = precision_kwargs(half, self.device)
        self.model = None
        self.weights_used: str | None = None
        errors: list[str] = []
        try:
            from ultralytics import YOLO
        except Exception as e:  # pragma: no cover - phụ thuộc môi trường
            raise ModelLoadError(f"chưa cài ultralytics ({e})") from e
        for w in [spec.weights, *spec.fallback]:
            path = resolve_weights_path(w)
            if path is None:
                continue
            if ("/" in path or "\\" in path) and not Path(path).exists():
                errors.append(f"không thấy file {path}")
                continue
            try:
                self.model = YOLO(path)
                self.weights_used = path
                break
            except Exception as e:
                errors.append(f"{path}: {e}")
        if self.model is None:
            raise ModelLoadError("; ".join(errors) or "chưa khai báo weights")
        self.task = getattr(self.model, "task", "detect")
        names = getattr(self.model, "names", {}) or {}
        self.names: dict[int, str] = {int(k): str(v) for k, v in (names.items() if isinstance(names, dict)
                                                                   else enumerate(names))}
        cmap = spec.classes
        if cmap is None and name == "base":
            cmap = COCO_BASE_MAP
        self.class_map: dict[int, str] = {}
        for i, n in self.names.items():
            if cmap is None:
                self.class_map[i] = n
            elif n in cmap:
                self.class_map[i] = cmap[n]
        self._warmup()

    # ------------------------------------------------------------------
    def has_class(self, canonical: str) -> bool:
        return canonical in self.class_map.values()

    def ids_for(self, canonicals: list[str] | None) -> list[int] | None:
        if canonicals is None:
            return sorted(self.class_map) if self.spec.classes is not None or self.name == "base" else None
        return [i for i, c in self.class_map.items() if c in canonicals]

    def _warmup(self) -> None:
        dummy = np.zeros((max(64, self.spec.imgsz), max(64, self.spec.imgsz), 3), np.uint8)
        try:
            self._raw([dummy], self.spec.imgsz, self.spec.conf, None)
        except Exception as e:
            if self.device != "cpu":
                log.warning("Model %s lỗi trên %s (%s) — chuyển CPU", self.name, self.device, e)
                self.device, self._prec = "cpu", {}
                self._raw([dummy], self.spec.imgsz, self.spec.conf, None)
            else:
                raise

    def _raw(self, images: list[np.ndarray], imgsz: int, conf: float, classes: list[int] | None):
        kw: dict[str, Any] = dict(imgsz=imgsz, conf=conf, iou=self.spec.iou, device=self.device, verbose=False,
                                  **self._prec)
        if classes is not None and self.task != "classify":
            kw["classes"] = classes
        return self.model.predict(images, **kw)

    def predict(self, images: list[np.ndarray], imgsz: int | None = None, conf: float | None = None,
                classes: list[str] | None = None) -> list[ModelOutput]:
        if not images:
            return []
        t0 = time.perf_counter()
        ids = self.ids_for(classes)
        results = self._raw(images, imgsz or self.spec.imgsz, self.spec.conf if conf is None else conf, ids)
        outs = [self._parse(r) for r in results]
        if self.name == "base":
            outs = [merge_cross_class(o) for o in outs]
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return outs

    def _parse(self, r: Any) -> ModelOutput:
        out = ModelOutput()
        if getattr(r, "probs", None) is not None and self.task == "classify":
            data = r.probs.data.cpu().numpy() if hasattr(r.probs.data, "cpu") else np.asarray(r.probs.data)
            out.probs = {self.class_map.get(i, self.names.get(i, str(i))): float(p) for i, p in enumerate(data)}
            return out
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = _np(boxes.xyxy)
        conf = _np(boxes.conf)
        cls = _np(boxes.cls).astype(int)
        keep = [i for i, c in enumerate(cls) if int(c) in self.class_map]
        for i in keep:
            out.boxes.append(Detection(xyxy[i], float(conf[i]), self.class_map[int(cls[i])], self.name))
        kp = getattr(r, "keypoints", None)
        if kp is not None and getattr(kp, "data", None) is not None and len(keep):
            data = _np(kp.data)
            if data.ndim == 3:
                out.keypoints = data[keep]
        masks = getattr(r, "masks", None)
        if masks is not None and len(keep):
            try:
                polys = masks.xy
                out.masks_xy = [np.asarray(polys[i], dtype=np.float64) for i in keep]
            except Exception:
                out.masks_xy = None
        return out


# Model COCO hay cho CÙNG một xe hai hộp khác lớp (bán tải = "car" và "truck", xe khách nhỏ = "car" và "bus").
# NMS của YOLO làm theo từng lớp nên không gộp được → tracker tạo 2 ID → một xe bị lập 2 hồ sơ.
CROSS_CLASS_GROUPS: tuple[frozenset[str], ...] = (frozenset({"car", "truck", "bus"}),
                                                   frozenset({"motorcycle", "bicycle"}))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_cross_class(out: "ModelOutput", iou_thr: float = 0.7) -> "ModelOutput":
    """Gộp hộp trùng nhau giữa các lớp cùng nhóm (giữ hộp điểm cao hơn). Không đụng tới keypoints/masks."""
    if len(out.boxes) < 2 or out.keypoints is not None or out.masks_xy is not None:
        return out
    order = sorted(range(len(out.boxes)), key=lambda i: -out.boxes[i].conf)
    kept: list[int] = []
    for i in order:
        d = out.boxes[i]
        grp = next((g for g in CROSS_CLASS_GROUPS if d.cls in g), None)
        dup = grp is not None and any(
            out.boxes[k].cls in grp and out.boxes[k].cls != d.cls and _iou(out.boxes[k].xyxy, d.xyxy) >= iou_thr
            for k in kept)
        if not dup:
            kept.append(i)
    out.boxes = [out.boxes[i] for i in sorted(kept)]
    return out


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"]


class DnnYoloModel:
    """Model YOLO dạng ONNX (xuất từ Ultralytics) chạy bằng OpenCV DNN — KHÔNG cần PyTorch/Ultralytics.

    Dùng cho máy không cài được PyTorch (máy ảo, mini PC ở cổng trường) hoặc khi muốn gói chạy thật nhẹ.
    Chỉ hỗ trợ bài toán detect; ONNX phải xuất với kích thước cố định (imgsz) — vd:
        python -m traffic_safety_ai export-models --which base --format onnx --imgsz 960
    Tên lớp: đọc tệp `<model>.names.txt` (mỗi dòng một tên) nếu có, không có mà model 80 lớp thì coi là COCO.
    """

    task = "detect"
    fixed_size = True              # ONNX kích thước cố định → không dùng để chạy hàng loạt ảnh cắt nhỏ

    def __init__(self, name: str, spec: ModelSpec, path: str):
        import cv2

        self.name = name
        self.spec = spec
        self.device = "cpu (OpenCV DNN)"
        self.weights_used = path
        self.net = cv2.dnn.readNetFromONNX(path)
        self.imgsz = int(spec.imgsz)
        names_file = Path(path).with_suffix(".names.txt")
        names: list[str] | None = None
        if names_file.exists():
            names = [x.strip() for x in names_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        out = self._forward(np.zeros((self.imgsz, self.imgsz, 3), np.uint8))
        nc = out.shape[0] - 4
        if names is None:
            names = COCO80 if nc == 80 else [str(i) for i in range(nc)]
        if len(names) != nc:
            raise ModelLoadError(f"{path}: model có {nc} lớp nhưng tệp tên lớp có {len(names)} dòng")
        self.names = dict(enumerate(names))
        cmap = spec.classes if spec.classes is not None else (COCO_BASE_MAP if name == "base" else None)
        self.class_map = {i: (n if cmap is None else cmap[n]) for i, n in self.names.items()
                          if cmap is None or n in cmap}
        self.last_ms = 0.0

    has_class = YoloModel.has_class
    ids_for = YoloModel.ids_for

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        import cv2

        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        px, py = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, px, py

    def _forward(self, square: np.ndarray) -> np.ndarray:
        import cv2

        blob = cv2.dnn.blobFromImage(square, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        return np.asarray(out)[0]          # (4 + số lớp, số ô dự đoán)

    def predict(self, images: list[np.ndarray], imgsz: int | None = None, conf: float | None = None,
                classes: list[str] | None = None) -> list[ModelOutput]:
        import cv2

        t0 = time.perf_counter()
        thr = self.spec.conf if conf is None else conf
        ids = self.ids_for(classes)
        allowed = set(ids) if ids is not None else None
        outs: list[ModelOutput] = []
        for img in images:
            sq, r, px, py = self._letterbox(img)
            pred = self._forward(sq).T                       # (N, 4 + nc)
            scores = pred[:, 4:]
            cls = scores.argmax(axis=1)
            sc = scores[np.arange(len(cls)), cls]
            keep = sc >= thr
            if allowed is not None:
                keep &= np.isin(cls, list(allowed))
            pred, cls, sc = pred[keep], cls[keep], sc[keep]
            o = ModelOutput()
            if len(sc):
                cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
                x1 = (cx - bw / 2 - px) / r
                y1 = (cy - bh / 2 - py) / r
                ww, hh = bw / r, bh / r
                # NMS theo từng lớp: dời hộp mỗi lớp ra xa nhau rồi NMS một lần
                off = cls.astype(np.float64) * 10000.0
                rects = np.stack([x1 + off, y1 + off, ww, hh], axis=1).tolist()
                idx = cv2.dnn.NMSBoxes(rects, sc.astype(float).tolist(), thr, float(self.spec.iou))
                H, W = img.shape[:2]
                for i in np.asarray(idx).reshape(-1):
                    c = int(cls[i])
                    if c not in self.class_map:
                        continue
                    box = np.array([max(0.0, x1[i]), max(0.0, y1[i]), min(W - 1.0, x1[i] + ww[i]),
                                    min(H - 1.0, y1[i] + hh[i])])
                    o.boxes.append(Detection(box, float(sc[i]), self.class_map[c], self.name))
            outs.append(merge_cross_class(o) if self.name == "base" else o)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return outs


def load_model(name: str, spec: ModelSpec, device: str = "auto", half: bool = True) -> YoloModel | DnnYoloModel:
    """Chọn backend: Ultralytics (PyTorch) nếu có; tệp .onnx khi không có PyTorch (hoặc spec.backend: opencv)."""
    want_dnn = (spec.backend or "") == "opencv"
    try:
        import ultralytics  # noqa: F401
        have_ul = True
    except Exception:
        have_ul = False
    errors: list[str] = []
    if want_dnn or not have_ul:
        for w in [spec.weights, *spec.fallback]:
            path = resolve_weights_path(w)
            if path and path.lower().endswith(".onnx") and Path(path).exists():
                try:
                    return DnnYoloModel(name, spec, path)
                except Exception as e:
                    errors.append(f"{path}: {e}")
        if not have_ul:
            raise ModelLoadError("; ".join(errors) or "chưa cài ultralytics/PyTorch và không có tệp .onnx dự phòng")
    return YoloModel(name, spec, device, half)


# ==================================================================== kho model
MODEL_NAMES = ("base", "pose", "helmet", "phone", "plate", "plate_ocr", "cargo", "lane_seg", "orientation",
               "turn_signal")


class ModelHub:
    """Nạp lười các model cần thiết và báo năng lực hiện có."""

    def __init__(self, cfg: ModelsConfig):
        self.cfg = cfg
        self.models: dict[str, YoloModel | DnnYoloModel] = {}
        self.status: dict[str, str] = {}

    def load(self, names: set[str]) -> None:
        for n in names:
            if n in self.models or n not in MODEL_NAMES:
                continue
            spec: ModelSpec = getattr(self.cfg, n)
            if not spec.enabled:
                self.status[n] = "đã tắt trong models.yaml"
                continue
            if n == "plate_ocr" and (spec.backend or "yolo_chars") != "yolo_chars":
                self.status[n] = f"dùng backend {spec.backend}"
                continue
            if not spec.weights and not spec.fallback:
                self.status[n] = "chưa khai báo weights"
                continue
            try:
                m = load_model(n, spec, self.cfg.device, self.cfg.half)
                self.models[n] = m
                self.status[n] = f"OK ({Path(m.weights_used or '').name}, {m.device})"
                log.info("Nạp model %s: %s", n, self.status[n])
            except ModelLoadError as e:
                self.status[n] = f"không nạp được: {e}"
                log.warning("Model %s: %s", n, self.status[n])

    def get(self, name: str) -> YoloModel | DnnYoloModel | None:
        return self.models.get(name)

    def capabilities(self) -> dict[str, tuple[bool, str]]:
        base = self.get("base")
        caps: dict[str, tuple[bool, str]] = {}
        caps["vehicles"] = (base is not None, self.status.get("base", "chưa nạp"))
        caps["persons"] = (base is not None and base.has_class("person"), "model nền (COCO person)")
        caps["traffic_light_det"] = (base is not None and base.has_class("traffic_light"), "model nền (COCO traffic light)")
        ph = self.get("phone")
        base_crops = base is not None and base.has_class("phone") and not getattr(base, "fixed_size", False)
        caps["phone"] = (ph is not None or base_crops,
                         "model điện thoại riêng" if ph else ("model nền (COCO cell phone) trên ảnh cắt" if base_crops
                                                            else "model nền ONNX kích thước cố định — không chạy ảnh cắt"))
        caps["pose"] = (self.get("pose") is not None, self.status.get("pose", "chưa nạp"))
        caps["helmet"] = (self.get("helmet") is not None, self.status.get("helmet", "chưa nạp"))
        caps["plate_detect"] = (True, self.status.get("plate", "") if self.get("plate") else
                                "dùng bộ dò cổ điển (độ chính xác thấp) — nên huấn luyện models/plate.pt")
        ocr_spec = self.cfg.plate_ocr
        backend = ocr_spec.backend or "yolo_chars"
        if backend == "yolo_chars":
            caps["plate_ocr"] = (self.get("plate_ocr") is not None, self.status.get("plate_ocr", "chưa nạp"))
        elif backend == "none":
            caps["plate_ocr"] = (False, "đã tắt OCR (backend none)")
        else:
            caps["plate_ocr"] = (True, f"backend {backend}")
        caps["cargo"] = (self.get("cargo") is not None, self.status.get("cargo", "chưa nạp"))
        caps["lane_seg"] = (self.get("lane_seg") is not None, self.status.get("lane_seg", "chưa nạp"))
        caps["orientation"] = (self.get("orientation") is not None, self.status.get("orientation", "chưa nạp"))
        return caps


def export_model(weights: str, fmt: str = "onnx", imgsz: int = 1280, half: bool = True,
                 device: str = "auto") -> str:
    """Xuất .pt sang .onnx (CPU/GPU phổ thông) hoặc .engine (TensorRT FP16, cần GPU NVIDIA + TensorRT)."""
    from ultralytics import YOLO

    dev = pick_device(device)
    kw: dict[str, Any] = {"format": fmt, "imgsz": imgsz, "device": dev}
    if fmt == "engine":
        kw.update(precision_kwargs(half, dev) or {"half": True})
    path = YOLO(resolve_weights_path(weights) or weights).export(**kw)
    return str(path)
