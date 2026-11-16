"""Đọc biển số theo track: dò biển trên ảnh cắt của xe → OCR → ngữ pháp biển VN → hợp nhất nhiều khung.

Backend (chọn trong models.yaml):
  * dò biển: YOLO tự huấn luyện (models/plate.pt) — khuyến nghị; nếu chưa có thì bộ dò cổ điển (hình thái học).
  * OCR: `yolo_chars` (YOLO dò từng ký tự 0-9/A-Z — cách phổ biến cho biển 2 dòng của VN),
         `fast_plate_ocr` (pip install fast-plate-ocr[onnx]), `easyocr` (pip install easyocr), `none`.

Quyền riêng tư: bản đọc chỉ nằm trong RAM theo track; khi track kết thúc mà xe KHÔNG liên quan vi phạm,
mọi bản đọc và ảnh cắt bị xoá (không lưu biển số của người không vi phạm).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from traffic_safety_ai.core.types import MOTOR_VEHICLES, Detection, Track
from traffic_safety_ai.lpr.fusion import FusedPlate, PlateFusion, PlateRead
from traffic_safety_ai.lpr.plate_color import classify_plate_color
from traffic_safety_ai.utils.geometry import crop as crop_box
from traffic_safety_ai.utils.geometry import iou

log = logging.getLogger(__name__)

CHARSET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ")


@dataclass
class PlateBox:
    xyxy: np.ndarray       # toạ độ trong ảnh cắt của xe
    conf: float
    layout: str | None = None


@dataclass
class OCRResult:
    text: str
    conf: float
    layout: str | None = None


class PlateDetectorBase(Protocol):
    name: str

    def detect(self, crops: list[np.ndarray]) -> list[list[PlateBox]]: ...


class OCRBase(Protocol):
    name: str

    def read(self, plates: list[np.ndarray], layouts: list[str | None]) -> list[OCRResult | None]: ...


# ===================================================================== dò biển
class YoloPlateDetector:
    name = "yolo"

    def __init__(self, model: Any):
        self.model = model

    def detect(self, crops: list[np.ndarray]) -> list[list[PlateBox]]:
        outs = self.model.predict(crops)
        res: list[list[PlateBox]] = []
        for o in outs:
            boxes = []
            for d in o.boxes:
                name = d.cls.lower()
                if "2" in name or "square" in name or "two" in name:
                    layout = "2line"
                elif "1" in name or "long" in name or "one" in name:
                    layout = "1line"
                else:
                    layout = "2line" if d.w / max(d.h, 1.0) < 2.4 else "1line"
                boxes.append(PlateBox(d.xyxy, d.conf, layout))
            boxes.sort(key=lambda b: -b.conf)
            res.append(boxes)
        return res


class ClassicPlateDetector:
    """Dò biển bằng hình thái học (không cần model). Độ chính xác thấp — chỉ để có ảnh biển cho người duyệt."""

    name = "classic"

    def detect(self, crops: list[np.ndarray]) -> list[list[PlateBox]]:
        return [classic_plate_candidates(c) for c in crops]


def classic_plate_candidates(crop: np.ndarray, top_k: int = 2) -> list[PlateBox]:
    if crop is None or crop.size == 0:
        return []
    h, w = crop.shape[:2]
    if h < 30 or w < 30:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    y0 = int(h * 0.3)
    g = gray[y0:]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 18), max(3, h // 28)))
    cands: list[tuple[float, PlateBox]] = []
    for op in (cv2.MORPH_BLACKHAT, cv2.MORPH_TOPHAT):
        resp = cv2.morphologyEx(g, op, k)
        gx = np.abs(cv2.Sobel(resp, cv2.CV_32F, 1, 0, ksize=3))
        rng = float(gx.max() - gx.min())
        if rng < 1e-6:
            continue
        gx = (255 * (gx - gx.min()) / rng).astype(np.uint8)
        gx = cv2.GaussianBlur(gx, (5, 5), 0)
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w // 10), max(3, h // 22)))
        closed = cv2.morphologyEx(gx, cv2.MORPH_CLOSE, kc)
        _, th = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.dilate(cv2.erode(th, None, iterations=1), None, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 16 or ch < 8:
                continue
            ar = cw / ch
            frac = (cw * ch) / float(w * h)
            if not (1.0 <= ar <= 8.0 and 0.004 <= frac <= 0.25):
                continue
            fill = cv2.contourArea(c) / float(cw * ch)
            prior = max(1.0 - abs(ar - 1.4) / 1.2, 1.0 - abs(ar - 5.0) / 3.0, 0.0)
            score = 0.6 * fill + 0.4 * prior
            layout = "2line" if ar < 2.4 else "1line"
            # vùng tìm được là dải chữ → nới ra để lấy cả viền biển
            dx, dy = 0.08 * cw, 0.35 * ch
            box = np.array([max(0, x - dx), max(0, y + y0 - dy), min(w, x + cw + dx), min(h, y + ch + y0 + dy)], float)
            cands.append((score, PlateBox(box, score, layout)))
    cands.sort(key=lambda s: -s[0])
    out: list[PlateBox] = []
    for _, b in cands:
        if all(iou(b.xyxy, o.xyxy) < 0.3 for o in out):
            out.append(b)
        if len(out) >= top_k:
            break
    return out


# ===================================================================== OCR
def decode_char_boxes(dets: list[Detection], line_gap_ratio: float = 0.55) -> OCRResult | None:
    """Ghép các hộp ký tự thành chuỗi; tự nhận biển 1 dòng / 2 dòng theo vị trí theo chiều dọc."""
    chars = [d for d in dets if len(d.cls) == 1 and d.cls.upper() in CHARSET or d.cls == "Đ"]
    # NMS giữa các lớp: hai hộp trùng nhau là cùng một ký tự → giữ hộp tin cậy hơn
    chars.sort(key=lambda d: -d.conf)
    kept: list[Detection] = []
    for d in chars:
        if all(iou(d.xyxy, k.xyxy) < 0.5 for k in kept):
            kept.append(d)
    if len(kept) < 4:
        return None
    cy = np.array([d.center[1] for d in kept])
    hs = np.array([d.h for d in kept])
    med_h = float(np.median(hs))
    order = np.argsort(cy)
    gaps = np.diff(cy[order])
    layout = "1line"
    lines: list[list[Detection]]
    if len(gaps) and gaps.max() > line_gap_ratio * med_h:
        k = int(np.argmax(gaps)) + 1
        top = [kept[i] for i in order[:k]]
        bottom = [kept[i] for i in order[k:]]
        lines = [top, bottom]
        layout = "2line"
    else:
        lines = [kept]
    text = "".join(d.cls.upper() for line in lines for d in sorted(line, key=lambda d: d.center[0]))
    confs = np.array([d.conf for d in kept])
    conf = float(np.exp(np.log(np.clip(confs, 1e-3, 1)).mean()))
    return OCRResult(text, conf, layout)


class YoloCharOCR:
    name = "yolo_chars"

    def __init__(self, model: Any):
        self.model = model

    def read(self, plates: list[np.ndarray], layouts: list[str | None]) -> list[OCRResult | None]:
        if not plates:
            return []
        prepared = [_upscale(p, 96) for p in plates]
        outs = self.model.predict(prepared)
        return [decode_char_boxes(o.boxes) for o in outs]


class FastPlateOCRBackend:
    """Backend tuỳ chọn: thư viện fast-plate-ocr (model toàn cầu, CHƯA kiểm chứng với biển 2 dòng VN)."""

    name = "fast_plate_ocr"

    def __init__(self, model_name: str | None = None):
        from fast_plate_ocr import LicensePlateRecognizer  # type: ignore

        self.m = LicensePlateRecognizer(model_name or "cct-s-v2-global-model")

    def read(self, plates: list[np.ndarray], layouts: list[str | None]) -> list[OCRResult | None]:
        out: list[OCRResult | None] = []
        for p in plates:
            try:
                g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if p.ndim == 3 else p
                res = self.m.run(g)
                text = _first_text(res)
                out.append(OCRResult(text.replace("_", ""), 0.7, None) if text else None)
            except Exception as e:  # pragma: no cover - phụ thuộc thư viện ngoài
                log.debug("fast_plate_ocr lỗi: %s", e)
                out.append(None)
        return out


class EasyOCRBackend:
    name = "easyocr"

    def __init__(self) -> None:
        import easyocr  # type: ignore

        self.r = easyocr.Reader(["en"], gpu=False)

    def read(self, plates: list[np.ndarray], layouts: list[str | None]) -> list[OCRResult | None]:
        out: list[OCRResult | None] = []
        allow = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
        for p in plates:
            try:
                res = self.r.readtext(_upscale(p, 96), allowlist=allow, detail=1)
            except Exception:  # pragma: no cover
                out.append(None)
                continue
            if not res:
                out.append(None)
                continue
            items = sorted(res, key=lambda x: (float(np.mean([pt[1] for pt in x[0]])), float(x[0][0][0])))
            text = "".join(x[1] for x in items)
            conf = float(np.mean([x[2] for x in items]))
            layout = "2line" if len(items) >= 2 else "1line"
            out.append(OCRResult(text, conf, layout))
        return out


def _first_text(res: Any) -> str:
    if res is None:
        return ""
    if isinstance(res, str):
        return res
    if isinstance(res, tuple) and res:
        return _first_text(res[0])
    if isinstance(res, list | np.ndarray) and len(res):
        return _first_text(res[0])
    for attr in ("plate", "text", "label"):
        if hasattr(res, attr):
            return str(getattr(res, attr))
    return str(res)


def _upscale(img: np.ndarray, target_h: int) -> np.ndarray:
    h = img.shape[0]
    if h <= 0 or h >= target_h:
        return img
    s = target_h / float(h)
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def plate_quality(plate: np.ndarray) -> float:
    """0..1: kết hợp độ nét (phương sai Laplacian) và kích thước biển trên ảnh."""
    if plate is None or plate.size == 0:
        return 0.0
    g = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim == 3 else plate
    h = g.shape[0]
    gs = cv2.resize(g, (max(8, int(g.shape[1] * 48 / max(h, 1))), 48)) if h > 0 else g
    sharp = float(cv2.Laplacian(gs, cv2.CV_64F).var())
    return 0.5 * min(1.0, sharp / 300.0) + 0.5 * min(1.0, h / 30.0)


# ===================================================================== điều phối
class PlateReader:
    def __init__(self, detector: PlateDetectorBase | None, ocr: OCRBase | None, every_n: int = 2,
                 max_per_frame: int = 8, min_vehicle_h: float = 36.0, min_plate_h: float = 9.0,
                 classes: tuple[str, ...] = MOTOR_VEHICLES):
        self.detector = detector
        self.ocr = ocr
        self.every_n = max(1, every_n)
        self.max_per_frame = max_per_frame
        self.min_vehicle_h = min_vehicle_h
        self.min_plate_h = min_plate_h
        self.classes = classes
        self.fusions: dict[int, PlateFusion] = {}
        self.plate_only: dict[int, tuple[np.ndarray, float]] = {}   # có ảnh biển nhưng chưa OCR được
        self.last_ms = 0.0

    @property
    def available(self) -> bool:
        return self.detector is not None

    def process(self, frame: np.ndarray, tracks: list[Track], frame_idx: int, t: float,
                priority_ids: set[int] | None = None) -> None:
        if self.detector is None:
            return
        priority_ids = priority_ids or set()
        cands = []
        for tr in tracks:
            if tr.cls not in self.classes or tr.h < self.min_vehicle_h:
                continue
            fu = self.fusions.get(tr.id)
            if fu is not None and fu.done() and tr.id not in priority_ids:
                continue
            if (frame_idx + tr.id) % self.every_n and tr.id not in priority_ids:
                continue
            cands.append(tr)
        if not cands:
            return
        cands.sort(key=lambda tr: (tr.id not in priority_ids, -tr.h))
        cands = cands[: self.max_per_frame]
        import time as _t

        t0 = _t.perf_counter()
        vcrops, offsets = [], []
        for tr in cands:
            c, box = crop_box(frame, tr.xyxy, pad=0.06)
            vcrops.append(c)
            offsets.append(box[:2])
        found = self.detector.detect(vcrops)
        plates, metas = [], []
        for tr, boxes, off, vc in zip(cands, found, offsets, vcrops):
            if not boxes:
                continue
            pb = boxes[0]
            gx1, gy1 = pb.xyxy[0] + off[0], pb.xyxy[1] + off[1]
            gb = np.array([gx1, gy1, pb.xyxy[2] + off[0], pb.xyxy[3] + off[1]])
            if gb[3] - gb[1] < self.min_plate_h:
                continue
            pc, _ = crop_box(frame, gb, pad=0.08)
            if pc.size == 0:
                continue
            plates.append(pc)
            metas.append((tr, pb, gb))
        results: list[OCRResult | None] = []
        if plates and self.ocr is not None:
            results = self.ocr.read(plates, [m[1].layout for m in metas])
        else:
            results = [None] * len(plates)
        for (tr, pb, gb), pc, ocr in zip(metas, plates, results):
            q = plate_quality(pc)
            color, _ = classify_plate_color(pc)
            tr.attrs["plate_box"] = gb
            if ocr is None or not ocr.text:
                prev = self.plate_only.get(tr.id)
                if prev is None or q > prev[1]:
                    self.plate_only[tr.id] = (pc.copy(), q)
                continue
            fu = self.fusions.setdefault(tr.id, PlateFusion(tr.cls))
            fu.add(PlateRead(ocr.text, ocr.conf * pb.conf ** 0.5, q, frame_idx, t, crop=pc.copy(), color=color,
                             layout=ocr.layout or pb.layout))
        self.last_ms = (_t.perf_counter() - t0) * 1000.0

    def add_read(self, track_id: int, vehicle_cls: str, read: PlateRead) -> None:
        """Nạp kết quả đọc từ nguồn ngoài (mô phỏng, hoặc camera LPR chuyên dụng)."""
        self.fusions.setdefault(track_id, PlateFusion(vehicle_cls)).add(read)

    def result(self, track_id: int) -> FusedPlate | None:
        fu = self.fusions.get(track_id)
        return fu.result() if fu else None

    def plate_image(self, track_id: int) -> np.ndarray | None:
        res = self.result(track_id)
        if res is not None and res.crop is not None:
            return res.crop
        po = self.plate_only.get(track_id)
        return po[0] if po else None

    def forget(self, track_id: int) -> None:
        self.fusions.pop(track_id, None)
        self.plate_only.pop(track_id, None)


def build_plate_reader(hub: Any, models_cfg: Any) -> tuple[PlateReader, str]:
    """Tạo PlateReader từ ModelHub + cấu hình. Trả (reader, mô tả backend)."""
    det: PlateDetectorBase | None
    plate_model = hub.get("plate") if hub is not None else None
    det = YoloPlateDetector(plate_model) if plate_model is not None else ClassicPlateDetector()
    spec = models_cfg.plate_ocr
    backend = (spec.backend or "yolo_chars").lower()
    ocr: OCRBase | None = None
    note = ""
    try:
        if backend == "yolo_chars":
            m = hub.get("plate_ocr") if hub is not None else None
            ocr = YoloCharOCR(m) if m is not None else None
            if ocr is None:
                note = "chưa có model ký tự (models/plate_chars.pt) — chỉ lưu ảnh biển để người duyệt đọc"
        elif backend == "fast_plate_ocr":
            ocr = FastPlateOCRBackend(spec.model_name)
        elif backend == "easyocr":
            ocr = EasyOCRBackend()
    except Exception as e:
        note = f"không khởi tạo được OCR {backend}: {e}"
        ocr = None
    desc = f"dò biển: {det.name}; OCR: {ocr.name if ocr else 'không có'}" + (f" ({note})" if note else "")
    reader = PlateReader(det, ocr, every_n=models_cfg.plate.every_n, max_per_frame=models_cfg.plate.max_crops)
    return reader, desc
