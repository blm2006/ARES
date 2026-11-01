"""Validation với các GATE thật, không phải một con số mAP để khoe.

Bốn thứ v1 thiếu:
  1. Eval trên CẢ golden (đóng băng) VÀ rolling (30 ngày gần nhất).
  2. Breakdown THEO TỪNG CAMERA + gate: không camera nào tụt > 2%. mAP tổng tăng
     mà một camera tụt là chuyện rất thường, và người dùng chỉ nhìn camera của họ.
  3. recall@IoU0.5 được coi trọng hơn mAP tổng — đếm người thì bỏ sót đắt hơn
     lệch box vài pixel.
  4. Ablation: so với model train CHỈ bằng dữ liệu source='random'. Nếu active
     learning không thắng nhóm đối chứng này thì toàn bộ nguồn A đang là chi phí
     vô ích, và tốt hơn là biết điều đó.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.config import Settings, get_settings
from common.schemas import BBox

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Metric tự tính (không phụ thuộc ultralytics) để test chạy được offline
# ----------------------------------------------------------------------
def match_boxes(preds: list[BBox], gts: list[BBox], iou_thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy matching theo confidence. Trả (tp, fp, fn)."""
    preds = sorted(preds, key=lambda b: -b.conf)
    used = set()
    tp = 0
    for p in preds:
        best, best_iou = -1, iou_thr
        for i, g in enumerate(gts):
            if i in used:
                continue
            v = p.iou(g)
            if v >= best_iou:
                best, best_iou = i, v
        if best >= 0:
            used.add(best)
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def average_precision(preds_by_img: dict, gts_by_img: dict, iou_thr: float = 0.5) -> float:
    """AP kiểu VOC-11 đơn giản hoá, một class (person)."""
    rows = []
    n_gt = sum(len(v) for v in gts_by_img.values())
    if n_gt == 0:
        return 0.0
    for img, preds in preds_by_img.items():
        gts = list(gts_by_img.get(img, []))
        used = set()
        for p in sorted(preds, key=lambda b: -b.conf):
            best, best_iou = -1, iou_thr
            for i, g in enumerate(gts):
                if i in used:
                    continue
                v = p.iou(g)
                if v >= best_iou:
                    best, best_iou = i, v
            if best >= 0:
                used.add(best)
                rows.append((p.conf, 1))
            else:
                rows.append((p.conf, 0))
    if not rows:
        return 0.0
    rows.sort(key=lambda r: -r[0])
    tps = np.cumsum([r[1] for r in rows])
    fps = np.cumsum([1 - r[1] for r in rows])
    recall = tps / n_gt
    precision = tps / np.maximum(tps + fps, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
        ap += p / 11
    return float(ap)


@dataclass
class EvalResult:
    n_images: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    map50: float = 0.0

    @property
    def recall50(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def precision50(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def as_dict(self) -> dict:
        return {
            "n_images": self.n_images,
            "map50": round(self.map50, 4),
            "recall50": round(self.recall50, 4),
            "precision50": round(self.precision50, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def evaluate_set(
    preds_by_img: dict[str, list[BBox]],
    gts_by_img: dict[str, list[BBox]],
    iou_thr: float = 0.5,
) -> EvalResult:
    r = EvalResult(n_images=len(gts_by_img))
    for img, gts in gts_by_img.items():
        tp, fp, fn = match_boxes(preds_by_img.get(img, []), gts, iou_thr)
        r.tp += tp
        r.fp += fp
        r.fn += fn
    r.map50 = average_precision(preds_by_img, gts_by_img, iou_thr)
    return r


# ----------------------------------------------------------------------
# Gate
# ----------------------------------------------------------------------
@dataclass
class ValidationReport:
    golden: dict = field(default_factory=dict)
    rolling: dict = field(default_factory=dict)
    per_camera: dict = field(default_factory=dict)
    baseline_random: dict | None = None
    gates: dict = field(default_factory=dict)
    passed: bool = False
    primary_metric: str = "recall50"

    def as_dict(self) -> dict:
        return {
            "golden": self.golden,
            "rolling": self.rolling,
            "per_camera": self.per_camera,
            "baseline_random": self.baseline_random,
            "gates": self.gates,
            "passed": self.passed,
            "primary_metric": self.primary_metric,
        }


def check_gates(
    new: ValidationReport,
    previous: dict | None,
    settings: Settings | None = None,
) -> ValidationReport:
    s = settings or get_settings()
    gates: dict = {}
    metric = s.val_primary_metric

    # Gate 1 — không camera nào tụt quá ngưỡng.
    prev_cams = (previous or {}).get("per_camera", {})
    worst_cam, worst_drop = None, 0.0
    for cam, cur in new.per_camera.items():
        prev = prev_cams.get(cam)
        if not prev:
            continue
        p, c = prev.get(metric, 0.0), cur.get(metric, 0.0)
        drop = (p - c) / p * 100 if p > 0 else 0.0
        if drop > worst_drop:
            worst_cam, worst_drop = cam, drop
    gates["per_camera"] = {
        "ok": worst_drop <= s.val_per_camera_max_drop_pct,
        "worst_camera": worst_cam,
        "worst_drop_pct": round(worst_drop, 2),
        "limit_pct": s.val_per_camera_max_drop_pct,
    }

    # Gate 2 — golden set không được tụt so với model trước.
    prev_golden = (previous or {}).get("golden", {}).get(metric)
    cur_golden = new.golden.get(metric, 0.0)
    gates["golden"] = {
        "ok": prev_golden is None or cur_golden >= prev_golden,
        "prev": prev_golden,
        "now": cur_golden,
        "metric": metric,
    }

    # Gate 3 — phải thắng baseline chỉ-random. Đây là chỗ chứng minh nguồn A có giá trị.
    if new.baseline_random:
        b = new.baseline_random.get(metric, 0.0)
        c = new.rolling.get(metric, 0.0)
        gates["beats_random_baseline"] = {
            "ok": c >= b,
            "active_learning": c,
            "random_only": b,
            "delta": round(c - b, 4),
            "note": "" if c >= b else
                    "Active learning KHÔNG thắng random. Nguồn A đang tốn chi phí review "
                    "mà không mang lại gì — xem lại tiêu chí 2.1 trước khi thu thêm dữ liệu.",
        }
    else:
        gates["beats_random_baseline"] = {"ok": True, "note": "chưa chạy ablation"}

    new.gates = gates
    new.primary_metric = metric
    new.passed = all(g.get("ok", False) for g in gates.values())
    return new


def load_labels(labels_dir: Path, ids: list[str] | None = None) -> dict[str, list[BBox]]:
    """Đọc label YOLO .txt -> BBox toạ độ chuẩn hoá * (1,1). Dùng cho eval nội bộ."""
    out: dict[str, list[BBox]] = {}
    for p in Path(labels_dir).glob("*.txt"):
        if ids is not None and p.stem not in ids:
            continue
        boxes = []
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _c, cx, cy, w, h = (float(x) for x in parts)
            boxes.append(BBox(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2, conf=1.0))
        out[p.stem] = boxes
    return out


def save_report(report: ValidationReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
