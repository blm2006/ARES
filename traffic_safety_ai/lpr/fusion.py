"""Hợp nhất kết quả đọc biển số của một xe qua NHIỀU khung hình.

Một khung đơn lẻ hay đọc sai 1 ký tự (mờ, nghiêng, loá). Trong lúc xe đi qua camera ta có hàng chục lần đọc:
  1. mỗi lần đọc có trọng số = độ tin OCR × chất lượng ảnh (nét, to) × (hợp lệ theo ngữ pháp biển VN);
  2. bỏ phiếu cả CHUỖI, đồng thời bỏ phiếu TỪNG VỊ TRÍ ký tự giữa các lần đọc cùng mẫu;
  3. độ tin cậy cuối = tỉ lệ đồng thuận × (1 − e^(−số lần khớp/2)) → một lần đọc duy nhất không bao giờ "chắc chắn".
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from traffic_safety_ai.lpr.vn_plate import COLOR_LABELS, PlateParse, parse_plate, province_label


@dataclass
class PlateRead:
    raw: str
    conf: float
    quality: float
    frame_idx: int
    t: float
    parse: PlateParse | None = None
    crop: np.ndarray | None = None
    color: str | None = None
    layout: str | None = None


@dataclass
class FusedPlate:
    text: str
    display: str
    kind: str
    valid: bool
    confidence: float
    n_reads: int
    n_agree: int
    province_code: str
    province: str
    color: str
    color_label: str
    crop: np.ndarray | None = None
    candidates: list[tuple[str, float]] = field(default_factory=list)


class PlateFusion:
    def __init__(self, vehicle_cls: str | None = None, max_reads: int = 40):
        self.vehicle_cls = vehicle_cls
        self.max_reads = max_reads
        self.reads: list[PlateRead] = []
        self.n_seen = 0

    @staticmethod
    def weight(r: PlateRead) -> float:
        if r.parse is None:
            return 0.0
        w = max(0.05, r.conf) * (0.3 + 0.7 * max(0.0, min(1.0, r.quality)))
        w *= 1.0 if r.parse.valid else 0.35
        return w / (1.0 + 0.5 * r.parse.corrections)

    def add(self, r: PlateRead) -> None:
        self.n_seen += 1
        if r.parse is None and r.raw:
            r.parse = parse_plate(r.raw, self.vehicle_cls, r.layout)
        if r.parse is None:
            return
        self.reads.append(r)
        if len(self.reads) > self.max_reads:
            # giữ các lần đọc có trọng số cao nhất (nhưng không ôm ảnh cắt của tất cả)
            self.reads.sort(key=self.weight, reverse=True)
            for x in self.reads[self.max_reads:]:
                x.crop = None
            self.reads = self.reads[: self.max_reads]

    def done(self, min_agree: int = 5, min_conf: float = 0.9) -> bool:
        res = self.result()
        return res is not None and res.valid and res.n_agree >= min_agree and res.confidence >= min_conf

    def result(self) -> FusedPlate | None:
        reads = [r for r in self.reads if r.parse is not None]
        if not reads:
            return None
        groups: dict[str, float] = defaultdict(float)
        for r in reads:
            groups[r.parse.text] += self.weight(r)
        total = sum(groups.values()) or 1e-9
        best_text = max(groups, key=groups.get)
        best_parse = next(r.parse for r in reads if r.parse.text == best_text)
        final_text, final_parse = best_text, best_parse
        share = groups[best_text] / total

        # bỏ phiếu theo từng vị trí trong các lần đọc cùng mẫu (cùng độ dài & loại xe)
        same = [r for r in reads if r.parse.template == best_parse.template]
        if len(same) >= 3:
            L = len(best_text)
            votes = [Counter() for _ in range(L)]
            for r in same:
                w = self.weight(r)
                for i, ch in enumerate(r.parse.text):
                    votes[i][ch] += w
            composite = "".join(v.most_common(1)[0][0] for v in votes)
            agree = float(np.mean([v.most_common(1)[0][1] / max(sum(v.values()), 1e-9) for v in votes]))
            if composite != best_text:
                cp = parse_plate(composite, self.vehicle_cls)
                if cp is not None and cp.valid and agree > share + 0.1:
                    final_text, final_parse, share = cp.text, cp, agree

        n_agree = sum(1 for r in reads if _hamming_ok(r.parse.text, final_text, 0))
        n_close = sum(1 for r in reads if _hamming_ok(r.parse.text, final_text, 1))
        conf = share * (1.0 - math.exp(-max(n_agree, 0.5 * n_close) / 2.0))

        colors = Counter()
        for r in reads:
            if r.color and r.color != "unknown":
                colors[r.color] += self.weight(r)
        color = colors.most_common(1)[0][0] if colors else "unknown"

        best_crop_read = max((r for r in reads if r.crop is not None and _hamming_ok(r.parse.text, final_text, 1)),
                             key=lambda r: r.quality * r.conf, default=None)
        if best_crop_read is None:
            best_crop_read = max((r for r in reads if r.crop is not None), key=lambda r: r.quality, default=None)
        cands = sorted(((next(r.parse.display for r in reads if r.parse.text == t), w / total)
                        for t, w in groups.items()), key=lambda x: -x[1])[:3]
        return FusedPlate(
            text=final_text, display=final_parse.display, kind=final_parse.kind, valid=final_parse.valid,
            confidence=float(round(conf, 4)), n_reads=len(reads), n_agree=n_agree,
            province_code=final_parse.province_code, province=province_label(final_parse.province_code),
            color=color, color_label=COLOR_LABELS.get(color, COLOR_LABELS["unknown"]),
            crop=None if best_crop_read is None else best_crop_read.crop, candidates=cands,
        )


def _hamming_ok(a: str, b: str, max_diff: int) -> bool:
    if len(a) != len(b):
        return False
    return sum(1 for x, y in zip(a, b) if x != y) <= max_diff
