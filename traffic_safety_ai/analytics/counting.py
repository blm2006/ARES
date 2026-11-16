"""Đếm lưu lượng qua mặt cắt, đo tốc độ bằng cặp vạch, lấy mẫu tốc độ — số liệu cho báo cáo khảo sát giao thông.

Hệ số quy đổi ra xe con (xcqđ) mặc định theo TCVN 4054:2005 Bảng 2 (địa hình đồng bằng): xe đạp 0,2; xe máy 0,3;
xe con 1,0; xe tải 2 trục và xe buýt dưới 25 chỗ 2,0; xe tải ≥3 trục và xe buýt lớn 2,5. TCVN 13592:2022 (đường đô
thị) gần như giữ nguyên các hệ số này. Có thể đổi trong `analytics.pcu` của file cấu hình.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from traffic_safety_ai.core.config import SiteConfig
from traffic_safety_ai.core.types import VEHICLE_CLASSES, FrameContext, Track
from traffic_safety_ai.rules.base_rule import on_two_wheeler
from traffic_safety_ai.utils.geometry import line_crossing, side_of_line

PCU_DEFAULT: dict[str, float] = {"bicycle": 0.2, "motorcycle": 0.3, "car": 1.0, "truck": 2.0, "bus": 2.5}


def bucket_start(when: datetime, minutes: int = 15) -> str:
    m = (when.minute // minutes) * minutes
    return when.replace(minute=m, second=0, microsecond=0).isoformat(timespec="minutes")


def _forward_sign(points: list, direction: list) -> int:
    a, b = points[0], points[1]
    s0 = side_of_line(direction[0], a, b)
    s1 = side_of_line(direction[1], a, b)
    if s0 == s1:
        return 0
    return 1 if s1 > s0 else -1


class LineCounter:
    """Đếm xe cắt mặt cắt, có TRỄ (hysteresis) để không đếm sai khi điểm chân xe rung ngay trên vạch.

    Ý tưởng tham khảo LineZone của roboflow/supervision (viết lại): một lần cắt vạch chỉ là "chờ xác nhận"; được
    tính khi điểm chân đã sang hẳn bên kia ≥ `hyst` (8% chiều cao hộp, tối thiểu 2 px). Quay lại bên cũ → huỷ.
    Xe mất dấu (bị che, ra khỏi khung) khi đang chờ xác nhận vẫn được tính — nó đã cắt vạch thật. Rất quan trọng
    khi đường ùn tắc: xe máy dừng đè đúng vạch đếm, hộp rung qua lại vạch hàng chục lần.
    """

    def __init__(self, scene: SiteConfig, bucket_minutes: int = 15, hyst_frac: float = 0.08,
                 hyst_min_px: float = 2.0):
        self.scene = scene
        self.bucket_minutes = bucket_minutes
        self.hyst_frac = hyst_frac
        self.hyst_min_px = hyst_min_px
        self.counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
        self.totals: dict[tuple[str, str, str], int] = defaultdict(int)
        self._done: set[tuple[int, str]] = set()
        # (track, vạch) → (hướng cắt, nhãn chiều, lớp xe, mốc thời gian)
        self._pending: dict[tuple[int, str], tuple[int, str, str, datetime]] = {}

    @staticmethod
    def _signed_dist(p, a, b) -> float:
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = (dx * dx + dy * dy) ** 0.5
        if n <= 1e-9:
            return 0.0
        return (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / n

    def _commit(self, key: tuple[int, str], cls: str, label: str, when: datetime) -> None:
        ln = next((x for x in self.scene.lines.counting if x.id == key[1]), None)
        name = (ln.name or ln.id) if ln is not None else key[1]
        b = bucket_start(when, self.bucket_minutes)
        self.counts[(key[1], name, b, cls, label)] += 1
        self.totals[(key[1], cls, label)] += 1

    def update(self, ctx: FrameContext) -> list[tuple[Track, str, str]]:
        out = []
        # camera CCTV vừa rung/xoay: đưa điểm khung trước về toạ độ khung này (warp nền) để xe đứng yên không bị
        # đếm là "cắt vạch" chỉ vì cả khung hình trôi theo camera
        warp = ctx.aux.get("camera_warp")
        for tr in ctx.tracks:
            if tr.cls not in VEHICLE_CLASSES and tr.cls != "person":
                continue
            if len(tr.history) < 2:
                continue
            if tr.cls == "person" and (tr.id in ctx.rider_of or on_two_wheeler(ctx, tr)):
                continue            # người ngồi trên xe không tính là người đi bộ
            h0, h1 = tr.history[-2], tr.history[-1]
            p0, p1 = h0.ground_px, h1.ground_px
            hyst = max(self.hyst_min_px, self.hyst_frac * max(1.0, tr.xyxy[3] - tr.xyxy[1]))
            for ln in self.scene.lines.counting:
                key = (tr.id, ln.id)
                pend = self._pending.get(key)
                if pend is not None:
                    sd = self._signed_dist(p1, ln.points[0], ln.points[1]) * pend[0]
                    if sd >= hyst:
                        del self._pending[key]
                        self._commit(key, tr.cls, pend[1], pend[3])
                        out.append((tr, ln.id, pend[1]))
                    elif sd <= -hyst:
                        del self._pending[key]          # chỉ rung qua vạch rồi quay lại → chưa từng cắt thật
                        self._done.discard(key)
                    continue
                if key in self._done:
                    continue
                q0 = p0
                if warp is not None:
                    if h1.frame_idx - h0.frame_idx != 1:
                        continue    # mất dấu đúng lúc camera trôi → không đủ căn cứ để đếm bước này
                    q = np.asarray(warp, float) @ np.array([p0[0], p0[1], 1.0])
                    q0 = (float(q[0]), float(q[1]))
                d, _ = line_crossing(q0, p1, ln.points[0], ln.points[1], margin=0.02)
                if d == 0:
                    continue
                fwd = _forward_sign(ln.points, ln.direction)
                label = ln.positive_label if (fwd == 0 or d == fwd) else ln.negative_label
                self._done.add(key)
                # d = +1: cross2 đi từ âm sang dương; _signed_dist cùng dấu với cross2
                if self._signed_dist(p1, ln.points[0], ln.points[1]) * d >= hyst:
                    self._commit(key, tr.cls, label, ctx.wall_time)       # đã sang hẳn bên kia ngay bước này
                    out.append((tr, ln.id, label))
                else:
                    self._pending[key] = (d, label, tr.cls, ctx.wall_time)
        return out

    def finalize(self) -> None:
        """Hết video: xe đang chờ xác nhận ở vạch (vd video cắt ngang lúc xe vừa qua) vẫn được tính."""
        for tid in {k[0] for k in self._pending}:
            self.forget(tid)

    def forget(self, track_id: int) -> None:
        for key in [k for k in self._pending if k[0] == track_id]:
            _d, label, cls, when = self._pending.pop(key)
            self._commit(key, cls, label, when)              # mất dấu khi đang qua vạch: vẫn là một lượt qua
        self._done = {k for k in self._done if k[0] != track_id}

    def flush_rows(self) -> list[tuple[str, str, str, str, str, int]]:
        rows = [(lid, lname, b, cls, d, n) for (lid, lname, b, cls, d), n in self.counts.items()]
        self.counts.clear()
        return rows


class SpeedTrapMeter:
    """Đo tốc độ trung bình đoạn: thời gian đi giữa hai vạch ảo cách nhau `distance_m` (nội suy dưới mức khung)."""

    def __init__(self, scene: SiteConfig):
        self.scene = scene
        self._t: dict[tuple[int, str, str], float] = {}

    def forget(self, track_id: int) -> None:
        for k in [k for k in self._t if k[0] == track_id]:
            del self._t[k]

    def update(self, ctx: FrameContext) -> list[tuple[Track, float]]:
        out = []
        if ctx.aux.get("camera_moving"):
            return out               # camera rung: vạch đo trên ảnh lệch khỏi vạch thật → thời điểm cắt vạch vô nghĩa
        for tr in ctx.tracks:
            if tr.cls not in VEHICLE_CLASSES or len(tr.history) < 2:
                continue
            h0, h1 = tr.history[-2], tr.history[-1]
            for trap in self.scene.lines.speed_traps:
                for name, line in (("a", trap.line_a), ("b", trap.line_b)):
                    d, frac = line_crossing(h0.ground_px, h1.ground_px, line[0], line[1], margin=0.05)
                    if d == 0:
                        continue
                    fwd = _forward_sign(line, trap.direction)
                    if fwd != 0 and d != fwd:
                        continue
                    self._t[(tr.id, trap.id, name)] = h0.t + frac * (h1.t - h0.t)
                ta = self._t.get((tr.id, trap.id, "a"))
                tb = self._t.get((tr.id, trap.id, "b"))
                if ta is not None and tb is not None and tb <= ta:
                    self._t.pop((tr.id, trap.id, "b"), None)      # cắt b trước a (đi lùi/nhiễu) → chờ lần cắt b mới
                    tb = None
                if ta is not None and tb is not None:
                    v = trap.distance_m / (tb - ta) * 3.6
                    tr.attrs["trap_speed"] = (v, ctx.t)
                    out.append((tr, v))
                    self._t.pop((tr.id, trap.id, "a"), None)
                    self._t.pop((tr.id, trap.id, "b"), None)
        return out


class SpeedSampler:
    """Mỗi xe một mẫu tốc độ đại diện (trung vị các ước lượng tin cậy) — để tính V50, V85.

    Theo cách khảo sát tốc độ điểm: chỉ lấy lúc xe ĐANG CHẠY (≥ min_kmh); xe dừng đèn đỏ/đỗ không kéo tụt V15–V85.
    """

    def __init__(self, max_sigma_kmh: float = 4.0, min_kmh: float = 5.0):
        self.max_sigma = max_sigma_kmh
        self.min_kmh = min_kmh
        self._samples: dict[int, list[float]] = defaultdict(list)
        self._cls: dict[int, str] = {}

    def update(self, ctx: FrameContext) -> None:
        if ctx.calib is None:
            return
        for tr in ctx.vehicles(("motorcycle", "car", "bus", "truck")):
            k = tr.kin
            if k.speed_kmh is None or k.window_s < 0.6 or k.speed_kmh < self.min_kmh:
                continue
            if k.speed_sigma_kmh is not None and k.speed_sigma_kmh > self.max_sigma:
                continue
            if not ctx.calib.in_calibrated_area(tr.bottom):
                continue
            self._samples[tr.id].append(k.speed_kmh)
            self._cls[tr.id] = tr.cls

    def finish(self, track_id: int) -> tuple[str, float] | None:
        s = self._samples.pop(track_id, None)
        cls = self._cls.pop(track_id, None)
        if not s or len(s) < 5 or cls is None:
            return None
        return cls, float(statistics.median(s))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize_counts(rows: list[dict[str, Any]], pcu: dict[str, float] | None = None) -> dict[str, Any]:
    """Bảng 15 phút theo mặt cắt: số xe từng loại, tổng, xcqđ; giờ cao điểm và hệ số giờ cao điểm PHF."""
    pcu = pcu or PCU_DEFAULT
    by_line: dict[str, dict[str, Any]] = {}
    for r in rows:
        line = by_line.setdefault(r["line_id"], {"name": r.get("line_name") or r["line_id"], "buckets": {}})
        b = line["buckets"].setdefault(r["bucket_start"], defaultdict(int))
        b[r["cls"]] += int(r["n"])
        b["_dir_" + r["direction"]] += int(r["n"])
    out: dict[str, Any] = {}
    for lid, line in by_line.items():
        table = []
        for bk in sorted(line["buckets"]):
            c = line["buckets"][bk]
            veh = {k: c.get(k, 0) for k in ("motorcycle", "bicycle", "car", "bus", "truck")}
            total = sum(veh.values())
            pc = sum(veh[k] * pcu.get(k, 1.0) for k in veh)
            dirs = {k[5:]: v for k, v in c.items() if k.startswith("_dir_")}
            table.append({"bucket": bk, **veh, "person": c.get("person", 0), "total": total, "pcu": round(pc, 1),
                          "dirs": dirs})
        peak = None
        for i in range(len(table)):
            win = table[i:i + 4]
            if len(win) < 4:
                break
            pcu_h = sum(x["pcu"] for x in win)
            if peak is None or pcu_h > peak["pcu_h"]:
                max15 = max(x["pcu"] for x in win)
                peak = {"start": win[0]["bucket"], "pcu_h": round(pcu_h, 1),
                        "phf": round(pcu_h / (4 * max15), 3) if max15 else None}
        out[lid] = {"name": line["name"], "rows": table, "peak": peak,
                    "total": sum(x["total"] for x in table), "pcu": round(sum(x["pcu"] for x in table), 1)}
    return out


def summarize_speeds(rows: list[dict[str, Any]], limit_kmh: float | None = None) -> dict[str, Any]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["cls"]].append(float(r["speed_kmh"]))
    by["Tất cả"] = [x for v in list(by.values()) for x in v]
    out = {}
    for cls, v in by.items():
        if not v:
            continue
        out[cls] = {"n": len(v), "mean": round(statistics.mean(v), 1), "v15": round(percentile(v, 0.15), 1),
                    "v50": round(percentile(v, 0.5), 1), "v85": round(percentile(v, 0.85), 1), "max": round(max(v), 1),
                    "pct_over": None if limit_kmh is None else round(100.0 * sum(x > limit_kmh for x in v) / len(v), 1)}
    return out
