"""Gán người ↔ xe hai bánh (ai đang ngồi trên xe nào).

Spec gốc dùng IoA > 0.3 theo từng frame — ở giao thông VN xe máy chen sát nhau nên một người đứng
cạnh xe hoặc ngồi xe bên cạnh rất dễ bị gán nhầm. Ở đây:
  * điểm ghép tổng hợp: IoA với hộp xe đã nới, vị trí chân so với thân xe, căn giữa theo chiều ngang,
    tỉ lệ kích thước và ĐỘ ĐỒNG PHA CHUYỂN ĐỘNG (cùng hướng, cùng tốc độ);
  * mỗi người chỉ thuộc tối đa một xe;
  * bỏ phiếu theo thời gian (vote_window frame) → không nhảy gán qua lại.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from traffic_safety_ai.core.types import Track
from traffic_safety_ai.utils.geometry import cosine_similarity, expand_box, ioa, unit


@dataclass
class AssociationConfig:
    up_ratio: float = 1.2         # nới hộp xe lên trên (theo chiều cao hộp xe) để chứa thân người ngồi
    side_ratio: float = 0.25
    down_ratio: float = 0.1
    min_ioa: float = 0.45
    min_score: float = 0.5
    vote_window: int = 12
    min_votes: int = 4
    min_size_ratio: float = 0.45  # chiều cao người / chiều cao xe (loại người ở xa phía sau trùng hình)
    max_size_ratio: float = 4.0
    child_height_ratio: float = 0.72


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def pair_score(p: Track, m: Track, cfg: AssociationConfig) -> float:
    region = expand_box(m.xyxy, left=cfg.side_ratio, top=cfg.up_ratio, right=cfg.side_ratio, bottom=cfg.down_ratio)
    a = ioa(p.xyxy, region)
    if a < cfg.min_ioa:
        return 0.0
    mh, mw = max(m.h, 1.0), max(m.w, 1.0)
    ratio = p.h / mh
    if not (cfg.min_size_ratio <= ratio <= cfg.max_size_ratio):
        return 0.0
    # chân người ngồi trên xe nằm quanh nửa dưới thân xe
    foot = 1.0 - _clamp01(abs(p.xyxy[3] - (m.xyxy[1] + 0.8 * mh)) / (0.8 * mh))
    align = 1.0 - _clamp01(abs(p.center[0] - m.center[0]) / (0.5 * mw + 0.25 * max(p.w, 1.0)))
    coh = 0.5
    if p.kin.n >= 3 and m.kin.n >= 3:
        sp, sm = p.kin.speed_norm, m.kin.speed_norm
        if sp > 0.3 or sm > 0.3:
            cos = cosine_similarity((p.kin.vx_px, p.kin.vy_px), (m.kin.vx_px, m.kin.vy_px))
            r = min(p.kin.speed_px, m.kin.speed_px) / max(p.kin.speed_px, m.kin.speed_px, 1e-6)
            coh = max(0.0, cos) * r
            if cos < 0.0:          # đi ngược hướng nhau: chắc chắn không cùng xe
                return 0.0
        else:
            coh = 0.6
    return 0.30 * _clamp01(a / 0.9) + 0.25 * foot + 0.20 * align + 0.25 * coh


class RiderAssociator:
    def __init__(self, cfg: AssociationConfig | None = None):
        self.cfg = cfg or AssociationConfig()
        self._votes: dict[int, deque[int]] = {}
        self._last_seen: dict[int, int] = {}
        self._frame = 0

    def update(self, persons: list[Track], cycles: list[Track]) -> tuple[dict[int, list[int]], dict[int, int]]:
        cfg = self.cfg
        self._frame += 1
        for p in persons:
            best_id, best = -1, cfg.min_score
            for m in cycles:
                s = pair_score(p, m, cfg)
                if s > best:
                    best, best_id = s, m.id
            dq = self._votes.setdefault(p.id, deque(maxlen=cfg.vote_window))
            dq.append(best_id)
            self._last_seen[p.id] = self._frame
        # dọn người đã biến mất
        for pid in [k for k, f in self._last_seen.items() if self._frame - f > 3 * cfg.vote_window]:
            self._votes.pop(pid, None)
            self._last_seen.pop(pid, None)

        cycle_ids = {m.id for m in cycles}
        rider_of: dict[int, int] = {}
        for p in persons:
            dq = self._votes.get(p.id)
            if not dq:
                continue
            c = Counter(x for x in dq if x != -1 and x in cycle_ids)
            if not c:
                continue
            mid, n = c.most_common(1)[0]
            # cần đủ phiếu; track mới (ít frame) chỉ cần đa số tuyệt đối
            need = min(cfg.min_votes, max(1, len(dq) // 2 + 1))
            if n >= need and dq[-1] == mid:
                rider_of[p.id] = mid
            elif n >= cfg.min_votes:
                rider_of[p.id] = mid

        riders: dict[int, list[int]] = {}
        by_id = {m.id: m for m in cycles}
        pmap = {p.id: p for p in persons}
        for pid, mid in rider_of.items():
            riders.setdefault(mid, []).append(pid)
        for mid, pids in riders.items():
            riders[mid] = order_driver_first(by_id[mid], [pmap[i] for i in pids])
        return riders, rider_of


def order_driver_first(moto: Track, people: list[Track]) -> list[int]:
    """Sắp người lái lên đầu: người ở phía TRƯỚC nhất theo hướng chuyển động của xe."""
    if len(people) <= 1:
        return [p.id for p in people]
    v = (moto.kin.vx_px, moto.kin.vy_px)
    if moto.kin.speed_norm > 0.3 and (abs(v[0]) + abs(v[1])) > 1e-6:
        u = unit(v)
        mc = np.asarray(moto.center)
        key = {p.id: float(np.dot(np.asarray(p.center) - mc, u)) for p in people}
        return [pid for pid, _ in sorted(key.items(), key=lambda kv: -kv[1])]
    # đứng yên: không đoán được ai lái — giữ người cao nhất (thường là người lớn) lên đầu
    return [p.id for p in sorted(people, key=lambda p: -p.h)]


def maybe_child(passenger: Track, driver: Track, ratio: float = 0.72) -> bool:
    """Ước lượng thô: người ngồi sau thấp hơn hẳn người lái → có thể là trẻ em (cần người xác minh)."""
    return passenger.h < ratio * driver.h
