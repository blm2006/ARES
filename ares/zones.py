"""Spatial zoning: ánh xạ bounding box -> vùng (Zone 1..N).

Theo spec: lấy điểm giữa-đáy (x_center, y_bottom) của bbox — đó là chỗ chân người chạm sàn,
ổn định hơn tâm bbox khi người ngồi/đứng/che khuất một phần — rồi tra đa giác.
Đa giác lưu dạng chuẩn hoá 0..1 nên không đổi khi đổi độ phân giải camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from ares.config import RoomSpec

EPS = 1e-9


class BoxLike(Protocol):
    x1: float
    y1: float
    x2: float
    y2: float


def _on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> bool:
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > 1e-9:
        return False
    return min(ax, bx) - EPS <= px <= max(ax, bx) + EPS and min(ay, by) - EPS <= py <= max(ay, by) + EPS


def point_in_polygon(px: float, py: float, poly: Sequence[tuple[float, float]]) -> bool:
    """Ray casting; điểm nằm TRÊN cạnh tính là bên trong (chân người sát mép khung vẫn được đếm)."""
    n = len(poly)
    inside = False
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        if _on_segment(px, py, ax, ay, bx, by):
            return True
        if (ay > py) != (by > py):
            x_at = (bx - ax) * (py - ay) / (by - ay) + ax
            if px < x_at:
                inside = not inside
    return inside


def foot_point(box: BoxLike, width: int, height: int) -> tuple[float, float]:
    """Điểm giữa-đáy của bbox, chuẩn hoá 0..1 và kẹp vào khung hình."""
    cx = (box.x1 + box.x2) / 2.0 / max(1, width)
    by = box.y2 / max(1, height)
    return min(1.0, max(0.0, cx)), min(1.0, max(0.0, by))


@dataclass
class ZoneCount:
    total: int  # mọi người detect được (kể cả ngoài các vùng)
    zones: list[int]  # N_z1..N_zn
    unzoned: int  # người nằm ngoài mọi vùng

    @property
    def zoned_total(self) -> int:
        return sum(self.zones)


class ZoneMap:
    def __init__(self, room: RoomSpec):
        self.room = room
        self.polys = [z.polygon for z in room.zones]
        self.n = len(self.polys)

    def locate(self, xn: float, yn: float) -> int | None:
        """Chỉ số vùng (0-based) chứa điểm chuẩn hoá, hoặc None. Vùng chồng nhau: vùng có id nhỏ thắng."""
        for i, poly in enumerate(self.polys):
            if point_in_polygon(xn, yn, poly):
                return i
        return None

    def count(self, boxes: Iterable[BoxLike], width: int, height: int) -> ZoneCount:
        zones = [0] * self.n
        total = 0
        unzoned = 0
        for b in boxes:
            total += 1
            xn, yn = foot_point(b, width, height)
            idx = self.locate(xn, yn)
            if idx is None:
                unzoned += 1
            else:
                zones[idx] += 1
        return ZoneCount(total=total, zones=zones, unzoned=unzoned)

    def capacities(self) -> list[int]:
        return [z.capacity for z in self.room.zones]

    def k_factors(self) -> list[float]:
        return [z.k for z in self.room.zones]

    def draw(self, img, counts: Sequence[int] | None = None) -> None:
        """Vẽ viền vùng lên ảnh (in-place) cho dashboard."""
        import cv2
        import numpy as np

        h, w = img.shape[:2]
        palette = [(255, 160, 40), (80, 200, 120), (60, 140, 255), (200, 90, 220), (0, 200, 220), (180, 180, 60)]
        for i, z in enumerate(self.room.zones):
            pts = np.array([[int(x * (w - 1)), int(y * (h - 1))] for x, y in z.polygon], dtype=np.int32)
            col = palette[i % len(palette)]
            cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
            label = f"{z.id}" if counts is None else f"{z.id}: {counts[i]}"
            cv2.putText(img, label, (int(pts[:, 0].min()) + 6, int(pts[:, 1].min()) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
