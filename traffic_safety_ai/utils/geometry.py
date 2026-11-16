"""Hàm hình học dùng chung: IoU/IoA, điểm trong đa giác, cắt vạch, cosine.

Mọi hộp (box) ở dạng xyxy = (x1, y1, x2, y2) theo pixel. Điểm là (x, y).
Không phụ thuộc shapely: tự viết bằng numpy để cài đặt nhẹ và test được.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

Point = tuple[float, float]


# --------------------------------------------------------------------- hộp
def as_xyxy(b: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(b, dtype=np.float64).reshape(4)


def box_area(b: Sequence[float] | np.ndarray) -> float:
    x1, y1, x2, y2 = as_xyxy(b)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(b: Sequence[float] | np.ndarray) -> Point:
    x1, y1, x2, y2 = as_xyxy(b)
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def bottom_center(b: Sequence[float] | np.ndarray) -> Point:
    """Điểm tiếp đất (giữa cạnh đáy) — dùng cho mọi phép đo trên mặt đường."""
    x1, _y1, x2, y2 = as_xyxy(b)
    return (0.5 * (x1 + x2), float(y2))


def box_wh(b: Sequence[float] | np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = as_xyxy(b)
    return (max(0.0, x2 - x1), max(0.0, y2 - y1))


def intersection(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = as_xyxy(a)
    bx1, by1, bx2, by2 = as_xyxy(b)
    w = min(ax2, bx2) - max(ax1, bx1)
    h = min(ay2, by2) - max(ay1, by1)
    return max(0.0, w) * max(0.0, h)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = intersection(a, b)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def ioa(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over Area của a: phần diện tích của a nằm trong b (0..1)."""
    area = box_area(a)
    return float(intersection(a, b) / area) if area > 0 else 0.0


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Ma trận IoU (Na, Nb), vector hoá."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def expand_box(b: Sequence[float], left: float = 0.0, top: float = 0.0, right: float = 0.0,
               bottom: float = 0.0) -> np.ndarray:
    """Nới hộp theo TỈ LỆ của chiều rộng/cao (vd top=0.15 → nới lên 15% chiều cao)."""
    x1, y1, x2, y2 = as_xyxy(b)
    w, h = x2 - x1, y2 - y1
    return np.array([x1 - left * w, y1 - top * h, x2 + right * w, y2 + bottom * h], dtype=np.float64)


def clip_box(b: Sequence[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = as_xyxy(b)
    return np.array([min(max(x1, 0), width - 1), min(max(y1, 0), height - 1),
                     min(max(x2, 0), width - 1), min(max(y2, 0), height - 1)], dtype=np.float64)


def union_box(boxes: Iterable[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(list(boxes), dtype=np.float64).reshape(-1, 4)
    if len(arr) == 0:
        raise ValueError("union_box cần ít nhất một hộp")
    return np.array([arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()])


def crop(img: np.ndarray, b: Sequence[float], pad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Cắt ảnh theo hộp (có thể nới pad theo tỉ lệ). Trả (ảnh cắt, hộp thực đã kẹp vào khung)."""
    h, w = img.shape[:2]
    bb = expand_box(b, pad, pad, pad, pad) if pad else as_xyxy(b)
    x1, y1, x2, y2 = clip_box(bb, w, h)
    x1, y1, x2, y2 = int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0], np.array([x1, y1, x1, y1], dtype=np.float64)
    return img[y1:y2, x1:x2], np.array([x1, y1, x2, y2], dtype=np.float64)


# --------------------------------------------------------------------- đa giác
def as_polygon(poly: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(arr) < 3:
        raise ValueError("Đa giác cần ít nhất 3 đỉnh")
    return arr


def point_in_polygon(pt: Sequence[float], poly: Sequence[Sequence[float]] | np.ndarray) -> bool:
    """Ray casting; điểm nằm đúng trên cạnh được tính là bên trong."""
    x, y = float(pt[0]), float(pt[1])
    p = as_polygon(poly)
    n = len(p)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = p[i]
        xj, yj = p[j]
        # trên cạnh
        if _on_segment((x, y), (xi, yi), (xj, yj)):
            return True
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def points_in_polygon(pts: np.ndarray, poly: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Phiên bản vector hoá (không xét riêng điểm nằm trên cạnh)."""
    q = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = as_polygon(poly)
    x, y = q[:, 0], q[:, 1]
    inside = np.zeros(len(q), dtype=bool)
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        cond = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        inside ^= cond & (x < x_cross)
        j = i
    return inside


def polygon_area(poly: Sequence[Sequence[float]] | np.ndarray) -> float:
    p = as_polygon(poly)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(poly: Sequence[Sequence[float]] | np.ndarray) -> Point:
    p = as_polygon(poly)
    return (float(p[:, 0].mean()), float(p[:, 1].mean()))


def box_polygon_overlap(b: Sequence[float], poly: Sequence[Sequence[float]] | np.ndarray, samples: int = 6) -> float:
    """Tỉ lệ (xấp xỉ) diện tích hộp nằm trong đa giác, bằng lưới điểm mẫu."""
    x1, y1, x2, y2 = as_xyxy(b)
    xs = np.linspace(x1, x2, samples)
    ys = np.linspace(y1, y2, samples)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return float(points_in_polygon(pts, poly).mean())


# --------------------------------------------------------------------- đoạn thẳng
def cross2(o: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    """Tích có hướng (a-o) x (b-o). >0: b nằm bên trái tia o→a (toạ độ ảnh: y hướng xuống)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def side_of_line(p: Sequence[float], a: Sequence[float], b: Sequence[float], eps: float = 1e-9) -> int:
    c = cross2(a, b, p)
    if c > eps:
        return 1
    if c < -eps:
        return -1
    return 0


def _on_segment(p: Sequence[float], a: Sequence[float], b: Sequence[float], eps: float = 1e-9) -> bool:
    if abs(cross2(a, b, p)) > eps * max(1.0, math.dist(a, b)):
        return False
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def segment_intersection(p1: Sequence[float], p2: Sequence[float], q1: Sequence[float],
                         q2: Sequence[float]) -> tuple[bool, float, float]:
    """Giao của đoạn p1p2 và q1q2. Trả (có giao?, t trên p, u trên q) với giao = p1 + t(p2-p1)."""
    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (q2[0] - q1[0], q2[1] - q1[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return False, 0.0, 0.0
    qp = (q1[0] - p1[0], q1[1] - p1[1])
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    ok = (0.0 <= t <= 1.0) and (0.0 <= u <= 1.0)
    return ok, float(t), float(u)


def line_crossing(prev: Sequence[float], curr: Sequence[float], a: Sequence[float], b: Sequence[float],
                  margin: float = 0.0) -> tuple[int, float]:
    """Kiểm tra quỹ đạo prev→curr có cắt vạch ab không.

    Trả (hướng, t): hướng = +1 nếu đi từ bên phải sang bên trái của tia a→b (cross âm → dương),
    -1 nếu ngược lại, 0 nếu không cắt. t ∈ [0,1] là vị trí cắt trên đoạn prev→curr (để nội suy thời gian).
    margin: nới vạch về hai đầu theo tỉ lệ độ dài (bắt xe đi sát mép vạch).
    """
    if margin > 0:
        dx, dy = b[0] - a[0], b[1] - a[1]
        a = (a[0] - margin * dx, a[1] - margin * dy)
        b = (b[0] + margin * dx, b[1] + margin * dy)
    s0 = cross2(a, b, prev)
    s1 = cross2(a, b, curr)
    if (s0 > 0 and s1 > 0) or (s0 < 0 and s1 < 0) or (s0 == 0 and s1 == 0):
        return 0, 0.0
    ok, t, _u = segment_intersection(prev, curr, a, b)
    if not ok:
        return 0, 0.0
    direction = 1 if s1 > s0 else -1
    return direction, t


def distance_point_segment(p: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def polyline_crossing(prev: Sequence[float], curr: Sequence[float], line: Sequence[Sequence[float]]) -> int:
    """Cắt một đường gấp khúc (vạch liền nhiều đoạn)? Trả hướng của đoạn đầu tiên bị cắt hoặc 0."""
    pts = np.asarray(line, dtype=np.float64).reshape(-1, 2)
    for i in range(len(pts) - 1):
        d, _ = line_crossing(prev, curr, pts[i], pts[i + 1])
        if d != 0:
            return d
    return 0


# --------------------------------------------------------------------- vector
def norm(v: Sequence[float]) -> float:
    return float(math.hypot(v[0], v[1]))


def unit(v: Sequence[float]) -> np.ndarray:
    n = norm(v)
    if n < 1e-12:
        return np.zeros(2)
    return np.asarray([v[0] / n, v[1] / n], dtype=np.float64)


def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    nu, nv = norm(u), norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float((u[0] * v[0] + u[1] * v[1]) / (nu * nv))


def angle_deg(u: Sequence[float], v: Sequence[float]) -> float:
    c = max(-1.0, min(1.0, cosine_similarity(u, v)))
    return math.degrees(math.acos(c))


def scale_points(pts: Sequence[Sequence[float]], sx: float, sy: float) -> list[list[float]]:
    return [[float(p[0]) * sx, float(p[1]) * sy] for p in pts]
