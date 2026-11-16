"""Chọn vùng ảnh đưa vào model nền: cắt theo vùng quan tâm (ROI) và chia ô (tiling) cho vật nhỏ ở xa.

Ý tưởng tham khảo (viết lại, không chép mã):
  • SAHI (obss/sahi) — suy luận trên các ô chồng lấn rồi gộp kết quả, giúp bắt vật nhỏ mà ảnh thu nhỏ làm mất.
  • Thói quen của các hệ đếm xe (supervision, ultralytics solutions) — chỉ xử lý vùng có vạch/vùng khai báo.

Vì sao có lợi:
  • Camera cổng trường thường chỉ cần nửa dưới khung (lòng đường gần). Cắt bỏ trời, mái nhà, cây → ảnh đưa vào
    model nhỏ hơn (Ultralytics dùng letterbox hình chữ nhật) nên NHANH hơn, đồng thời cùng imgsz thì vật trong vùng
    cắt TO hơn → bắt xe máy ở xa tốt hơn.
  • Chia ô: mỗi ô được phóng to lên imgsz → xe máy chỉ 15–20 px ở xa vẫn nhận được; tốn thời gian tỉ lệ số ô.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from traffic_safety_ai.core.types import Detection

Rect = tuple[int, int, int, int]


def _scene_points(scene) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    z = scene.zones
    for group in (z.sidewalks, z.crosswalks, z.lanes, z.restricted, z.no_parking, z.sightline, z.roadway,
                  z.speed_zone):
        for zone in group:
            pts.extend((float(x), float(y)) for x, y in zone.polygon)
    ln = scene.lines
    for group in (ln.stop_lines, ln.solid_lines, ln.counting, ln.speed_traps):
        for line in group:
            for attr in ("points", "line_a", "line_b", "confirm_points"):
                v = getattr(line, attr, None)
                if v:
                    pts.extend((float(x), float(y)) for x, y in v)
    return pts


def scene_roi(scene, frame_w: int, frame_h: int, margin: float = 0.15, max_frac: float = 0.8,
              spec: str | Sequence[int] | None = "auto") -> Rect | None:
    """Hình chữ nhật cần chạy model nền; None = cả khung.

    spec: "auto" (bao các vùng/vạch đã khai báo + lề), "full"/None (cả khung) hoặc [x1, y1, x2, y2] tự đặt.
    Lề mặc định 15% kích thước khung mỗi phía để xe được theo dõi TRƯỚC khi chạm vùng/vạch (RB-05 cần quỹ đạo
    trước vạch dừng, RB-03 cần 1 s dịch chuyển...). Nếu vùng cắt vẫn chiếm > max_frac diện tích thì dùng cả khung.
    """
    if spec is None or (isinstance(spec, str) and spec.lower() in ("full", "none", "off", "")):
        return None
    if not isinstance(spec, str):
        x1, y1, x2, y2 = (int(v) for v in spec)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        return (x1, y1, x2, y2) if x2 - x1 >= 32 and y2 - y1 >= 32 else None
    if getattr(scene.camera, "channel", "cctv") != "cctv":
        return None                                    # POV: vật xuất hiện ở mọi chỗ, luôn dùng cả khung
    pts = _scene_points(scene)
    if not pts:
        return None
    a = np.asarray(pts, dtype=np.float64)
    mx, my = margin * frame_w, margin * frame_h
    x1 = int(max(0, np.floor(a[:, 0].min() - mx)))
    y1 = int(max(0, np.floor(a[:, 1].min() - my)))
    x2 = int(min(frame_w, np.ceil(a[:, 0].max() + mx)))
    y2 = int(min(frame_h, np.ceil(a[:, 1].max() + my)))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    if (x2 - x1) * (y2 - y1) > max_frac * frame_w * frame_h:
        return None
    return x1, y1, x2, y2


def make_tiles(rect: Rect, cols: int, rows: int, overlap: float = 0.2) -> list[Rect]:
    """Chia rect thành cols×rows ô chồng lấn `overlap` (tỉ lệ cạnh ô)."""
    x1, y1, x2, y2 = rect
    cols, rows = max(1, int(cols)), max(1, int(rows))
    W, H = x2 - x1, y2 - y1
    tw = W / (cols - (cols - 1) * overlap)
    th = H / (rows - (rows - 1) * overlap)
    out: list[Rect] = []
    for r in range(rows):
        for c in range(cols):
            ox = x1 + c * tw * (1 - overlap)
            oy = y1 + r * th * (1 - overlap)
            out.append((int(round(ox)), int(round(oy)), int(round(min(x2, ox + tw))), int(round(min(y2, oy + th)))))
    return out


def _iou_ios(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + ab - inter), inter / min(aa, ab)


def merge_tiled(dets: Iterable[Detection], iou_thr: float = 0.5, ios_thr: float = 0.75) -> list[Detection]:
    """Gộp kết quả nhiều ô + lượt cả vùng.

    Hai hộp cùng lớp là một vật khi IoU ≥ iou_thr, HOẶC hộp nhỏ nằm gần trọn trong hộp lớn (IoS ≥ ios_thr) —
    trường hợp ô cắt ngang qua xe chỉ thấy nửa xe. Khi đó giữ HỘP LỚN (đủ xe) và điểm tin cao hơn.
    """
    items = sorted(dets, key=lambda d: -d.conf)
    kept: list[Detection] = []
    for d in items:
        merged = False
        for i, k in enumerate(kept):
            if k.cls != d.cls:
                continue
            iou, ios = _iou_ios(k.xyxy, d.xyxy)
            if iou >= iou_thr or ios >= ios_thr:
                ka = (k.xyxy[2] - k.xyxy[0]) * (k.xyxy[3] - k.xyxy[1])
                da = (d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1])
                if ios >= ios_thr and da > 1.3 * ka:     # hộp mới là xe đầy đủ, hộp cũ chỉ là mảnh ở mép ô
                    kept[i] = Detection(d.xyxy.copy(), max(k.conf, d.conf), d.cls, d.source, dict(d.attrs))
                merged = True
                break
        if not merged:
            kept.append(d)
    return kept


def shift(dets: Iterable[Detection], dx: int, dy: int, clip_w: int, clip_h: int) -> list[Detection]:
    out = []
    for d in dets:
        b = d.xyxy.astype(np.float64) + np.array([dx, dy, dx, dy], dtype=np.float64)
        b[[0, 2]] = np.clip(b[[0, 2]], 0, clip_w - 1)
        b[[1, 3]] = np.clip(b[[1, 3]], 0, clip_h - 1)
        if b[2] - b[0] >= 2 and b[3] - b[1] >= 2:
            d.xyxy = b
            out.append(d)
    return out


def touches_inner_edge(d: Detection, tile: Rect, roi: Rect, pad: float = 2.0) -> bool:
    """Hộp chạm mép ô mà mép đó KHÔNG phải mép vùng → có thể chỉ là mảnh xe (dùng để hạ ưu tiên)."""
    x1, y1, x2, y2 = d.xyxy
    tx1, ty1, tx2, ty2 = tile
    rx1, ry1, rx2, ry2 = roi
    return ((x1 <= tx1 + pad and tx1 > rx1) or (y1 <= ty1 + pad and ty1 > ry1)
            or (x2 >= tx2 - pad and tx2 < rx2) or (y2 >= ty2 - pad and ty2 < ry2))
