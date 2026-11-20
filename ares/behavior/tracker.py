"""Bám vết người đơn giản (IoU + khoảng cách tâm), tự viết, mỗi camera một bộ.

Không dùng tracker của Ultralytics vì ở đây MỘT model pose chạy theo lô cho nhiều camera; tracker của
Ultralytics giữ trạng thái theo model nên không tách được camera. Người nằm thường bị model bỏ sót vài khung,
nên track đang "nằm/gục" được giữ lâu hơn (keep_lying_s) để bộ đếm bất động không bị reset.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

HIST_S = 40.0  # giữ lịch sử ~40 giây / người (đủ cho ngưỡng bất động 30 s + clip)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


@dataclass
class Sample:
    t: float
    kp: np.ndarray  # (17,3) pixel
    box: np.ndarray  # (4,)
    conf: float


@dataclass
class Track:
    id: int
    hist: deque = field(default_factory=deque)
    last_t: float = 0.0
    first_t: float = 0.0
    hits: int = 0
    _sizes: deque = field(default_factory=lambda: deque(maxlen=150))
    state: dict = field(default_factory=dict)  # trạng thái của luật (rules.py)

    @property
    def last(self) -> Sample:
        return self.hist[-1]

    def add(self, s: Sample) -> None:
        if not self.hist:
            self.first_t = s.t
        self.hist.append(s)
        self.last_t = s.t
        self.hits += 1
        w, h = s.box[2] - s.box[0], s.box[3] - s.box[1]
        self._sizes.append(float(max(w, h)))
        while self.hist and s.t - self.hist[0].t > HIST_S:
            self.hist.popleft()

    @property
    def ref_h(self) -> float:
        """Chiều cao tham chiếu: phân vị 80 của cạnh dài khung người gần đây (đứng hay nằm đều ~ chiều cao người)."""
        if not self._sizes:
            return 100.0
        return float(np.percentile(np.asarray(self._sizes), 80))

    def window(self, t0: float, t1: float) -> list[Sample]:
        return [s for s in self.hist if t0 <= s.t <= t1]


def _inter_min(a: np.ndarray, b: np.ndarray) -> float:
    """Diện tích giao / diện tích khung NHỎ hơn: 1.0 = khung nhỏ nằm gọn trong khung lớn (khung trùng của cùng 1 người)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / m) if m > 0 else 0.0


def _in_poly(x: float, y: float, poly) -> bool:
    inside, n = False, len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1:
            inside = not inside
    return inside


class Tracker:
    """Bám vết + "cổng sinh" người mới (gate):

    * ``warmup_s`` giây đầu: ai thấy là nhận (người đã có sẵn trong khung lúc bắt đầu).
    * Sau đó người MỚI chỉ được sinh ở **rìa khung hình** (cách mép < ``edge`` × cạnh) hoặc trong **vùng cửa ra vào**
      (``entry_zones``, đa giác pixel) — tức là phải đi từ ngoài vào.
    * Khung người mới mọc ra ở GIỮA khung hình:
        - trùng phần lớn với 1 người đang có (khung vỡ đôi, 1 người thành 2) -> bỏ;
        - gần chỗ 1 người vừa mất dấu (bị che, model bỏ sót) trong ``recover_s`` giây -> nối lại ĐÚNG mã cũ;
        - còn lại -> chỉ là "ứng viên", không hiện, không đếm; phải xuất hiện liên tục ``confirm_s`` giây mới được nhận
          (vd người ngồi khuất sau bàn từ đầu rồi đứng lên). ``confirm_s = 0`` = không bao giờ nhận người mọc giữa khung.
    ``gate=False``: hành vi cũ (mọi khung chưa ghép đều thành người mới) — dùng khi tách dữ liệu train.
    """

    def __init__(self, keep_s: float = 2.0, keep_lying_s: float = 15.0, gate: bool = False, warmup_s: float = 2.0,
                 edge: float = 0.06, confirm_s: float = 4.0, recover_s: float = 6.0, entry_zones=None):
        self.keep_s, self.keep_lying_s = keep_s, keep_lying_s
        self.gate, self.warmup_s, self.edge, self.confirm_s = gate, warmup_s, edge, confirm_s
        self.recover_s = recover_s if gate else 0.0
        self.entry_zones = entry_zones or []  # [[(x,y),...], ...] pixel
        self.tracks: dict[int, Track] = {}
        self.pending: list[Track] = []  # ứng viên mọc giữa khung, chưa được nhận
        self._next = 1
        self.t0: float | None = None
        self.frame_wh: tuple[int, int] | None = None
        self.stats = {"born_edge": 0, "born_warmup": 0, "born_entry": 0, "born_confirmed": 0, "dup_dropped": 0,
                      "recovered": 0, "pending_dropped": 0}

    def _new(self, t: float, kp, box, conf) -> Track:
        tr = Track(self._next)
        self._next += 1
        tr.add(Sample(t, kp.astype(np.float32), box.astype(np.float32), float(conf)))
        self.tracks[tr.id] = tr
        return tr

    def birth_ok(self, t: float, box) -> str:
        """'' = không được sinh ở đây; còn lại = lý do được sinh (warmup | edge | entry)."""
        if not self.gate or self.t0 is None or t - self.t0 <= self.warmup_s:
            return "warmup"
        if self.frame_wh is None:
            return "edge"  # không biết cỡ khung (test bơm keypoint) -> không xét rìa
        W, H = self.frame_wh
        mx, my = self.edge * W, self.edge * H
        if box[0] <= mx or box[1] <= my or box[2] >= W - mx or box[3] >= H - my:
            return "edge"
        fx, fy = (box[0] + box[2]) / 2, box[3]
        if any(_in_poly(fx, fy, z) or _in_poly((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, z) for z in self.entry_zones):
            return "entry"
        return ""

    def update(self, t: float, kps: np.ndarray, boxes: np.ndarray, confs: np.ndarray,
               frame_wh: tuple[int, int] | None = None) -> list[Track]:
        """kps (n,17,3), boxes (n,4), confs (n,) -> các track được cập nhật ở khung này.
        frame_wh = (rộng, cao) của ảnh mà boxes thuộc về (để biết rìa khung hình)."""
        if self.t0 is None:
            self.t0 = t
        if frame_wh is not None:
            self.frame_wh = (int(frame_wh[0]), int(frame_wh[1]))
        live = list(self.tracks.values())
        pairs = []
        for di in range(len(boxes)):
            c1 = ((boxes[di][0] + boxes[di][2]) / 2, (boxes[di][1] + boxes[di][3]) / 2)
            for tr in live:
                lb = tr.last.box
                ov = iou(boxes[di], lb)
                c0 = ((lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2)
                dist = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])) / max(tr.ref_h, 1.0)
                gap = t - tr.last_t
                lim = 0.35 + 0.5 * min(gap, 3.0 if self.gate else 2.0)
                if ov >= 0.15 or dist <= lim:
                    pairs.append((ov - 0.5 * dist - 0.1 * max(0.0, gap - 0.5), di, tr.id))
        pairs.sort(reverse=True)
        used_d, used_t, out = set(), set(), []
        for _, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            tr = self.tracks[tid]
            if t - tr.last_t > 0.6:
                self.stats["recovered"] += 1
            tr.add(Sample(t, kps[di].astype(np.float32), boxes[di].astype(np.float32), float(confs[di])))
            out.append(tr)
        fresh = [di for di in range(len(boxes)) if di not in used_d]
        if self.gate and fresh:
            cur = [tr.last.box for tr in out]
            keep = []
            for di in sorted(fresh, key=lambda d: -float(confs[d])):
                if t != self.t0 and any(_inter_min(boxes[di], b) >= 0.6 for b in cur):
                    self.stats["dup_dropped"] += 1  # khung trùng / vỡ đôi của người đang có
                    continue
                keep.append(di)
                cur.append(boxes[di])
            fresh = keep
        # ứng viên giữa khung đang chờ: ghép tiếp
        matched_pending: set[int] = set()
        for di in list(fresh):
            why = self.birth_ok(t, boxes[di])
            if why:
                self.stats["born_" + why] += 1
                out.append(self._new(t, kps[di], boxes[di], confs[di]))
                continue
            best, bd = None, 1e9
            for k, pt in enumerate(self.pending):
                if k in matched_pending:
                    continue
                lb = pt.last.box
                d = float(np.hypot((boxes[di][0] + boxes[di][2] - lb[0] - lb[2]) / 2,
                                   (boxes[di][1] + boxes[di][3] - lb[1] - lb[3]) / 2)) / max(pt.ref_h, 1.0)
                if (iou(boxes[di], lb) >= 0.2 or d <= 0.5) and d < bd:
                    best, bd = k, d
            if best is None:
                pt = Track(0)
                pt.add(Sample(t, kps[di].astype(np.float32), boxes[di].astype(np.float32), float(confs[di])))
                self.pending.append(pt)
                matched_pending.add(len(self.pending) - 1)
                continue
            matched_pending.add(best)
            pt = self.pending[best]
            pt.add(Sample(t, kps[di].astype(np.float32), boxes[di].astype(np.float32), float(confs[di])))
            if self.confirm_s > 0 and t - pt.first_t >= self.confirm_s and pt.hits >= 3:
                tr = self._new(t, kps[di], boxes[di], confs[di])
                tr.hist = deque(pt.hist)  # giữ lịch sử để luật hành vi có dữ liệu ngay
                tr.first_t, tr.hits, tr._sizes = pt.first_t, pt.hits, pt._sizes
                self.pending[best] = None  # type: ignore[assignment]
                self.stats["born_confirmed"] += 1
                out.append(tr)
        alive = [p for p in self.pending if p is not None and t - p.last_t <= 0.6]
        self.stats["pending_dropped"] += sum(1 for p in self.pending if p is not None) - len(alive)
        self.pending = alive
        for tid, tr in list(self.tracks.items()):
            down = tr.state.get("posture_hold") in ("lying", "slumped")
            keep = self.keep_lying_s if down else max(self.keep_s, self.recover_s)
            if t - tr.last_t > keep:
                del self.tracks[tid]
        return out
