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


class Tracker:
    def __init__(self, keep_s: float = 2.0, keep_lying_s: float = 15.0):
        self.keep_s, self.keep_lying_s = keep_s, keep_lying_s
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, t: float, kps: np.ndarray, boxes: np.ndarray, confs: np.ndarray) -> list[Track]:
        """kps (n,17,3), boxes (n,4), confs (n,) -> các track được cập nhật ở khung này."""
        live = list(self.tracks.values())
        pairs = []
        for di in range(len(boxes)):
            for tr in live:
                lb = tr.last.box
                ov = iou(boxes[di], lb)
                c1 = ((boxes[di][0] + boxes[di][2]) / 2, (boxes[di][1] + boxes[di][3]) / 2)
                c0 = ((lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2)
                dist = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])) / max(tr.ref_h, 1.0)
                gap = t - tr.last_t
                if ov >= 0.15 or dist <= 0.35 + 0.5 * min(gap, 2.0):
                    pairs.append((ov - 0.5 * dist, di, tr.id))
        pairs.sort(reverse=True)
        used_d, used_t, out = set(), set(), []
        for _, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            tr = self.tracks[tid]
            tr.add(Sample(t, kps[di].astype(np.float32), boxes[di].astype(np.float32), float(confs[di])))
            out.append(tr)
        for di in range(len(boxes)):
            if di not in used_d:
                tr = Track(self._next)
                self._next += 1
                tr.add(Sample(t, kps[di].astype(np.float32), boxes[di].astype(np.float32), float(confs[di])))
                self.tracks[tr.id] = tr
                out.append(tr)
        for tid, tr in list(self.tracks.items()):
            down = tr.state.get("posture_hold") in ("lying", "slumped")
            if t - tr.last_t > (self.keep_lying_s if down else self.keep_s):
                del self.tracks[tid]
        return out
