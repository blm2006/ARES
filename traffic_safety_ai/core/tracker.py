"""Multi-object tracker kiểu ByteTrack (tự viết lại theo ý tưởng bài báo, không chép mã).

Ý tưởng chính của ByteTrack: ghép hai vòng — detection tin cậy cao trước, rồi dùng cả detection
tin cậy THẤP để cứu những track đang bị che khuất (thay vì vứt bỏ như SORT).

Bổ sung so với bản gốc:
  * Ghép có xét lớp (xe máy không ghép với người; cho phép xe máy↔xe đạp, ô tô↔tải↔buýt vì model hay nhầm).
  * Nhận ma trận bù chuyển động camera (GMC, kiểu BoT-SORT) — bắt buộc cho góc nhìn POV.
  * Lưu lịch sử bằng deque(maxlen) và tự xoá track mất quá lâu → không rò rỉ RAM khi chạy nhiều giờ.
  * Lớp của track = bỏ phiếu có trọng số theo confidence (ổn định khi model dao động giữa các lớp).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from traffic_safety_ai.core.kalman import KalmanXYWH, xywh_to_xyxy, xyxy_to_xywh
from traffic_safety_ai.core.types import Detection, Track, TrackPoint, classes_compatible
from traffic_safety_ai.utils.geometry import iou_matrix

try:  # scipy có sẵn khi cài ultralytics
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None

_BIG = 1e6


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.55
    match_thresh: float = 0.8            # chi phí tối đa vòng 1 (1 - IoU*score)
    second_match_thresh: float = 0.5     # vòng 2 với detection điểm thấp (1 - IoU)
    unconfirmed_match_thresh: float = 0.7
    track_buffer_s: float = 1.5          # giữ track bị mất tối đa bao lâu (giây)
    min_hits: int = 2
    history_len: int = 90
    fuse_score: bool = True
    class_aware: bool = True


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian (scipy) hoặc tham lam nếu thiếu scipy. Bỏ các cặp có chi phí > thresh."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    matches: list[tuple[int, int]] = []
    if _lsa is not None:
        c = np.where(cost > thresh, thresh + _BIG, cost)
        rows, cols = _lsa(c)
        for r, k in zip(rows, cols):
            if cost[r, k] <= thresh:
                matches.append((int(r), int(k)))
    else:  # pragma: no cover - chỉ khi không có scipy
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        used_r, used_c = set(), set()
        for r, k in order:
            if cost[r, k] > thresh:
                break
            if r in used_r or k in used_c:
                continue
            used_r.add(int(r))
            used_c.add(int(k))
            matches.append((int(r), int(k)))
    mr = {r for r, _ in matches}
    mc = {k for _, k in matches}
    return matches, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]


class ByteTracker:
    def __init__(self, cfg: TrackerConfig | None = None, fps: float = 30.0):
        self.cfg = cfg or TrackerConfig()
        self.fps = max(1.0, float(fps))
        self.kf = KalmanXYWH()
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self._last_frame: int | None = None
        self._last_t: float | None = None
        self._frames_seen = 0
        self._removed: list[Track] = []

    # ------------------------------------------------------------------ public
    @property
    def max_lost_frames(self) -> int:
        return max(1, int(round(self.cfg.track_buffer_s * self.fps)))

    def pop_removed(self) -> list[Track]:
        out, self._removed = self._removed, []
        return out

    def active(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.state == "confirmed" and t.time_since_update == 0]

    def update(self, dets: list[Detection], frame_idx: int, t: float, warp: np.ndarray | None = None,
               to_world: Callable[[tuple[float, float]], tuple[float, float]] | None = None) -> list[Track]:
        cfg = self.cfg
        steps = 1 if self._last_frame is None else max(1, frame_idx - self._last_frame)
        if self._last_t is not None and t > self._last_t:
            # nguồn trực tiếp (RTSP) bỏ khung khi máy chậm: số thứ tự khung vẫn liền nhau nhưng thời gian nhảy →
            # dự đoán Kalman và hạn giữ track mất dấu phải theo thời gian thật, không theo số khung
            steps = max(steps, int(round((t - self._last_t) * self.fps)))
        self._last_frame = frame_idx
        self._last_t = t
        self._frames_seen += 1

        # 1) dự đoán
        for tr in self.tracks.values():
            tr.mean, tr.cov = self.kf.predict(tr.mean, tr.cov, float(steps))
            if warp is not None:
                tr.mean, tr.cov = self.kf.apply_affine(tr.mean, tr.cov, warp)

        high = [d for d in dets if d.conf >= cfg.high_thresh]
        low = [d for d in dets if cfg.low_thresh <= d.conf < cfg.high_thresh]
        pool = [tr for tr in self.tracks.values() if tr.state in ("confirmed", "lost")]
        unconfirmed = [tr for tr in self.tracks.values() if tr.state == "tentative"]
        updated: set[int] = set()

        # 2) vòng 1: track đã xác nhận / đang mất ↔ detection điểm cao
        cost = self._cost(pool, high, fuse=cfg.fuse_score)
        m1, u_tr, u_det = linear_assignment(cost, cfg.match_thresh)
        for i, j in m1:
            self._apply(pool[i], high[j], frame_idx, t, to_world)
            pool[i].state = "confirmed"
            updated.add(pool[i].id)
        remaining_high = [high[j] for j in u_det]

        # 3) vòng 2: track còn lại (đang theo, chưa mất) ↔ detection điểm thấp
        r_tracks = [pool[i] for i in u_tr if pool[i].state == "confirmed"]
        cost2 = self._cost(r_tracks, low, fuse=False)
        m2, u_tr2, _ = linear_assignment(cost2, cfg.second_match_thresh)
        for i, j in m2:
            self._apply(r_tracks[i], low[j], frame_idx, t, to_world)
            updated.add(r_tracks[i].id)
        for i in u_tr2:
            r_tracks[i].state = "lost"
        for tr in pool:
            if tr.id not in updated:
                tr.time_since_update += steps
                if tr.state == "confirmed":
                    tr.state = "lost"

        # 4) track mới chưa xác nhận ↔ detection điểm cao còn lại
        cost3 = self._cost(unconfirmed, remaining_high, fuse=cfg.fuse_score)
        m3, u_unc, u_det3 = linear_assignment(cost3, cfg.unconfirmed_match_thresh)
        for i, j in m3:
            tr = unconfirmed[i]
            self._apply(tr, remaining_high[j], frame_idx, t, to_world)
            updated.add(tr.id)
            if tr.hits >= cfg.min_hits:
                tr.state = "confirmed"
        for i in u_unc:
            self._remove(unconfirmed[i])

        # 5) sinh track mới
        for j in u_det3:
            d = remaining_high[j]
            if d.conf < cfg.new_track_thresh:
                continue
            tr = Track(self._next_id, d, frame_idx, t, cfg.history_len)
            self._next_id += 1
            tr.mean, tr.cov = self.kf.initiate(xyxy_to_xywh(d.xyxy))
            tr.history.append(self._point(d, frame_idx, t, to_world))
            # frame đầu tiên của video: xác nhận ngay (như ByteTrack) để không mất đối tượng đã có sẵn
            tr.state = "confirmed" if (self._frames_seen == 1 or cfg.min_hits <= 1) else "tentative"
            self.tracks[tr.id] = tr
            updated.add(tr.id)

        # 6) xoá track mất quá lâu
        for tr in list(self.tracks.values()):
            if tr.state == "lost" and tr.time_since_update > self.max_lost_frames:
                self._remove(tr)

        return [tr for tr in self.tracks.values() if tr.state == "confirmed" and tr.id in updated]

    def predicted_box(self, tr: Track) -> np.ndarray:
        return xywh_to_xyxy(tr.mean) if tr.mean is not None else tr.xyxy

    # ------------------------------------------------------------------ nội bộ
    def _cost(self, tracks: list[Track], dets: list[Detection], fuse: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        tb = np.stack([self.predicted_box(t) for t in tracks])
        db = np.stack([d.xyxy for d in dets])
        ious = iou_matrix(tb, db)
        if fuse:
            scores = np.array([d.conf for d in dets])[None, :]
            cost = 1.0 - ious * scores
        else:
            cost = 1.0 - ious
        if self.cfg.class_aware:
            for i, tr in enumerate(tracks):
                c = tr.cls
                for j, d in enumerate(dets):
                    if not classes_compatible(c, d.cls):
                        cost[i, j] = _BIG
        return cost

    @staticmethod
    def _point(d: Detection, frame_idx: int, t: float,
               to_world: Callable[[tuple[float, float]], tuple[float, float]] | None) -> TrackPoint:
        g = d.bottom
        gm = None
        if to_world is not None:
            try:
                gm = to_world(g)
            except Exception:
                gm = None
        return TrackPoint(frame_idx, t, d.xyxy.copy(), d.conf, g, gm)

    def _apply(self, tr: Track, d: Detection, frame_idx: int, t: float,
               to_world: Callable[[tuple[float, float]], tuple[float, float]] | None) -> None:
        tr.mean, tr.cov = self.kf.update(tr.mean, tr.cov, xyxy_to_xywh(d.xyxy))
        tr.xyxy = d.xyxy.copy()
        tr.conf = d.conf
        tr.det_attrs = dict(d.attrs)
        tr.cls_votes[d.cls] += d.conf
        tr.hits += 1
        tr.time_since_update = 0
        tr.last_frame = frame_idx
        tr.last_t = t
        tr.history.append(self._point(d, frame_idx, t, to_world))

    def _remove(self, tr: Track) -> None:
        tr.state = "removed"
        self.tracks.pop(tr.id, None)
        self._removed.append(tr)
