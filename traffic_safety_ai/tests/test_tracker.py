from collections import Counter, defaultdict

import numpy as np

from traffic_safety_ai.core.tracker import ByteTracker, TrackerConfig
from traffic_safety_ai.core.types import Detection


def _box(cx, cy, w, h):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _run(objs, frames=120, miss_every=0, noise=1.0, seed=0, warp_shift=0.0):
    rng = np.random.default_rng(seed)
    trk = ByteTracker(TrackerConfig(), fps=30)
    assign = defaultdict(Counter)
    for f in range(frames):
        dets, gt = [], []
        for k, (x0, y0, vx, vy, w, h, cls) in enumerate(objs):
            cx, cy = x0 + vx * f + warp_shift * f, y0 + vy * f
            if miss_every and (f + k) % miss_every == 0 and f > 3:
                continue  # detector bỏ sót
            d = Detection(_box(cx + rng.normal(0, noise), cy + rng.normal(0, noise), w, h),
                          0.85 if f % 7 else 0.3, cls)
            dets.append(d)
            gt.append(k)
        warp = None
        if warp_shift:
            warp = np.array([[1.0, 0, warp_shift], [0, 1.0, 0]])
        out = trk.update(dets, f, f / 30.0, warp=warp)
        for tr in out:
            # gán theo IoU lớn nhất với GT
            best = max(range(len(dets)), key=lambda j: _iou(tr.xyxy, dets[j].xyxy))
            assign[gt[best]][tr.id] += 1
    return trk, assign


def _iou(a, b):
    from traffic_safety_ai.utils.geometry import iou
    return iou(a, b)


def test_ids_stable_with_missed_and_low_score_detections():
    objs = [(100, 100, 3, 0, 40, 80, "person"), (100, 300, 5, 1, 60, 60, "motorcycle"),
            (600, 200, -4, 0, 120, 80, "car")]
    _, assign = _run(objs, miss_every=9)
    for k in range(3):
        top_id, n = assign[k].most_common(1)[0]
        assert n / sum(assign[k].values()) > 0.95, assign[k]


def test_crossing_person_and_motorcycle_not_swapped():
    objs = [(100, 200, 4, 0, 40, 90, "person"), (500, 205, -4, 0, 60, 70, "motorcycle")]
    _, assign = _run(objs, frames=100)
    ids0 = set(assign[0])
    ids1 = set(assign[1])
    assert not (ids0 & ids1), "người và xe máy không được dùng chung ID"


def test_lost_tracks_are_removed_and_memory_bounded():
    trk = ByteTracker(TrackerConfig(track_buffer_s=0.5), fps=30)
    rng = np.random.default_rng(3)
    for f in range(1500):
        dets = []
        # đối tượng xuất hiện ngắn rồi biến mất liên tục
        for k in range(5):
            if (f // 40 + k) % 2 == 0:
                cx = 50 + (f % 40) * 5 + k * 150
                dets.append(Detection(_box(cx, 100 + k * 60, 40, 40), 0.9, "car"))
        trk.update(dets, f, f / 30.0)
        trk.pop_removed()
        assert len(trk.tracks) < 40
    assert rng is not None


def test_gmc_warp_keeps_identity_under_camera_pan():
    objs = [(300, 200, 0, 0, 50, 50, "car"), (500, 260, 0, 0, 50, 50, "car")]
    _, assign = _run(objs, frames=80, warp_shift=12.0, noise=0.5)
    for k in range(2):
        top_id, n = assign[k].most_common(1)[0]
        assert n / sum(assign[k].values()) > 0.95
