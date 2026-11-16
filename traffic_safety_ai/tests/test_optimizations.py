"""Kiểm thử các tối ưu rút từ khảo sát GitHub: ROI/chia ô cho model nền, đếm có trễ, hồi quy 2 cột, giới hạn
tần suất model ảnh cắt."""

import numpy as np

from traffic_safety_ai.analytics.counting import LineCounter
from traffic_safety_ai.core.attributes import _due
from traffic_safety_ai.core.config import CountingLineCfg, SiteConfig, ZoneCfg
from traffic_safety_ai.core.kinematics import robust_fit, robust_fit_xy
from traffic_safety_ai.core.regions import make_tiles, merge_tiled, scene_roi, shift
from traffic_safety_ai.core.types import Detection
from traffic_safety_ai.tests.helpers import World, box_at


# ------------------------------------------------------------------ vùng chạy model nền
def _scene_bottom_half() -> SiteConfig:
    sc = SiteConfig()
    sc.zones.lanes.append(ZoneCfg(id="L", polygon=[(100, 400), (1180, 400), (1180, 700), (100, 700)],
                                  direction=[(600, 690), (600, 410)]))
    return sc


def test_scene_roi_auto_covers_zones_with_margin():
    roi = scene_roi(_scene_bottom_half(), 1280, 720)
    assert roi is not None
    x1, y1, x2, y2 = roi
    assert y1 <= 400 - 0.15 * 720 + 1 and y2 == 720 and x1 == 0 and x2 == 1280


def test_scene_roi_full_when_nothing_declared_or_pov_or_too_big():
    assert scene_roi(SiteConfig(), 1280, 720) is None
    sc = _scene_bottom_half()
    sc.camera.channel = "pov"
    assert scene_roi(sc, 1280, 720) is None
    big = SiteConfig()
    big.zones.roadway.append(ZoneCfg(id="R", polygon=[(0, 50), (1280, 50), (1280, 720), (0, 720)]))
    assert scene_roi(big, 1280, 720) is None
    assert scene_roi(_scene_bottom_half(), 1280, 720, spec="full") is None
    assert scene_roi(SiteConfig(), 1280, 720, spec=[10, 20, 500, 400]) == (10, 20, 500, 400)


def test_tiles_cover_roi_with_overlap():
    tiles = make_tiles((0, 300, 1200, 720), 2, 1, overlap=0.2)
    assert len(tiles) == 2
    (a1, _, a2, _), (b1, _, b2, _) = tiles
    assert a1 == 0 and b2 == 1200 and b1 < a2          # có phần chồng lấn


def _d(box, conf, cls="car"):
    return Detection(np.array(box, float), conf, cls)


def test_merge_tiled_prefers_full_box_over_tile_fragment():
    full = _d([100, 100, 200, 160], 0.6)
    fragment = _d([150, 100, 200, 160], 0.8)               # ô cắt ngang xe: chỉ thấy nửa sau
    other = _d([400, 100, 480, 160], 0.7)
    out = merge_tiled([full, fragment, other])
    assert len(out) == 2
    kept = min(out, key=lambda d: d.xyxy[0])
    assert kept.xyxy[0] == 100 and kept.conf == 0.8


def test_shift_moves_boxes_back_to_frame():
    out = shift([_d([10, 10, 50, 40], 0.9)], 100, 300, 1280, 720)
    assert out[0].xyxy.tolist() == [110, 310, 150, 340]


# ------------------------------------------------------------------ đếm có trễ
def _counter_world():
    sc = SiteConfig()
    sc.lines.counting.append(CountingLineCfg(id="L", name="L", points=[(0, 200), (400, 200)],
                                             direction=[(200, 100), (200, 300)]))
    return sc, World(sc, calib=False)


def test_jitter_on_line_is_not_counted():
    sc, w = _counter_world()
    lc = LineCounter(sc)
    for i in range(90):                                     # xe máy đứng đè vạch, điểm chân rung ±1 px
        y_px = 200 + (1 if i % 2 else -1)
        lc.update(w.step({1: ("motorcycle", box_at(5, y_px / 20.0, 20, 40))}))
    lc.finalize()
    assert sum(lc.totals.values()) in (0, 1)              # tối đa 1 lượt (khi hết video vẫn đang đè vạch)
    sc2, w2 = _counter_world()
    lc2 = LineCounter(sc2)
    for i in range(90):
        y_px = 200 + (1 if i % 2 else -1)
        lc2.update(w2.step({1: ("motorcycle", box_at(5, y_px / 20.0, 20, 40))}))
    assert sum(lc2.totals.values()) == 0                  # trước khi kết thúc: chưa đếm gì


def test_real_crossing_counted_once_with_direction():
    sc, w = _counter_world()
    lc = LineCounter(sc)
    for i in range(60):
        lc.update(w.step({1: ("car", box_at(5, 8.0 + i * 0.08, 40, 30))}))   # đi từ y=160 px xuống y≈254 px
    assert lc.totals[("L", "car", "Chiều thuận")] == 1 and sum(lc.totals.values()) == 1


def test_pending_crossing_counted_when_track_lost():
    sc, w = _counter_world()
    lc = LineCounter(sc)
    for i in range(12):                                    # vừa nhích qua vạch thì bị che mất
        lc.update(w.step({1: ("car", box_at(5, 9.6 + i * 0.04, 40, 30))}))
    assert sum(lc.totals.values()) == 0
    lc.forget(1)
    assert sum(lc.totals.values()) == 1


# ------------------------------------------------------------------ hồi quy 2 cột
def test_robust_fit_xy_matches_single_column():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 25)
    x = 3.0 + 12.0 * t + rng.normal(0, 0.2, len(t))
    y = -1.0 - 4.0 * t + rng.normal(0, 0.2, len(t))
    y[7] += 30.0                                           # ngoại lai ở cột y
    (bx, sx, rx), (by, sy, ry) = robust_fit_xy(t, x, y)
    ex, ey = robust_fit(t, x), robust_fit(t, y)
    assert np.allclose([bx, sx, rx], ex) and np.allclose([by, sy, ry], ey)
    assert abs(by + 4.0) < 0.5


# ------------------------------------------------------------------ giới hạn tần suất
class _T:
    def __init__(self):
        self.attrs = {}


def test_due_limits_rate_per_object():
    tr = _T()
    hits = [t for t in np.arange(0, 1.0, 1 / 30) if _due(tr, "_k", float(t), 6.0)]
    assert 5 <= len(hits) <= 7
    tr2 = _T()
    assert all(_due(tr2, "_k", float(t), 0.0) for t in np.arange(0, 1.0, 1 / 30))
