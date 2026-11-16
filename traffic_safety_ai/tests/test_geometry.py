import numpy as np
import pytest

from traffic_safety_ai.utils import geometry as g


def test_iou_ioa_basic():
    a = [0, 0, 10, 10]
    b = [5, 5, 15, 15]
    assert g.iou(a, b) == pytest.approx(25 / 175)
    assert g.ioa(a, b) == pytest.approx(0.25)
    assert g.iou(a, [20, 20, 30, 30]) == 0.0
    m = g.iou_matrix(np.array([a, b]), np.array([a, b, [100, 100, 101, 101]]))
    assert m.shape == (2, 3)
    assert m[0, 0] == pytest.approx(1.0) and m[1, 1] == pytest.approx(1.0) and m[0, 2] == 0.0


def test_point_in_polygon_and_vectorized_agree():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert g.point_in_polygon((5, 5), poly)
    assert not g.point_in_polygon((15, 5), poly)
    assert g.point_in_polygon((10, 5), poly)          # trên cạnh
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 15, size=(500, 2))
    vec = g.points_in_polygon(pts, poly)
    ref = np.array([g.point_in_polygon(p, poly) for p in pts])
    on_edge = (np.isclose(pts, 0) | np.isclose(pts, 10)).any(axis=1)
    assert (vec[~on_edge] == ref[~on_edge]).all()


def test_concave_polygon():
    poly = [(0, 0), (10, 0), (10, 10), (5, 4), (0, 10)]
    assert g.point_in_polygon((5, 2), poly)
    assert not g.point_in_polygon((5, 8), poly)


def test_line_crossing_direction_and_fraction():
    a, b = (0, 5), (10, 5)                      # vạch ngang y=5, tia hướng sang phải
    d, t = g.line_crossing((5, 0), (5, 10), a, b)
    assert d != 0 and t == pytest.approx(0.5)
    d2, _ = g.line_crossing((5, 10), (5, 0), a, b)
    assert d2 == -d
    assert g.line_crossing((5, 0), (5, 4), a, b)[0] == 0          # chưa tới vạch
    assert g.line_crossing((20, 0), (20, 10), a, b)[0] == 0       # ngoài đoạn vạch
    assert g.line_crossing((20, 0), (20, 10), a, b, margin=1.5)[0] != 0


def test_side_cosine_distance():
    assert g.side_of_line((0, -1), (0, 0), (1, 0)) == -g.side_of_line((0, 1), (0, 0), (1, 0))
    assert g.cosine_similarity((1, 0), (-1, 0)) == pytest.approx(-1)
    assert g.angle_deg((1, 0), (0, 1)) == pytest.approx(90)
    assert g.distance_point_segment((5, 3), (0, 0), (10, 0)) == pytest.approx(3)


def test_crop_and_expand():
    img = np.zeros((100, 200, 3), np.uint8)
    c, box = g.crop(img, [190, 90, 250, 150])
    assert c.shape[0] > 0 and box[2] <= 199
    e = g.expand_box([10, 10, 20, 30], top=0.5)
    assert e[1] == pytest.approx(0)
    assert g.box_polygon_overlap([0, 0, 10, 10], [(0, 0), (5, 0), (5, 10), (0, 10)]) == pytest.approx(0.5, abs=0.1)
