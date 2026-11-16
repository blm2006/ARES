import math

import numpy as np
import pytest

from traffic_safety_ai.core.homography import CalibrationError, GroundCalibration, rectangle_world_points
from traffic_safety_ai.core.kinematics import KinematicsEstimator, robust_fit
from traffic_safety_ai.core.types import Detection, Track

# Homography "thật" (mét → pixel) của một camera nhìn xiên xuống đường.
H_TRUE = np.array([[40.0, -12.0, 500.0], [0.0, -8.0, 900.0], [0.0, -0.012, 1.0]])


def world_to_img(pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H_TRUE.T
    return p[:, :2] / p[:, 2:3]


def make_calib():
    wp = rectangle_world_points(7.0, 20.0)
    ip = world_to_img(wp)
    return GroundCalibration(ip, wp)


def test_calibration_roundtrip():
    cal = make_calib()
    assert cal.reproj_error_m < 1e-4 and cal.ok
    rng = np.random.default_rng(1)
    w = rng.uniform([0, 0], [7, 20], size=(50, 2))
    back = cal.to_world(world_to_img(w))
    assert np.abs(back - w).max() < 1e-4
    d = cal.distance_m(world_to_img([[0, 0]])[0], world_to_img([[0, 10]])[0])
    assert d == pytest.approx(10.0, abs=1e-4)
    assert cal.in_calibrated_area(world_to_img([[3.5, 10]])[0])
    assert not cal.in_calibrated_area(world_to_img([[3.5, 60]])[0], margin=0.0)


def test_calibration_rejects_degenerate():
    with pytest.raises(CalibrationError):
        GroundCalibration([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 0), (1, 0), (1, 1), (0, 1)])


def test_robust_fit_rejects_outlier():
    t = np.linspace(0, 1, 31)
    v = 2.0 + 5.0 * t
    v[10] += 50  # ngoại lai
    b, sig, _ = robust_fit(t, v)
    assert b == pytest.approx(5.0, rel=0.01)


def _drive(cal, speed_mps, noise_px=0.0, decel=0.0, fps=30.0, seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    tr = None
    est = KinematicsEstimator()
    for i in range(int(seconds * fps)):
        t = i / fps
        y = 2.0 + speed_mps * t - 0.5 * decel * t * t
        g = world_to_img([[3.0, y]])[0]
        g = g + rng.normal(0, noise_px, 2)
        box = np.array([g[0] - 20, g[1] - 60, g[0] + 20, g[1]])
        d = Detection(box, 0.9, "car")
        if tr is None:
            tr = Track(1, d, i, t)
            from traffic_safety_ai.core.types import TrackPoint
            tr.history.append(TrackPoint(i, t, box, 0.9, d.bottom, cal.point_to_world(d.bottom)))
        else:
            from traffic_safety_ai.core.types import TrackPoint
            tr.history.append(TrackPoint(i, t, box, 0.9, d.bottom, cal.point_to_world(d.bottom)))
            tr.last_t = t
        est.update(tr)
    return tr


def test_speed_estimate_clean_and_noisy():
    cal = make_calib()
    tr = _drive(cal, 10.0)
    assert tr.kin.speed_kmh == pytest.approx(36.0, rel=0.01)
    assert tr.kin.moving
    tr2 = _drive(cal, 10.0, noise_px=1.5)
    assert tr2.kin.speed_kmh == pytest.approx(36.0, rel=0.08)
    assert tr2.kin.speed_sigma_kmh is not None and tr2.kin.speed_sigma_kmh < 5


def test_deceleration_detected():
    cal = make_calib()
    tr = _drive(cal, 12.0, decel=5.0, seconds=2.0)
    assert tr.kin.accel_ms2 is not None and tr.kin.accel_ms2 == pytest.approx(-5.0, abs=1.0)


def test_scale_rate_positive_when_growing():
    tr = None
    est = KinematicsEstimator()
    from traffic_safety_ai.core.types import TrackPoint
    for i in range(30):
        t = i / 30
        h = 50 * math.exp(0.8 * t)
        box = np.array([100, 300 - h, 100 + h * 0.8, 300])
        d = Detection(box, 0.9, "car")
        if tr is None:
            tr = Track(1, d, i, t)
        tr.history.append(TrackPoint(i, t, box, 0.9, d.bottom, None))
        est.update(tr)
    assert tr.kin.scale_rate == pytest.approx(0.8, rel=0.05)
