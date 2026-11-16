import cv2
import numpy as np

from traffic_safety_ai.core.association import RiderAssociator, maybe_child
from traffic_safety_ai.core.ego_motion import EgoMotionEstimator
from traffic_safety_ai.core.kinematics import KinematicsEstimator
from traffic_safety_ai.core.traffic_light import LightStateTracker, classify_light_crop
from traffic_safety_ai.core.types import Detection, Track, TrackPoint


def _light(color: str, layout="vertical"):
    img = np.full((90, 30, 3), 25, np.uint8) if layout == "vertical" else np.full((30, 90, 3), 25, np.uint8)
    bgr = {"red": (30, 30, 240), "yellow": (0, 200, 255), "green": (120, 230, 40)}[color]
    if layout == "vertical":
        cy = {"red": 15, "yellow": 45, "green": 75}[color]
        cv2.circle(img, (15, cy), 11, bgr, -1)
    else:
        cx = {"red": 15, "yellow": 45, "green": 75}[color]
        cv2.circle(img, (cx, 15), 11, bgr, -1)
    return img


def test_light_classify_colors_and_off():
    for c in ("red", "yellow", "green"):
        assert classify_light_crop(_light(c))[0] == c
        assert classify_light_crop(_light(c, "horizontal"), "horizontal")[0] == c
    assert classify_light_crop(np.full((90, 30, 3), 25, np.uint8))[0] == "off"


def test_light_tracker_debounce_flicker_and_history():
    lt = LightStateTracker("L1", roi=None, min_consecutive=2, hold_s=0.5)
    t = 0.0
    seq = ["green"] * 10 + ["yellow"] * 6 + ["red"] * 4 + ["off"] + ["red"] * 10
    for c in seq:
        img = _light(c) if c != "off" else np.full((90, 30, 3), 25, np.uint8)
        st = lt.update_from_crop(img, t)
        t += 1 / 30
    assert st.state == "red"                       # khung "tắt" do PWM bị bỏ qua
    assert lt.state_at(0.2) == "green"
    t_red_start = 16 / 30
    assert abs(lt.red_since(t) - (t - t_red_start)) < 1e-6


def _track(tid, box, cls, vx=0.0, vy=0.0, n=10):
    tr = Track(tid, Detection(box, 0.9, cls), 0, 0.0)
    for i in range(n):
        tt = i / 30
        b = np.asarray(box, float) + np.array([vx * tt, vy * tt, vx * tt, vy * tt])
        tr.history.append(TrackPoint(i, tt, b, 0.9, ((b[0] + b[2]) / 2, b[3])))
        tr.xyxy = b
    KinematicsEstimator().update(tr)
    return tr


def test_rider_association_moving_together_and_pedestrian_rejected():
    moto = _track(1, [100, 200, 160, 260], "motorcycle", vy=60)
    driver = _track(2, [105, 130, 155, 250], "person", vy=60)
    passenger = _track(3, [108, 120, 150, 235], "person", vy=60)
    walker = _track(4, [150, 140, 190, 255], "person", vy=-50)    # đi ngược hướng sát bên
    ra = RiderAssociator()
    for _ in range(6):
        riders, rider_of = ra.update([driver, passenger, walker], [moto])
    assert set(riders[1]) == {2, 3}
    assert 4 not in rider_of
    assert riders[1][0] == 2                     # người ở phía trước theo hướng chạy (y tăng) là người lái
    assert not maybe_child(passenger, driver)
    child = _track(5, [110, 175, 140, 240], "person", vy=60)
    assert maybe_child(child, driver)


def test_ego_motion_recovers_translation_and_forward_motion():
    rng = np.random.default_rng(0)
    base = (rng.random((480, 640)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    est = EgoMotionEstimator(downscale=1.0)
    est.update(base, 0.0)
    M = np.float32([[1, 0, 6], [0, 1, 0]])
    shifted = cv2.warpAffine(base, M, (640, 480), borderMode=cv2.BORDER_REFLECT)
    em = est.update(shifted, 1 / 30)
    assert em.warp is not None and abs(em.warp[0, 2] - 6) < 1.0
    # zoom quanh điểm tụ (giả lập xe tiến về phía trước) → k_ground > 0, đang chạy
    est2 = EgoMotionEstimator(downscale=1.0, horizon_y_ratio=0.45)
    est2.update(base, 0.0)
    Z = cv2.getRotationMatrix2D((320, 216), 0, 1.03)
    zoomed = cv2.warpAffine(base, Z, (640, 480), borderMode=cv2.BORDER_REFLECT)
    em2 = est2.update(zoomed, 1 / 30)
    assert em2.k_ground is not None and em2.k_ground > 0 and em2.moving
    assert em2.static_scale_rate(400) > em2.static_scale_rate(300) > 0


# ---------------------------------------------------------------- backend ONNX (OpenCV DNN, không cần PyTorch)
def test_dnn_onnx_backend_detects_and_maps_classes():
    import pytest

    from traffic_safety_ai import MODELS_DIR
    from traffic_safety_ai.core.config import ModelSpec
    from traffic_safety_ai.core.detector import DnnYoloModel, load_model

    onnx = MODELS_DIR / "yolov8n_960.onnx"
    if not onnx.exists():
        pytest.skip("chưa có models/yolov8n_960.onnx")
    m = load_model("base", ModelSpec(weights=str(onnx), imgsz=960, conf=0.25, backend="opencv"))
    assert isinstance(m, DnnYoloModel) and m.has_class("motorcycle") and not m.has_class("airplane")
    img = np.full((540, 960, 3), 90, np.uint8)
    out = m.predict([img, img[:200, :300]])
    assert len(out) == 2 and all(isinstance(o.boxes, list) for o in out)
    assert m.ids_for(["car"]) == [2]
