"""Kiểm tra luật nhận diện hành vi bằng khung xương tổng hợp (không cần model/GPU) + luồng sự kiện -> clip -> API."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from ares.behavior.analyzer import Analyzer
from ares.config import AresSettings

H = 300.0  # chiều cao người trên ảnh (px)
FPS = 10.0


def cfg_for(tmp_path, **kw):
    base = dict(storage_root=tmp_path, behavior_mode="rules", behavior_still_s=30.0, behavior_clip_pre_s=1.0,
                behavior_clip_post_s=1.0)
    base.update(kw)
    return AresSettings(**base)


# ------------------------------------------------------------------ dựng khung xương giả
STAND = {  # toạ độ tương đối so với trung điểm hông, đơn vị H, y đi xuống
    0: (0, -0.42), 1: (-0.02, -0.44), 2: (0.02, -0.44), 3: (-0.05, -0.43), 4: (0.05, -0.43),
    5: (-0.10, -0.30), 6: (0.10, -0.30), 7: (-0.14, -0.15), 8: (0.14, -0.15), 9: (-0.12, 0.0), 10: (0.12, 0.0),
    11: (-0.06, 0.0), 12: (0.06, 0.0), 13: (-0.06, 0.25), 14: (0.06, 0.25), 15: (-0.06, 0.5), 16: (0.06, 0.5),
}


def person(hip_x, hip_y, theta_deg=0.0, rel=None, jitter=0.0, rng=None):
    """theta: nghiêng cả người quanh hông (0 đứng, 90 nằm ngang)."""
    rel = rel or STAND
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    kp = np.zeros((17, 3), np.float32)
    for j, (dx, dy) in rel.items():
        x, y = dx * H, dy * H
        kp[j] = (hip_x + c * x - s * y, hip_y + s * x + c * y, 0.9)
    if jitter and rng is not None:
        kp[:, :2] += rng.normal(0, jitter * H, (17, 2))
    box = np.array([kp[:, 0].min() - 10, kp[:, 1].min() - 10, kp[:, 0].max() + 10, kp[:, 1].max() + 10], np.float32)
    return kp, box


def sitting_rel(slump=False):
    r = dict(STAND)
    r.update({13: (-0.06 + 0.25, 0.01), 14: (0.06 + 0.25, 0.01), 15: (0.19, 0.26), 16: (0.31, 0.26)})
    if slump:  # thân đổ về trước, đầu thấp hơn vai
        r.update({5: (0.10, -0.24), 6: (0.22, -0.22), 0: (0.30, -0.18), 1: (0.29, -0.19), 2: (0.31, -0.19),
                  3: (0.26, -0.20), 4: (0.33, -0.20), 7: (0.25, -0.08), 8: (0.33, -0.08), 9: (0.32, 0.0), 10: (0.40, 0.0)})
    return r


def run(an, frames):
    """frames: list[(t, [(kp, box), ...])] -> list sự kiện."""
    evs = []
    for t, people in frames:
        kps = np.stack([p[0] for p in people]) if people else np.zeros((0, 17, 3), np.float32)
        boxes = np.stack([p[1] for p in people]) if people else np.zeros((0, 4), np.float32)
        res = an.process_poses([("cam", None, t, kps, boxes, np.full(len(people), 0.9, np.float32))], draw=False)
        evs += res[0].events
    return evs


def timeline(seconds, fn, t0=0.0):
    n = int(seconds * FPS)
    return [(t0 + i / FPS, fn(i / FPS)) for i in range(n)]


# ------------------------------------------------------------------ luật
def test_fall_then_lying_still(tmp_path):
    an = Analyzer(cfg_for(tmp_path))
    rng = np.random.default_rng(1)
    floor_hip = 600.0

    def f(t):
        if t < 3:
            return [person(400, 450, 0, jitter=0.004, rng=rng)]
        if t < 3.6:  # ngã trong 0,6 s: nghiêng 0 -> 90 độ, hông tụt xuống sàn
            a = (t - 3) / 0.6
            return [person(400 + 60 * a, 450 + (floor_hip - 450) * a, 90 * a)]
        return [person(460, floor_hip, 90, jitter=0.004, rng=rng)]

    evs = run(an, timeline(40, f))
    kinds = [e.kind for e in evs]
    assert kinds[0] == "FALL", kinds
    assert "LYING_DOWN" not in kinds  # đã báo té ngã thì không báo trùng "nằm xuống"
    assert "LYING_STILL" in kinds
    ls = next(e for e in evs if e.kind == "LYING_STILL")
    assert ls.detail["after_fall"] is True
    assert 33 <= ls.t <= 36  # ~3,6 s (bắt đầu nằm) + 30 s
    assert kinds.count("FALL") == 1


def test_sitting_down_is_not_a_fall(tmp_path):
    an = Analyzer(cfg_for(tmp_path))

    def f(t):
        if t < 3:
            return [person(400, 450, 0)]
        return [person(400, 520, 0, rel=sitting_rel())]  # ngồi thẳng, bất động

    assert run(an, timeline(45, f)) == []


def test_slumped_still(tmp_path):
    an = Analyzer(cfg_for(tmp_path))
    rng = np.random.default_rng(2)
    evs = run(an, timeline(40, lambda t: [person(400, 520, 0, rel=sitting_rel(True), jitter=0.004, rng=rng)]))
    assert [e.kind for e in evs] == ["SLUMPED"]


def test_lying_but_moving_is_lying_down_not_still(tmp_path):
    an = Analyzer(cfg_for(tmp_path))
    # nằm nhưng trở mình/cử động liên tục: KHÔNG phải bất động, nhưng vẫn báo "nằm dưới sàn" sau ~5 s
    evs = run(an, timeline(45, lambda t: [person(400 + 40 * math.sin(t), 600, 90 + 8 * math.sin(2.5 * t))]))
    assert [e.kind for e in evs] == ["LYING_DOWN"]
    assert evs[0].detail["moving"] is True and evs[0].detail["sudden"] is False and 4.5 <= evs[0].t <= 7


def wave_arm(kp_box, t, amp=0.08, hz=1.0):
    """Người nằm yên, chỉ cử động 1 cánh tay (khuỷu + cổ tay) — kiểu cử động mà trước đây bị tính nhầm là bất động."""
    kp, box = kp_box
    d = amp * H * math.sin(2 * math.pi * hz * t)
    kp = kp.copy()
    kp[7, 1] += 0.5 * d
    kp[9, 1] += d
    kp[9, 0] += 0.5 * d
    return kp, box


def test_lying_moving_only_an_arm_is_not_still(tmp_path):
    an = Analyzer(cfg_for(tmp_path, behavior_still_s=15.0))
    rng = np.random.default_rng(5)
    evs = run(an, timeline(40, lambda t: [wave_arm(person(400, 600, 90, jitter=0.004, rng=rng), t)]))
    assert [e.kind for e in evs] == ["LYING_DOWN"] and evs[0].detail["moving"] is True


def test_truly_still_lying_escalates(tmp_path):
    an = Analyzer(cfg_for(tmp_path, behavior_still_s=15.0))
    rng = np.random.default_rng(6)
    evs = run(an, timeline(30, lambda t: [person(400, 600, 90, jitter=0.004, rng=rng)]))
    assert [e.kind for e in evs] == ["LYING_DOWN", "LYING_STILL"]
    assert evs[0].detail["moving"] is False and 4.5 <= evs[0].t <= 7
    assert 15 <= evs[1].t <= 18


def test_movement_restarts_still_timer(tmp_path):
    an = Analyzer(cfg_for(tmp_path, behavior_still_s=15.0))
    rng = np.random.default_rng(7)

    def f(t):
        p = person(400, 600, 90, jitter=0.004, rng=rng)
        return [wave_arm(p, t) if 10 <= t < 12 else p]  # nằm yên 10 s, cử động tay 2 s, rồi lại nằm yên

    evs = run(an, timeline(32, f))
    still = [e for e in evs if e.kind == "LYING_STILL"]
    assert len(still) == 1 and still[0].t >= 12 + 15 - 0.5  # đếm lại từ lúc thôi cử động, không cộng dồn 10 s trước


def test_sudden_lying_without_big_drop_alerts_fast(tmp_path):
    """Đang đứng rồi nằm xuống (không đủ tiêu chí té ngã: nằm chậm 2,5 s) -> báo 'nằm xuống đột ngột' sau ~2 s."""
    an = Analyzer(cfg_for(tmp_path))
    rng = np.random.default_rng(8)

    def f(t):
        if t < 3:
            return [person(400, 450, 0, jitter=0.004, rng=rng)]
        if t < 5.5:
            a = (t - 3) / 2.5
            return [person(400, 450 + 60 * a, 90 * a)]
        return [wave_arm(person(400, 510, 90, jitter=0.004, rng=rng), t)]

    evs = run(an, timeline(15, f))
    assert [e.kind for e in evs] == ["LYING_DOWN"], [(e.kind, e.t) for e in evs]
    assert evs[0].detail["sudden"] is True and evs[0].t <= 9


def test_fight_and_quiet_neighbours(tmp_path):
    an = Analyzer(cfg_for(tmp_path))

    def punching(t):
        a, b = person(400, 450), person(560, 450)
        a[0][10, :2] += (0.35 * H * math.sin(18 * t), 0.1 * H * math.cos(18 * t))  # cổ tay phải vung nhanh
        b[0][9, :2] += (-0.3 * H * math.sin(16 * t + 1), 0.1 * H * math.sin(16 * t))
        return [a, b]

    evs = run(an, timeline(4, punching))
    assert [e.kind for e in evs] == ["FIGHT"]
    an2 = Analyzer(cfg_for(tmp_path))
    assert run(an2, timeline(10, lambda t: [person(400, 450), person(560, 450)])) == []


def test_cooldown_blocks_repeat(tmp_path):
    an = Analyzer(cfg_for(tmp_path, behavior_still_s=5.0))
    seq = timeline(12, lambda t: [person(400, 600, 90)])
    seq += timeline(3, lambda t: [person(400, 450, 0)], t0=12)  # đứng dậy
    seq += timeline(10, lambda t: [person(400, 600, 90)], t0=15)  # nằm lại ngay chỗ cũ trong thời gian chờ
    kinds = [e.kind for e in run(an, seq)]
    assert sorted(kinds) == ["LYING_DOWN", "LYING_STILL"]  # mỗi loại 1 lần, nằm lại chỗ cũ không báo lại


# ------------------------------------------------------------------ model: cửa sổ + mạng
def test_window_and_net_shapes():
    torch = pytest.importorskip("torch")
    from ares.behavior import CLASSES
    from ares.behavior.model import augment, build_net, build_window

    ts = np.arange(0, 3, 0.1)
    kps = np.stack([person(400, 450 + 50 * t)[0] for t in ts])
    x = build_window(ts, kps, 3.0, H)
    assert x.shape == (5, 16, 17, 2) and np.isfinite(x).all()
    assert x[1, -1, 11, 0] > x[1, 0, 11, 0]  # hông đi xuống được giữ trong toạ độ
    assert np.all(x[..., 1] == 0)  # không có người thứ hai
    xa = augment(x, np.random.default_rng(0))
    net = build_net(len(CLASSES))
    out = net(torch.from_numpy(np.stack([x, xa])))
    assert tuple(out.shape) == (2, len(CLASSES))


# ------------------------------------------------------------------ monitor: sự kiện -> DB -> clip -> API
class FakePose:
    """Người đứng 2 s rồi ngã, nằm yên."""

    def infer(self, images):
        out = []
        for img in images:
            t = float(img[0, 0, 0]) / 10.0  # thời gian mã hoá trong pixel đầu (để test không phụ thuộc đồng hồ)
            if t < 2:
                kp, box = person(400, 450, 0)
            elif t < 2.6:
                a = (t - 2) / 0.6
                kp, box = person(400, 450 + 150 * a, 90 * a)
            else:
                kp, box = person(400, 600, 90)
            out.append((kp[None], box[None], np.array([0.9], np.float32)))
        return out


class FakeClock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def time(self):
        return self.t

    def now(self):
        from datetime import datetime, timezone
        return datetime.fromtimestamp(self.t, timezone.utc)


def test_monitor_end_to_end(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ares.behavior.monitor import BehaviorMonitor
    from ares.behavior.routes import register

    cfg = cfg_for(tmp_path, behavior_clip_pre_s=1.0, behavior_clip_post_s=1.0, behavior_cameras="main")
    clock = FakeClock()
    emitted = []
    mon = BehaviorMonitor(cfg, clock, emit=emitted.append, name_of=lambda c: "Phòng thử",
                          analyzer=Analyzer(cfg, FakePose()))
    t_base = clock.t
    for i in range(60):  # 6 giây
        t = t_base + i / FPS
        clock.t = t
        img = np.zeros((360, 640, 3), np.uint8)
        img[0, 0, 0] = i  # t tương đối * 10
        mon.offer("main", img, t)
        mon.offer("other", img, t)  # không nằm trong behavior_cameras -> bị bỏ qua
        mon.step()
    mon.clips.flush()
    deadline = time.time() + 10
    while time.time() < deadline:
        items = mon.store.list()
        if items and items[0]["clip_ready"]:
            break
        time.sleep(0.1)
    items = mon.store.list()
    assert [e["kind"] for e in items] == ["FALL"]
    ev = items[0]
    assert ev["cam_id"] == "main" and ev["cam_name"] == "Phòng thử" and ev["status"] == "new"
    assert ev["clip_ready"] == 1 and ev["n_frames"] >= 10
    d = mon.root / ev["dir"]
    assert (d / "snapshot.jpg").exists() and (d / "frames" / "0001.jpg").exists() and (d / "keypoints.npz").exists()
    assert any(e.kind == "BEHAVIOR_FALL" for e in emitted)
    assert "other" not in [c["id"] for c in mon.status()["cameras"]]

    class RT:
        behavior = mon

    app = FastAPI()
    register(app, RT())
    c = TestClient(app)
    assert c.get("/api/behavior/status").json()["counts"] == {"new": 1}
    assert c.get(f"/api/behavior/events/{ev['id']}/frames/1.jpg").status_code == 200
    assert c.get("/api/behavior/events/../../etc/frames/1.jpg").status_code in (400, 404)
    assert c.post(f"/api/behavior/events/{ev['id']}/verdict", json={"verdict": "true"}).json() == {"ok": True}
    got = c.get(f"/api/behavior/events/{ev['id']}").json()
    assert got["verdict"] == "true" and got["status"] == "ack"
    assert c.get("/behavior").status_code == 200

    # sự kiện đã duyệt sinh được cửa sổ train
    from ares.behavior.train import windows_from_event
    xs, ys = windows_from_event(mon.store.get(ev["id"]), mon.root)
    assert xs and set(ys) == {1}  # lớp "fall"
    mon.stop()


def test_pose_falls_back_to_cpu_on_cuda_error():
    """torch không hỗ trợ card (vd RTX 50xx + cu124): 'no kernel image' -> chuyển CPU 1 lần, không lỗi lặp lại."""
    from types import SimpleNamespace

    from ares.behavior.pose import PoseEstimator

    calls = []

    class FakeModel:
        def predict(self, images, device="cpu", **kw):
            calls.append(device)
            if device != "cpu":
                raise RuntimeError("CUDA error: no kernel image is available for execution on the device\nmore")
            return [SimpleNamespace(boxes=None, keypoints=None) for _ in images]

    pe = object.__new__(PoseEstimator)
    pe.model, pe.device, pe.half, pe.imgsz, pe.conf, pe.fallback_reason = FakeModel(), "cuda:0", True, 640, 0.3, ""
    out = pe.infer([np.zeros((32, 32, 3), np.uint8)])
    assert len(out) == 1 and out[0][0].shape == (0, 17, 3)
    assert pe.device == "cpu" and not pe.half and pe.fallback_reason.startswith("CUDA error: no kernel image")
    pe.infer([np.zeros((32, 32, 3), np.uint8)])
    assert calls == ["cuda:0", "cpu", "cpu"]
