"""AI bổ sung (cháy/khói, hút thuốc): xác nhận theo thời gian, gắn điếu thuốc với người, sự kiện, dữ liệu train dần."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from ares.config import AresSettings
from ares.hazards.detector import HazardDet

W, H = 640, 480


class FakeHz:
    """Model giả: trả hộp theo lịch t -> danh sách hộp."""

    def __init__(self, name, plan):
        self.name, self.plan, self.k = name, plan, 0

    def infer(self, imgs):
        out = []
        for _ in imgs:
            out.append(self.plan(self.k))
            self.k += 1
        return out


class FakePose:
    device = "cpu"

    def __init__(self, people):
        self.people = people  # hàm k -> list hộp người

    def infer(self, imgs):
        out = []
        for _ in imgs:
            boxes = self.people()
            kps = []
            for x1, y1, x2, y2 in boxes:
                kp = np.zeros((17, 3), np.float32)
                cx = (x1 + x2) / 2
                kp[0] = (cx, y1 + 15, 0.9)  # mũi
                kp[9] = (x1, (y1 + y2) / 2, 0.9)
                kp[10] = (x2, (y1 + y2) / 2, 0.9)
                for j, (dx, fy) in {5: (-15, 0.25), 6: (15, 0.25), 11: (-10, 0.55), 12: (10, 0.55), 13: (-10, 0.75),
                                    14: (10, 0.75), 15: (-10, 0.95), 16: (10, 0.95)}.items():
                    kp[j] = (cx + dx, y1 + fy * (y2 - y1), 0.9)
                kps.append(kp)
            b = np.array(boxes, np.float32).reshape(-1, 4)
            out.append((np.array(kps, np.float32).reshape(-1, 17, 3), b, np.full(len(b), 0.9, np.float32)))
        return out


def run(cfg, hz, pose, seconds=12, fps=10):
    from ares.behavior.analyzer import Analyzer

    an = Analyzer(cfg, pose, None, None, hz)
    img = np.zeros((H, W, 3), np.uint8)
    evs, alerts = [], []
    for k in range(int(seconds * fps)):
        r = an.process([("cam", img, k / fps)], draw=True)[0]
        evs += r.events
        alerts.append(r.alert)
    return an, evs, alerts


def test_fire_needs_several_checks(tmp_path):
    cfg = AresSettings(storage_root=tmp_path, behavior_device="cpu")
    fire = HazardDet("fire", "LỬA", (100, 100, 200, 200), 0.8)
    # 1 khung lẻ có lửa (báo nhầm kiểu đèn vàng) -> không báo
    _, evs, _ = run(cfg, {"fire": FakeHz("fire", lambda k: [fire] if k == 3 else [])}, FakePose(lambda: []))
    assert not [e for e in evs if e.kind == "FIRE"]
    # lửa liên tục -> báo 1 lần (có hạn chống báo lặp), thanh trạng thái BÁO ĐỘNG
    _, evs, alerts = run(cfg, {"fire": FakeHz("fire", lambda k: [fire])}, FakePose(lambda: []))
    f = [e for e in evs if e.kind == "FIRE"]
    assert len(f) == 1 and f[0].track_ids == [] and f[0].detail["what"] == "lửa" and f[0].detail["boxes"]
    assert any(a and a[0] == 2 for a in alerts)


def test_smoke_alone_needs_more_hits(tmp_path):
    cfg = AresSettings(storage_root=tmp_path, behavior_device="cpu")
    smoke = HazardDet("smoke", "KHÓI", (10, 10, 300, 200), 0.6)
    _, evs, _ = run(cfg, {"fire": FakeHz("fire", lambda k: [smoke] if k % 2 == 0 else [])}, FakePose(lambda: []))
    assert not evs  # khói chập chờn (bụi gỗ) 3/5 lần: chưa đủ 4/5
    _, evs, _ = run(cfg, {"fire": FakeHz("fire", lambda k: [smoke])}, FakePose(lambda: []))
    assert [e.detail["what"] for e in evs if e.kind == "FIRE"] == ["khói"]


def test_cigarette_near_mouth_is_smoking(tmp_path):
    cfg = AresSettings(storage_root=tmp_path, behavior_device="cpu")
    person = (300, 100, 380, 400)
    near = HazardDet("cigarette", "THUỐC LÁ", (335, 110, 350, 125), 0.7)  # sát mũi
    far = HazardDet("cigarette", "THUỐC LÁ", (40, 400, 55, 415), 0.7)  # điếu thuốc trên bàn xa
    _, evs, _ = run(cfg, {"smoking": FakeHz("smoking", lambda k: [far])}, FakePose(lambda: [person]))
    assert not evs
    an, evs, _ = run(cfg, {"smoking": FakeHz("smoking", lambda k: [near])}, FakePose(lambda: [person]))
    s = [e for e in evs if e.kind == "SMOKING"]
    assert len(s) == 1 and s[0].track_ids == [1]
    v = an.cams["cam"].views(11.9)
    assert v[0].level >= 1 and "Hút thuốc" in v[0].text


def test_feedback_dataset_from_verdicts(tmp_path):
    from ares.behavior.store import EventStore
    from ares.hazards.feedback import build_dataset, collect_ppe_for_labeling, save_sample

    cfg = AresSettings(storage_root=tmp_path, behavior_device="cpu")
    root = tmp_path / "behavior"
    st = EventStore(root)
    raw = np.zeros((960, 1920, 3), np.uint8)
    hz = SimpleNamespace(t=5.0, dets={"fire": [HazardDet("fire", "LỬA", (100, 100, 300, 300), 0.8)]})
    ca = SimpleNamespace(hz=hz, ppe_on=False, ppe_t=0, ppe_dets=[])
    for i, (kind, verdict) in enumerate([("FIRE", "true"), ("FIRE", "false"), ("FIRE", ""), ("PPE", "false")]):
        d = root / "events" / f"e{i}"
        d.mkdir(parents=True)
        save_sample(d, raw, SimpleNamespace(kind=kind, t=5.0), ca)
        st.add({"id": f"e{i}", "t": 5.0, "kind": kind, "verdict": verdict, "dir": f"events/e{i}", "status": "ack"})
    st.close()
    meta = json.loads((root / "events" / "e0" / "dets.json").read_text())
    assert meta["w"] == 1280 and meta["hazard"]["fire"][0][1] == round(100 * 1280 / 1920, 1)  # hộp theo ảnh đã thu nhỏ
    ds = build_dataset(cfg, "fire", {0: "smoke", 1: "fire"}, tmp_path / "ds")
    assert ds["counts"]["pos"] == 1 and ds["counts"]["neg"] == 1  # sự kiện chưa đánh giá không dùng
    labels = sorted((tmp_path / "ds" / "labels").rglob("*.txt"))
    txt = {p.stem: p.read_text() for p in labels}
    assert txt["e0"].startswith("1 ") and txt["e1"] == ""  # Đúng -> hộp lớp fire; Báo nhầm -> nhãn rỗng
    assert collect_ppe_for_labeling(cfg) == 1 and collect_ppe_for_labeling(cfg) == 0


def test_train_refuses_without_enough_samples(tmp_path, monkeypatch):
    from ares.hazards import train as T

    cfg = AresSettings(storage_root=tmp_path, behavior_device="cpu")
    monkeypatch.setattr("ares.hazards.detector.weights_for", lambda c, n: tmp_path / "khong-co.pt")
    r = T.train_one(cfg, "fire")
    assert not r["ok"] and "chưa cài" in r["message"]


def test_ppe_multi_model_merges(tmp_path):
    from ares.ppe.detector import MultiPPEDetector, PPEDet

    class D:
        def __init__(self, item, w):
            self.items, self.device, self.weights, self.item = [item], "cpu", w, item

        def infer(self, imgs):
            return [[PPEDet(self.item, True, (0, 0, 1, 1), 0.9)] for _ in imgs]

        def describe(self):
            return {"classes": {self.item: self.item}}

    m = MultiPPEDetector([D("goggles", "a.pt"), D("helmet", "b.pt")])
    r = m.infer([np.zeros((4, 4, 3), np.uint8)])
    assert [d.item for d in r[0]] == ["goggles", "helmet"] and m.items == ["goggles", "helmet"]
    assert m.describe()["models"] == ["a.pt", "b.pt"]
