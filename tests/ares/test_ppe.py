"""AI bảo hộ: ánh xạ lớp của model, ghép đồ bảo hộ với từng người, bộ đếm thiếu / cấm — bằng khung xương + hộp giả
(không cần model/GPU)."""

from __future__ import annotations

import numpy as np

from ares.behavior.analyzer import Analyzer, summarize
from ares.ppe import PRESETS, clean_rules, describe
from ares.ppe.detector import PPEDet, map_class
from ares.ppe.rules import assess, regions
from tests.ares.test_behavior import FPS, H, cfg_for, person

WOOD = PRESETS["woodshop"]  # bắt buộc kính + chụp tai + khẩu trang, cấm găng tay


def gear(kp, items):
    """Hộp đồ bảo hộ giả đặt đúng chỗ trên người có khung xương `kp`."""
    out = []
    nose, leye, reye, lear, rear, lwri, rwri = kp[0, :2], kp[1, :2], kp[2, :2], kp[3, :2], kp[4, :2], kp[9, :2], kp[10, :2]
    hs = 0.1 * H
    for it in items:
        if it == "goggles":
            c = (leye + reye) / 2
            out.append(PPEDet("goggles", True, (c[0] - hs * 0.6, c[1] - hs * 0.2, c[0] + hs * 0.6, c[1] + hs * 0.2), 0.8))
        elif it == "ear":
            c = (lear + rear) / 2
            out.append(PPEDet("ear", True, (c[0] - hs * 0.8, c[1] - hs * 0.9, c[0] + hs * 0.8, c[1] + hs * 0.4), 0.8))
        elif it == "mask":
            out.append(PPEDet("mask", True, (nose[0] - hs * 0.4, nose[1], nose[0] + hs * 0.4, nose[1] + hs * 0.5), 0.8))
        elif it == "helmet":
            out.append(PPEDet("helmet", True, (nose[0] - hs * 0.7, nose[1] - hs * 1.4, nose[0] + hs * 0.7, nose[1] - hs * 0.5), 0.8))
        elif it == "gloves":
            for w in (lwri, rwri):
                out.append(PPEDet("gloves", True, (w[0] - hs * 0.3, w[1], w[0] + hs * 0.3, w[1] + hs * 0.6), 0.8))
    return out


def run_ppe(an, seconds, people_fn, gear_fn, t0=0.0, cam="cam", ppe_every=5):
    """Chạy 10 khung/giây, AI bảo hộ 2 lần/giây. -> (sự kiện, views khung cuối)."""
    evs, views = [], []
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        people = people_fn(t)
        kps = np.stack([p[0] for p in people])
        boxes = np.stack([p[1] for p in people])
        ppe = {cam: gear_fn(t, people)} if i % ppe_every == 0 else None
        r = an.process_poses([(cam, None, t, kps, boxes, np.full(len(people), 0.9, np.float32))], draw=False, ppe=ppe)[0]
        evs += r.events
        views = r.tracks
    return evs, views


def test_map_class_sh17_and_construction_ppe():
    sh17 = {"Person": None, "Head": None, "Face": None, "Glasses": ("goggles", True), "Face-mask-medical": ("mask", True),
            "Face-guard": ("goggles", True), "Ear": None, "Earmuffs": ("ear", True), "ear-mufs": ("ear", True), "Hands": None,
            "Gloves": ("gloves", True), "Foot": None, "Shoes": ("boots", True), "Safety-vest": ("vest", True),
            "Tools": None, "Helmet": ("helmet", True), "Medical-suit": ("coat", True), "Safety-suit": ("coat", True)}
    cppe = {"helmet": ("helmet", True), "gloves": ("gloves", True), "vest": ("vest", True), "boots": ("boots", True),
            "goggles": ("goggles", True), "none": None, "Person": None, "no_helmet": ("helmet", False),
            "no_goggle": ("goggles", False), "no_gloves": ("gloves", False), "no_boots": ("boots", False),
            "NO-Hardhat": ("helmet", False), "NO-Safety Vest": ("vest", False), "NO-Mask": ("mask", False),
            "Hardhat": ("helmet", True), "machinery": None, "Safety Cone": None}
    for name, want in {**sh17, **cppe}.items():
        assert map_class(name) == want, name


def test_rules_parsing_and_text():
    assert clean_rules("woodshop") == {"required": ["goggles", "ear", "mask"], "forbidden": ["gloves"]}
    assert clean_rules("goggles, helmet,xyz") == {"required": ["goggles", "helmet"], "forbidden": []}
    assert clean_rules({"required": ["gloves", "gloves"], "forbidden": ["gloves", "ear", "?"]}) == \
        {"required": ["gloves"], "forbidden": ["ear"]}  # một món không thể vừa bắt buộc vừa cấm
    assert clean_rules(None) == clean_rules("") == {"required": [], "forbidden": []}
    assert describe(["goggles", "ear"], ["gloves"]) == \
        "thiếu kính bảo hộ, chụp tai chống ồn; đang đeo găng tay (bị cấm ở đây)"


def test_assess_assigns_gear_to_the_right_person_and_needs_visibility():
    a, b = person(300, 400), person(520, 400)
    dets = gear(a[0], ["goggles", "ear"]) + gear(b[0], ["mask", "gloves"])
    items = ["goggles", "ear", "mask", "gloves", "helmet"]
    sa, sb = assess([a, b], dets, items)
    assert sa == {"goggles": "worn", "ear": "worn", "mask": "missing", "gloves": "missing", "helmet": "missing"}
    assert sb == {"goggles": "missing", "ear": "missing", "mask": "worn", "gloves": "worn", "helmet": "missing"}
    # quay lưng (không thấy mũi/mắt): kính + khẩu trang "không rõ", chụp tai và nón vẫn xét được
    kp, box = person(300, 400)
    kp = kp.copy()
    kp[0:3, 2] = 0.0
    assert assess([(kp, box)], [], items)[0] == {"goggles": "unknown", "ear": "missing", "mask": "unknown",
                                                 "gloves": "missing", "helmet": "missing"}
    # ở quá xa (đầu ~6 px sau khi thu nhỏ): không kết luận thiếu món nào
    assert set(assess([a], [], items, scale=0.2)[0].values()) == {"unknown"}
    # model có lớp "không đội nón" -> thiếu, kể cả khi người ở xa
    far = assess([a], [PPEDet("helmet", False, gear(a[0], ["helmet"])[0].box, 0.7)], ["helmet"], scale=0.2)
    assert far[0] == {"helmet": "missing"}
    assert regions(a[0], a[1])["face"] and abs(regions(a[0], a[1])["hs"] - 0.1 * H) < 2


def test_missing_gear_alerts_after_5s_then_clears_when_worn(tmp_path):
    an = Analyzer(cfg_for(tmp_path))
    an.ppe_rules["cam"] = clean_rules(WOOD)
    who = lambda t: [person(400, 430)]  # noqa: E731
    # 0–9 s: không đeo gì
    evs, views = run_ppe(an, 9, who, lambda t, p: [])
    assert [e.kind for e in evs] == ["PPE"]
    e = evs[0]
    assert 5.0 <= e.t <= 7.5 and e.detail["missing"] == ["goggles", "ear", "mask"] and e.detail["forbidden"] == []
    assert e.detail["text"] == "thiếu kính bảo hộ, chụp tai chống ồn, khẩu trang"
    v = views[0]
    assert v.level == 1 and v.ppe_text == "Thiếu kính, chụp tai, khẩu trang" and v.text == "Đứng"
    assert summarize(views) == (1, "CHÚ Ý: Thiếu kính, chụp tai, khẩu trang (người #1)")
    # 9–15 s: đeo đủ -> hết cảnh báo, hiện "đủ bảo hộ"; vẫn vi phạm thì KHÔNG báo lại dồn dập
    evs, views = run_ppe(an, 6, who, lambda t, p: gear(p[0][0], ["goggles", "ear", "mask"]), t0=9)
    assert evs == [] and views[0].level == 0 and views[0].ppe["all_ok"] and views[0].ppe_text == ""
    # 15–25 s: tháo kính ra -> báo lại (đã đeo rồi tháo là vi phạm mới), chỉ ghi món đang thiếu
    evs, views = run_ppe(an, 10, who, lambda t, p: gear(p[0][0], ["ear", "mask"]), t0=15)
    assert [e.detail["missing"] for e in evs] == [["goggles"]]


def test_forbidden_gloves_and_detector_misses_are_tolerated(tmp_path):
    an = Analyzer(cfg_for(tmp_path))
    an.ppe_rules["cam"] = clean_rules(WOOD)
    who = lambda t: [person(400, 430)]  # noqa: E731
    full = ["goggles", "ear", "mask"]
    # model sót kính ở 1/2 số lần (vật nhỏ): vẫn coi là có đeo, không báo nhầm
    k = {"n": 0}

    def flaky(t, p):
        k["n"] += 1
        return gear(p[0][0], full if k["n"] % 2 else ["ear", "mask"])

    evs, views = run_ppe(an, 12, who, flaky)
    assert evs == [] and views[0].ppe["all_ok"]
    # đeo găng tay trong xưởng gỗ (bị cấm) -> cảnh báo sau ~5 s
    evs, views = run_ppe(an, 9, who, lambda t, p: gear(p[0][0], full + ["gloves"]), t0=12)
    assert [(e.detail["missing"], e.detail["forbidden"]) for e in evs] == [([], ["gloves"])]
    assert views[0].ppe_text == "Đeo găng tay (cấm)" and "đang đeo găng tay (bị cấm ở đây)" in evs[0].detail["text"]


def test_no_rules_no_ppe_and_per_camera_rules(tmp_path):
    an = Analyzer(cfg_for(tmp_path))
    who = lambda t: [person(400, 430)]  # noqa: E731
    evs, views = run_ppe(an, 8, who, lambda t, p: [])
    assert evs == [] and views[0].ppe is None and views[0].level == 0  # địa điểm không chọn món nào: không kiểm
    an.ppe_rules["xuong"] = clean_rules({"required": ["helmet"]})
    evs, _ = run_ppe(an, 8, who, lambda t, p: [], cam="xuong")
    assert [e.detail["missing"] for e in evs] == [["helmet"]]
    # mặc định của máy qua ARES_PPE_REQUIRED / ARES_PPE_FORBIDDEN
    an2 = Analyzer(cfg_for(tmp_path, ppe_required="woodshop", ppe_forbidden="boots"))
    assert an2.rules_for("bat-ky") == {"required": ["goggles", "ear", "mask"], "forbidden": ["gloves", "boots"]}


def test_real_process_runs_ppe_model_only_when_needed_and_draws(tmp_path):
    """Analyzer.process: chỉ gọi YOLO đồ bảo hộ cho camera có luật + có người, tối đa ppe_fps lần/giây; vẽ không lỗi."""
    calls = []

    class FakePose:
        device = "cpu"

        def infer(self, images):
            kp, box = person(400, 430)
            return [(kp[None], box[None], np.array([0.9], np.float32)) for _ in images]

    class FakePPE:
        def infer(self, images):
            calls.append(len(images))
            kp, _ = person(400, 430)
            return [gear(kp, ["goggles"]) for _ in images]

    an = Analyzer(cfg_for(tmp_path, ppe_fps=2.0), FakePose(), None, FakePPE())
    an.ppe_rules["a"] = clean_rules({"required": ["goggles", "ear"]})
    img = np.full((720, 960, 3), 60, np.uint8)
    evs = []
    for i in range(80):
        res = an.process([("a", img, i / 10), ("b", img, i / 10)], draw=True)
        evs += [e for r in res for e in r.events]
        assert res[0].image.shape == (480, 640, 3)
    assert set(calls) == {1} and 15 <= len(calls) <= 17  # chỉ camera "a", 2 lần/giây trong 8 giây
    assert [(e.cam_id, e.detail["missing"]) for e in evs] == [("a", ["ear"])]


# ------------------------------------------------------------------ server: luật theo địa điểm -> máy AI; thử video kèm đồ bảo hộ
def test_server_place_rules_reach_edge_and_video_jobs(tmp_path):
    import json

    from fastapi.testclient import TestClient

    import ares.ppe as edge_ppe
    import ares_server.ppe as srv_ppe
    from ares.behavior.monitor import BehaviorMonitor
    from ares.clock import ManualClock
    from ares_server.app import create_app
    from ares_server.config import ServerSettings
    from ares_server.db import ServerDB

    assert srv_ppe.ITEMS == edge_ppe.ITEMS and srv_ppe.PRESETS == edge_ppe.PRESETS  # hai bản danh mục phải giống nhau

    db = ServerDB(tmp_path / "s.db")
    app = create_app(ServerSettings(data_dir=tmp_path / "srv"), db)
    X = {"X-ARES": "1"}
    sid, key = db.create_site("Máy chủ")
    Hd = {"Authorization": f"Bearer {key}", "X-Edge-Id": "may-chu"}
    db.create_user("tri", "mk-dai-va-kho", "Trí", "admin")
    db.create_user("an.nv", "mat-khau-cua-an", "An", "viewer")
    db.create_user("binh", "mat-khau-cua-binh", "Bình", "viewer")
    an, binh = (TestClient(app, client=(ip, 5000)) for ip in ("192.168.1.20", "192.168.1.30"))
    for c, u, p in ((an, "an.nv", "mat-khau-cua-an"), (binh, "binh", "mat-khau-cua-binh")):
        assert c.post("/api/login", json={"user": u, "password": p}, headers=X).status_code == 200
    edge = TestClient(app)
    hb = {"edge": {"name": "Máy chủ"}, "cameras": [{"id": "main", "name": "Xưởng", "online": True, "n_people": 1}],
          "ai": {"behavior": {"running": True, "ppe": {"ready": True, "weights": "ppe.pt", "items": ["goggles", "ear"]}}}}
    r = edge.post("/api/v1/heartbeat", headers=Hd, content=json.dumps(hb)).json()
    assert r["ppe_rules"] == {"main": {"required": [], "forbidden": []}}  # chưa địa điểm nào chọn

    cat = an.get("/api/ppe").json()
    assert cat["presets"]["woodshop"]["forbidden"] == ["gloves"] and cat["items"]["ear"]["name"] == "chụp tai chống ồn"
    assert cat["model"]["ready"] and cat["model"]["items"] == ["goggles", "ear"]
    # An đặt mẫu Xưởng gỗ cho địa điểm của mình, có camera của An ở đó
    pid = an.get("/api/places").json()["items"][0]["id"]
    cam = an.post("/api/my/cameras", json={"name": "Bàn cưa"}, headers=X).json()
    assert an.post(f"/api/places/{pid}/ppe", json={"preset": "woodshop"}, headers=X).json()["ppe"] == \
        {"required": ["goggles", "ear", "mask"], "forbidden": ["gloves"]}
    assert binh.post(f"/api/places/{pid}/ppe", json={"required": ["helmet"]}, headers=X).status_code == 404
    assert an.get("/api/places").json()["items"][0]["ppe"]["required"] == ["goggles", "ear", "mask"]
    r = edge.post("/api/v1/heartbeat", headers=Hd, content=json.dumps(hb)).json()
    assert r["ppe_rules"][cam["cam_id"]] == {"required": ["goggles", "ear", "mask"], "forbidden": ["gloves"]}
    assert r["ppe_rules"]["main"] == {"required": [], "forbidden": []}  # camera ở địa điểm khác: không kiểm

    # máy AI nhận luật, lưu ra file (mất mạng / khởi động lại vẫn còn), camera không có luật dùng mặc định của máy
    cfg = cfg_for(tmp_path / "edge", ppe_required="helmet")
    mon = BehaviorMonitor(cfg, ManualClock(__import__("datetime").datetime(2026, 11, 17, 8, 0)), analyzer=Analyzer(cfg))
    assert mon.set_ppe_rules(r["ppe_rules"]) is True and mon.set_ppe_rules(r["ppe_rules"]) is False
    assert mon.analyzer.rules_for(cam["cam_id"])["forbidden"] == ["gloves"]
    assert mon.analyzer.rules_for("main") == {"required": ["helmet"], "forbidden": []}
    from ares.ppe.load import load_detector, load_rules
    assert load_rules(cfg)[cam["cam_id"]]["required"] == ["goggles", "ear", "mask"]
    det, info = load_detector(cfg)
    assert det is None and not info["ready"] and "CAI_AI_BAO_HO.bat" in info["hint"]  # chưa cài model: báo rõ, không lỗi
    mon.store.close()

    # tắt kiểm ở địa điểm; tuỳ chọn riêng (bỏ cấm găng)
    assert an.post(f"/api/places/{pid}/ppe", json={"required": ["goggles", "zzz"], "forbidden": []}, headers=X).json()["ppe"] == \
        {"required": ["goggles"], "forbidden": []}
    assert an.post(f"/api/places/{pid}/ppe", json={"required": [], "forbidden": []}, headers=X).json()["ok"]
    assert db.place(pid)["ppe"] == ""

    # Thử video kèm kiểm đồ bảo hộ: máy AI nhận luật theo job
    video = b"\x00\x00\x00\x18ftypmp42" + b"x" * 5000
    assert an.post("/api/videos?name=a.mp4&ppe=khong-co", content=video, headers=X).status_code == 400
    jid = an.post("/api/videos?name=a.mp4&ppe=woodshop", content=video, headers=X).json()["id"]
    j = edge.post("/api/v1/jobs/next", headers=Hd, content=b"{}").json()
    assert j["id"] == jid and j["ppe"] == {"required": ["goggles", "ear", "mask"], "forbidden": ["gloves"]}
    jid2 = an.post("/api/videos?name=b.mp4", content=video, headers=X).json()["id"]
    assert edge.post("/api/v1/jobs/next", headers=Hd, content=b"{}").json()["ppe"] == {"required": [], "forbidden": []} and jid2


def test_video_analyze_with_ppe_rules(tmp_path):
    """analyze() thật trên video tổng hợp với pose + AI bảo hộ giả: ra sự kiện PPE kèm câu mô tả."""
    import cv2

    from ares.behavior.run_video import analyze

    p = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 720))
    for _ in range(90):
        vw.write(np.full((720, 640, 3), 80, np.uint8))
    vw.release()

    class Pose:
        device = "cpu"

        def infer(self, images):
            kp, box = person(320, 430)
            return [(kp[None], box[None], np.array([0.9], np.float32)) for _ in images]

    class Ppe:
        def infer(self, images):
            kp, _ = person(320, 430)
            return [gear(kp, ["goggles", "mask"]) for _ in images]

    r = analyze(cfg_for(tmp_path), Pose(), None, p, None, 10.0, frames_zip=tmp_path / "f.zip", ppe=Ppe(), ppe_rules="woodshop")
    assert [(e["kind"], e["detail"]["missing"]) for e in r["events"]] == [("PPE", ["ear"])]
    assert r["events"][0]["label"] == "Vi phạm đồ bảo hộ" and r["events"][0]["detail"]["text"] == "thiếu chụp tai chống ồn"
    assert r["ppe"] == {"rules": {"required": ["goggles", "ear", "mask"], "forbidden": ["gloves"]}, "ready": True}
    assert any(lv == 1 and "Thiếu chụp tai" in txt for _, lv, txt in r["status_runs"])
    # không có model đồ bảo hộ: vẫn phân tích hành vi bình thường, ghi rõ chưa sẵn sàng
    r2 = analyze(cfg_for(tmp_path), Pose(), None, p, None, 10.0, ppe=None, ppe_rules="woodshop")
    assert r2["events"] == [] and r2["ppe"]["ready"] is False
