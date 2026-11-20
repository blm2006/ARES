"""3 phần trên server: Người & môi trường (thử video vùng + đèn), Hành vi (train lại từ đánh giá), Giao thông (/traffic)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from tests.ares.test_server_uplink import fake_runtime, live_server  # noqa: F401 — fixture

X = {"X-ARES": "1"}
LAYOUT = {"zones": [{"id": "trai", "name": "Bàn trái", "poly": [[0, 0.3], [0.5, 0.3], [0.5, 1], [0, 1]]},
                    {"id": "phai", "name": "Bàn phải", "poly": [[0.5, 0.3], [1, 0.3], [1, 1], [0.5, 1]]},
                    {"id": "x", "name": "Thiếu điểm", "poly": [[0, 0], [1, 1]]}],
          "lights": [{"id": "d1", "name": "Đèn trái", "zones": ["trai"], "device_id": 7, "output": "led", "level": 70},
                     {"id": "d2", "name": "Đèn phải", "zones": ["phai", "khong-co"]}],
          "off_delay_s": 1.0}


def write_video(path: Path, seconds: float = 6.0, fps: int = 10) -> Path:
    import cv2

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    for _ in range(int(seconds * fps)):
        vw.write(np.full((240, 320, 3), 60, np.uint8))
    vw.release()
    return path


class WalkerPose:
    """Pose giả: 1 người đi từ mép trái sang mép phải trong 4 giây rồi ra khỏi khung."""

    device = "cpu"

    def __init__(self):
        self.k = 0

    def infer(self, imgs):
        out = []
        for img in imgs:
            H, W = img.shape[:2]
            self.k += 1
            t = self.k / 10.0
            if t <= 4.0:
                cx = 10 + (W - 20) * t / 4.0
                b = np.array([[cx - 15, 90, cx + 15, 230]], np.float32)
                out.append((np.zeros((1, 17, 3), np.float32), b, np.array([0.9], np.float32)))
            else:
                out.append((np.zeros((0, 17, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)))
        return out


def test_clean_layout_drops_bad_parts():
    from ares_server.layout import clean_layout

    L = clean_layout(LAYOUT)
    assert [z["id"] for z in L["zones"]] == ["trai", "phai"]  # vùng < 3 điểm bị bỏ
    assert L["lights"][1]["zones"] == ["phai"] and L["lights"][1]["device_id"] is None
    assert L["lights"][0]["output"] == "led" and L["lights"][0]["level"] == 70


def test_analyze_people_lights_follow_person(tmp_path):
    from ares.config import AresSettings
    from ares.people_video import analyze_people

    cfg = AresSettings(storage_root=tmp_path / "e")
    src = write_video(tmp_path / "a.avi")
    r = analyze_people(cfg, WalkerPose(), src, LAYOUT, None, 10.0, frames_zip=tmp_path / "f.zip")
    kinds = [(e["kind"], e["detail"]["light"]) for e in r["events"]]
    # đi qua bàn trái trước (đèn trái bật), sang bàn phải (đèn phải bật, đèn trái tắt sau 1 s), ra khỏi khung -> tắt hết
    assert kinds[0] == ("LIGHT_ON", "d1") and ("LIGHT_ON", "d2") in kinds
    assert kinds.index(("LIGHT_OFF", "d1")) > kinds.index(("LIGHT_ON", "d2"))
    assert ("LIGHT_OFF", "d2") in kinds
    assert r["people_max"] == 1 and r["frames"] == 60
    assert 0 < r["lights_stat"]["d1"]["pct"] < 100 and r["energy_saved_pct"] > 0
    assert r["light_runs"]["d1"][0][1] == 1 and (tmp_path / "f.zip").stat().st_size > 0
    assert r["track_stats"]["born_warmup"] == 1  # chỉ đúng 1 người, không bị nhân đôi
    # vị trí người từng khung (để web phát video gốc mượt và vẽ đè)
    assert len(r["frame_t"]) == len(r["tracks"]) == 60 and r["frame_t"][1] > r["frame_t"][0]
    b = next(fr[0] for fr in r["tracks"] if fr)
    assert len(b) == 6 and all(0 <= v <= 1 for v in b[1:5]) and b[5] in (-1, 0, 1)


def test_people_video_job_roundtrip(tmp_path, live_server):  # noqa: F811
    from ares.config import AresSettings
    from ares.uplink import Uplink
    from ares.video_jobs import VideoJobWorker

    scfg, app = live_server
    db = app.state.db
    sid, key = db.create_site("Máy chủ")
    db.create_user("an.nv", "mat-khau-cua-an", "An", "viewer")
    an = TestClient(app, client=("192.168.1.20", 5000))
    assert an.post("/api/login", json={"user": "an.nv", "password": "mat-khau-cua-an"}, headers=X).status_code == 200
    video = write_video(tmp_path / "phong.avi").read_bytes()
    assert an.post("/api/videos?name=p.avi&kind=xx", content=video, headers=X).status_code == 400
    assert an.post("/api/videos?name=p.avi&kind=people&opts=%7Bxx", content=video, headers=X).status_code == 400
    from urllib.parse import quote
    jid = an.post(f"/api/videos?name=phong.avi&kind=people&opts={quote(json.dumps(LAYOUT))}", content=video, headers=X).json()["id"]
    jb = an.post("/api/videos?name=hanh-vi.avi", content=video, headers=X).json()["id"]
    assert [j["id"] for j in an.get("/api/videos?kind=people").json()["items"]] == [jid]
    assert [j["id"] for j in an.get("/api/videos?kind=behavior").json()["items"]] == [jb]

    cfg = AresSettings(storage_root=tmp_path / "e", uplink_url=f"http://127.0.0.1:{scfg.port}", uplink_key=key,
                       uplink_edge_id="may-chu", uplink_video_width=320)
    up = Uplink(cfg, SimpleNamespace())
    w = VideoJobWorker(cfg, up, models_factory=lambda: (WalkerPose(), None, None))
    assert w.run_once() is True
    j = an.get(f"/api/videos/{jid}").json()
    assert j["status"] == "done", j["message"]
    R = j["result"]
    assert R["kind"] == "people" and R["mode"] == "vùng + đèn" and R["lights_stat"]["d1"]["switches"] >= 2
    assert R["layout"]["zones"][0]["name"] == "Bàn trái" and "d1" in R["light_runs"]
    assert an.get(f"/api/videos/{jid}/thumb.jpg").status_code == 200
    # web phát video gốc (có tua) + vị trí người từng khung
    src = an.get(f"/api/videos/{jid}/source", headers={"Range": "bytes=0-99"})
    assert src.status_code == 206 and len(src.content) == 100
    tr = an.get(f"/api/videos/{jid}/tracks.json").json()
    assert len(tr["t"]) == len(tr["b"]) == R["frames"] and any(tr["b"])
    # đổi bố trí rồi phân tích lại (không tải lại video)
    new = {"zones": LAYOUT["zones"][:1], "lights": LAYOUT["lights"][:1], "off_delay_s": 30}
    assert an.post(f"/api/videos/{jid}/opts", content=json.dumps(new), headers=X).json()["opts"]["off_delay_s"] == 30
    assert an.post(f"/api/videos/{jid}/opts", content=json.dumps(new), headers=X).status_code == 200
    assert an.post(f"/api/videos/{jb}/opts", content="{}", headers=X).status_code in (400, 409)
    assert an.post(f"/api/videos/{jid}/retry", headers=X).json()["ok"]
    assert db.video_job(jid)["opts"]["off_delay_s"] == 30


def test_retrain_command_from_dashboard(tmp_path, live_server):  # noqa: F811
    from ares.config import AresSettings
    from ares.uplink import Uplink

    scfg, app = live_server
    db = app.state.db
    sid, key = db.create_site("Máy chủ")
    db.create_user("admin", "mk-dai-va-kho", "QT", "admin")
    db.create_user("an.nv", "mat-khau-cua-an", "An", "viewer")
    adm = TestClient(app, client=("127.0.0.1", 5000))
    an = TestClient(app, client=("192.168.1.20", 5000))
    adm.post("/api/login", json={"user": "admin", "password": "mk-dai-va-kho"}, headers=X)
    an.post("/api/login", json={"user": "an.nv", "password": "mat-khau-cua-an"}, headers=X)
    assert an.post("/api/admin/retrain", json={"target": "behavior"}, headers=X).status_code == 403
    assert adm.post("/api/admin/retrain", json={"target": "behavior"}, headers=X).status_code == 409  # máy AI chưa chạy

    rt, store = fake_runtime(tmp_path)
    store.update("20261115-010203-abcdef", verdict="false")
    cfg = AresSettings(storage_root=tmp_path / "e", uplink_url=f"http://127.0.0.1:{scfg.port}", uplink_key=key,
                       uplink_edge_id="may-chu", uplink_video_jobs=False)
    up = Uplink(cfg, rt)
    called = {}

    def fake_train(epochs):
        called["epochs"] = epochs
        return {"ok": True, "message": "Model hành vi mới behavior-x.pt: macro-F1 0.80."}
    up.trainer._train_behavior = fake_train
    up.send_heartbeat()
    st = adm.get("/api/admin/retrain").json()
    assert st["ai_online"] and st["state"]["reviewed"] == {"true": 0, "false": 1}
    assert adm.post("/api/admin/retrain", json={"target": "behavior", "epochs": 25}, headers=X).json()["ok"]
    assert adm.post("/api/admin/retrain", json={"target": "behavior"}, headers=X).status_code == 409  # chưa nhận lệnh trước
    up.send_heartbeat()  # nhận lệnh -> train ở luồng nền
    for _ in range(100):
        if not up.trainer.state.get("running"):
            break
        time.sleep(0.02)
    assert called["epochs"] == 25 and up.trainer.state["ok"]
    up.send_heartbeat()  # gửi kết quả + trạng thái
    st = adm.get("/api/admin/retrain").json()
    assert st["state"]["ok"] and "macro-F1" in st["state"]["message"]
    assert st["history"][0]["ok"] == 1 and st["history"][0]["target"] == "behavior"


def test_traffic_mounted_admin_only(tmp_path):
    from ares_server.app import create_app
    from ares_server.config import ServerSettings

    app = create_app(ServerSettings(data_dir=tmp_path / "srv"))
    db = app.state.db
    db.create_user("admin", "mk-dai-va-kho", "QT", "admin")
    db.create_user("an.nv", "mat-khau-cua-an", "An", "viewer")
    anon = TestClient(app, client=("192.168.1.9", 5000))
    assert anon.get("/traffic/").status_code == 401 and anon.get("/traffic/api/info").status_code == 401
    an = TestClient(app, client=("192.168.1.20", 5000))
    an.post("/api/login", json={"user": "an.nv", "password": "mat-khau-cua-an"}, headers=X)
    assert an.get("/traffic/studio").status_code == 401 and an.get("/traffic/api/info").status_code == 403
    adm = TestClient(app, client=("127.0.0.1", 5000))
    tok = adm.post("/api/login", json={"user": "admin", "password": "mk-dai-va-kho"}, headers=X).json()["token"]
    info = adm.get("/api/traffic/info").json()
    assert info["ok"], info
    for p in ("/traffic/", "/traffic/studio", "/traffic/vi-pham"):
        r = adm.get(p)
        assert r.status_code == 200 and "/traffic/api/" in r.text and '"/api/' not in r.text and "'/api/" not in r.text
    assert adm.get("/traffic/api/info").json()["version"]
    assert adm.post("/traffic/api/studio/stop").status_code == 403  # vẫn giữ chống CSRF của gói giao thông
    # mở từ dashboard GitHub Pages bằng token: đặt cookie rồi chuyển trang
    nc = TestClient(app, client=("8.8.8.8", 5000))
    r = nc.get(f"/traffic/?t={tok}", follow_redirects=False)
    assert r.status_code == 303 and "ares_srv" in r.headers.get("set-cookie", "")
