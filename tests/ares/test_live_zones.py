"""Vùng & đèn cho camera thật: máy AI đếm người từng vùng -> server bật/tắt đèn -> lệnh tới mạch (Wi-Fi / USB)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from tests.ares.test_server_uplink import fake_runtime, live_server  # noqa: F401 — fixture

X = {"X-ARES": "1"}
LAY = {"zones": [{"id": "trai", "name": "Bàn trái", "poly": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]},
                 {"id": "phai", "name": "Bàn phải", "poly": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]}],
       "lights": [{"id": "d1", "name": "Đèn trái", "zones": ["trai"], "output": "relay"},
                  {"id": "d2", "name": "Đèn phải", "zones": ["phai"], "output": "led", "level": 60}],
       "off_delay_s": 5}


def setup_db(db):
    db.create_user("an.nv", "mat-khau-cua-an", "An", "viewer")
    db.create_user("binh", "mat-khau-cua-binh", "Bình", "viewer")
    an = next(u for u in db.users() if u["username"] == "an.nv")
    binh = next(u for u in db.users() if u["username"] == "binh")
    pid = db.default_place(an["id"])
    dev = db.add_device(an["id"], pid, "Mạch đèn phòng", "esp32")
    dev_b = db.add_device(binh["id"], db.default_place(binh["id"]), "Mạch của Bình", "usb")
    return an, binh, pid, dev, dev_b


def test_engine_on_off_delay_stale_modes(tmp_path):
    from ares_server.db import ServerDB
    from ares_server.layout import clean_layout
    from ares_server.zones import ZoneEngine

    db = ServerDB(tmp_path / "s.db")
    an, binh, pid, dev, _ = setup_db(db)
    L = clean_layout({**LAY, "lights": [{**LAY["lights"][0], "device_id": dev["id"]},
                                         {**LAY["lights"][1], "device_id": dev["id"]}]})
    key = (1, "may-chu", "cam1")
    db.set_cam_layout(*key, L, "an.nv")
    eng = ZoneEngine(db, stale_s=15)
    t0 = 1000.0
    eng.feed(key, {"trai": 1, "phai": 0}, 1, t0)
    v = eng.view(key, t0)
    assert v["lights"]["d1"]["on"] and not v["lights"]["d2"]["on"] and "1 người ở Bàn trái" in v["lights"]["d1"]["reason"]
    assert db.device(dev["id"])["desired"] == {"relay": 1, "led": 0}
    eng.feed(key, {"trai": 0, "phai": 2}, 2, t0 + 1)  # người sang bàn phải
    d = db.device(dev["id"])["desired"]
    assert d == {"relay": 1, "led": 60}  # đèn trái chưa tắt (trễ 5 giây), LED đèn phải 60 %
    assert "tắt sau" in eng.view(key, t0 + 1)["lights"]["d1"]["reason"]
    eng.feed(key, {"trai": 0, "phai": 2}, 2, t0 + 7)
    assert db.device(dev["id"])["desired"] == {"relay": 0, "led": 60}
    # máy AI ngừng gửi > 15 giây -> đèn Tự động bật để an toàn
    eng.tick(t0 + 30)
    v = eng.view(key, t0 + 30)
    assert v["stale"] and v["lights"]["d1"]["on"] and "an toàn" in v["lights"]["d1"]["reason"]
    # đặt tay Luôn tắt
    L["lights"][0]["mode"] = "off"
    db.set_cam_layout(*key, L, "an.nv")
    eng.reload()
    eng.feed(key, {"trai": 3, "phai": 0}, 3, t0 + 31)
    assert not eng.view(key, t0 + 31)["lights"]["d1"]["on"] and db.device(dev["id"])["desired"]["relay"] == 0
    assert eng.zones_for(1, "may-chu")["cam1"]["zones"][0]["id"] == "trai" and eng.zones_for(1, "khac") == {}
    db.set_cam_layout(*key, {**L, "lights": L["lights"][:1]}, "an.nv")
    eng.reload()
    assert list(eng.view(key, t0 + 31)["lights"]) == ["d1"]  # đèn đã xoá không còn trong trạng thái
    db.set_cam_layout(*key, L, "an.nv")
    eng.reload()
    # người dùng bấm lệnh tay ở mục Cảm biến: engine không ghi đè mỗi giây (chỉ ghi khi trạng thái đèn đổi)
    db.set_device(dev["id"], desired={"relay": 1, "led": 60})
    eng.feed(key, {"trai": 3, "phai": 1}, 4, t0 + 32)
    assert db.device(dev["id"])["desired"]["relay"] == 1


def test_dashboard_api_and_edge_counts(tmp_path, live_server):  # noqa: F811
    from ares.config import AresSettings
    from ares.uplink import Uplink

    scfg, app = live_server
    db = app.state.db
    sid, key = db.create_site("Máy chủ")
    an, binh, pid, dev, dev_b = setup_db(db)
    cam = db.add_client_cam(an["id"], sid, "may-chu", "Webcam phòng", pid)
    ca = TestClient(app, client=("192.168.1.20", 5000))
    cb = TestClient(app, client=("192.168.1.21", 5000))
    ca.post("/api/login", json={"user": "an.nv", "password": "mat-khau-cua-an"}, headers=X)
    cb.post("/api/login", json={"user": "binh", "password": "mat-khau-cua-binh"}, headers=X)
    cams = ca.get("/api/zones").json()["cams"]
    assert [c["cam_id"] for c in cams] == [cam["cam_id"]] and cb.get("/api/zones").json()["cams"] == []
    path = f"/api/zones/{sid}/may-chu/{cam['cam_id']}"
    assert cb.post(path, content=json.dumps(LAY), headers=X).status_code == 404  # camera của người khác
    bad = {**LAY, "lights": [{**LAY["lights"][0], "device_id": dev_b["id"]}]}
    assert ca.post(path, content=json.dumps(bad), headers=X).status_code == 400  # mạch của người khác
    good = {**LAY, "lights": [{**LAY["lights"][0], "device_id": dev["id"]}, LAY["lights"][1]]}
    r = ca.post(path, content=json.dumps(good), headers=X).json()
    assert r["ok"] and r["layout"]["lights"][0]["mode"] == "auto"
    assert ca.get(path).json()["layout"]["zones"][1]["name"] == "Bàn phải"

    # máy AI: heartbeat nhận vùng, đếm người theo điểm giữa chân, gửi lên -> đèn bật
    rt, _ = fake_runtime(tmp_path)
    from ares.behavior.tracker import Tracker
    tr = Tracker()
    now = time.time()
    tr.update(now, np.zeros((2, 17, 3), np.float32), np.array([[50, 100, 120, 400], [500, 100, 560, 300]], np.float32),
              np.array([0.9, 0.9], np.float32), frame_wh=(640, 480))
    rt.behavior.analyzer = SimpleNamespace(cams={cam["cam_id"]: SimpleNamespace(last_t=now, tracker=tr)})
    cfg = AresSettings(storage_root=tmp_path / "e", uplink_url=f"http://127.0.0.1:{scfg.port}", uplink_key=key,
                       uplink_edge_id="may-chu", uplink_video_jobs=False)
    up = Uplink(cfg, rt)
    up.send_heartbeat()
    assert list(up.zone_layouts) == [cam["cam_id"]]
    assert up.zone_counts()[cam["cam_id"]] == {"counts": {"trai": 1, "phai": 1}, "n": 2}
    assert up.send_zones() == 1
    st = ca.get("/api/zones-state").json()["cams"][f"{sid}/may-chu/{cam['cam_id']}"]
    assert st["lights"]["d1"]["on"] and st["counts"] == {"trai": 1, "phai": 1} and not st["stale"]
    assert ca.get(f"/api/devices/{dev['id']}/cmd").json()["desired"]["relay"] == 1  # mạch USB/Wi-Fi nhận lệnh này
    assert cb.get(f"/api/devices/{dev['id']}/cmd").status_code == 404
    # mạch ESP32 gửi số đo -> nhận lệnh bật relay trong câu trả lời
    tok = db.device(dev["id"])["token"]
    rr = TestClient(app).post("/api/v1/sensor", json={"temp": 27.5}, headers={"X-Device-Token": tok}).json()
    assert rr["cmd"]["relay"] == 1
    # xoá hết vùng/đèn -> máy AI thôi gửi
    assert ca.post(path, content="{}", headers=X).json()["ok"]
    up.send_heartbeat()
    assert up.zone_layouts == {} and up.zone_counts() == {}


def test_device_channels_firmware_and_engine(tmp_path, live_server):  # noqa: F811
    scfg, app = live_server
    db = app.state.db
    db.create_user("an.nv", "mat-khau-cua-an", "An", "viewer")
    c = TestClient(app, client=("192.168.1.20", 5000))
    c.post("/api/login", json={"user": "an.nv", "password": "mat-khau-cua-an"}, headers=X)
    chs = [{"name": "Đèn bàn trái"}, {"name": "Đèn máy cưa", "type": "relay"}, {"name": "Đèn 3", "pin": 3}, {"name": "Đèn 4"}]
    did = c.post("/api/devices", json={"name": "Mạch 4 đèn", "kind": "usb", "channels": chs}, headers=X).json()["id"]
    d = next(x for x in c.get("/api/devices").json()["items"] if x["id"] == did)
    assert [x["key"] for x in d["channels"]] == ["ch1", "ch2", "ch3", "ch4"] and d["channels"][1]["type"] == "relay"
    code = c.get(f"/api/devices/{did}/firmware.ino").text
    assert "#define N_CH 4" in code and "CH_PIN[N_CH] = {9, 7, 3, 10}" in code and "CH_PWM[N_CH] = {1, 0, 1, 1}" in code
    assert "{{CHANNELS}}" not in code
    # sửa số đèn sau khi đã thêm
    r = c.post(f"/api/devices/{did}", json={"channels": chs[:2]}, headers=X).json()
    assert len(r["channels"]) == 2 and "#define N_CH 2" in c.get(f"/api/devices/{did}/firmware.ino").text
    c.post(f"/api/devices/{did}", json={"channels": chs}, headers=X)
    # lệnh tay cho từng kênh
    assert c.post(f"/api/devices/{did}/cmd", json={"ch1": 40, "ch2": 1, "ch9": 5}, headers=X).json()["desired"] == {"ch1": 40, "ch2": 1}
    # đèn theo vùng dùng kênh: LED -> độ sáng, relay -> 1
    from ares_server.zones import ZoneEngine
    from ares_server.layout import clean_layout
    L = clean_layout({"zones": LAY["zones"], "lights": [
        {"id": "a", "zones": ["trai"], "device_id": did, "output": "ch1", "level": 70},
        {"id": "b", "zones": ["phai"], "device_id": did, "output": "ch2"},
        {"id": "c", "zones": ["phai"], "device_id": did, "output": "ch9"}]})
    assert [li["output"] for li in L["lights"]] == ["ch1", "ch2", "relay"]  # ch9 không hợp lệ
    key = (1, "may-chu", "cam")
    db.set_cam_layout(*key, L, "an.nv")
    eng = ZoneEngine(db)
    eng.feed(key, {"trai": 1, "phai": 1}, 2, 100.0)
    des = db.device(did)["desired"]
    assert des["ch1"] == 70 and des["ch2"] == 1
    # mạch đã nạp code cũ (không có kênh) vẫn nhận số đo kênh mạch báo lên
    tok = db.device(did)["token"]
    assert TestClient(app).post("/api/v1/sensor", json={"ch1": 70, "light": 33}, headers={"X-Device-Token": tok}).status_code == 200
