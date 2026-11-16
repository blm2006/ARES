"""Server trung tâm (ares_server) + kênh gửi từ Edge (ares/uplink.py)."""

from __future__ import annotations

import io
import json
import socket
import threading
import time
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ares_server.app import check_exposure, create_app
from ares_server.config import ServerSettings

JPG = b"\xff\xd8\xff\xe0" + b"0" * 200


def frames_zip(n=3) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for i in range(1, n + 1):
            z.writestr(f"{i:04d}.jpg", JPG)
    return b.getvalue()


def make(tmp_path, **kw):
    cfg = ServerSettings(data_dir=tmp_path / "srv", **kw)
    app = create_app(cfg)
    return cfg, app, TestClient(app)


def test_server_flow_local_mode(tmp_path):
    cfg, app, c = make(tmp_path)
    db = app.state.db
    sid, key = db.create_site("Trường A", "Q.1")
    H = {"Authorization": f"Bearer {key}", "X-Edge-Id": "may-1"}
    assert c.get("/api/v1/ping", headers={"Authorization": "Bearer sai"}).status_code == 401
    assert c.get("/api/v1/ping", headers=H).json()["site"] == "Trường A"
    hb = {"edge": {"name": "Phòng 101", "version": "0.1.0"},
          "cameras": [{"id": "main", "name": "Phòng 101", "online": True, "n_people": 7, "fps": 9.5, "behavior": True},
                      {"id": "usb1", "name": "Hành lang", "online": False, "n_people": 3}],
          "ai": {"behavior": {"running": True, "mode": "hybrid"}},
          "integrations": {"lighting": {"enabled": True, "avg_pwm": 55.0}, "ac": {"enabled": False}}}
    assert c.post("/api/v1/heartbeat", headers=H, content=json.dumps(hb)).json()["ok"]
    meta = {"id": "20261115-010203-abcdef", "t": time.time(), "cam_id": "main", "cam_name": "Phòng 101", "kind": "FALL",
            "label": "Té ngã", "severity": "error", "message": "Té ngã — camera 'Phòng 101'"}
    r = c.post("/api/v1/events", headers=H, data={"meta": json.dumps(meta)},
               files={"snapshot": ("s.jpg", JPG, "image/jpeg"), "frames": ("f.zip", frames_zip(), "application/zip")})
    assert r.status_code == 200 and r.json()["new"] is True
    eid = r.json()["id"]
    # gửi lại (mạng chập chờn) không tạo bản trùng
    assert c.post("/api/v1/events", headers=H, data={"meta": json.dumps(meta)}).json() == {"ok": True, "id": eid, "new": False}
    # file sai định dạng bị từ chối
    bad = dict(meta, id="x2")
    assert c.post("/api/v1/events", headers=H, data={"meta": json.dumps(bad)},
                  files={"snapshot": ("s.jpg", b"GIF89a....", "image/jpeg")}).status_code == 415

    # dashboard luôn cần đăng nhập
    assert c.get("/api/overview").status_code == 401
    db.create_user("quanly", "mk-dai-va-kho", "Quản lý", "admin")
    assert c.post("/api/login", json={"user": "quanly", "password": "mk-dai-va-kho"}, headers={"X-ARES": "1"}).status_code == 200
    o = c.get("/api/overview").json()
    assert o["totals"]["people"] == 7 and o["totals"]["cams_online"] == 1 and o["totals"]["places"] == 1
    assert o["totals"]["new_alerts"] == 1 and o["today"] == {"FALL": 1}
    pl = o["places"][0]  # camera chưa gán -> vào địa điểm mặc định của quản trị viên (mang tên người đó)
    assert pl["name"] == "Quản lý" and pl["cams_total"] == 2 and pl["cams_online"] == 1 and pl["people"] == 7
    s = c.get("/api/sites").json()["items"][0]
    assert s["edges_online"] == 1 and s["behavior_state"] == "running" and "lighting" in s["integrations"] and "ac" not in s["integrations"]
    ev = c.get(f"/api/events/{eid}").json()
    assert ev["n_frames"] == 3 and ev["site_name"] == "Trường A"
    assert c.get(f"/api/events/{eid}/snapshot.jpg").content == JPG
    assert c.get(f"/api/events/{eid}/frames/2.jpg").status_code == 200
    assert c.get(f"/api/events/{eid}/clip.mp4").status_code == 404
    # POST từ dashboard phải có X-ARES (chống CSRF)
    assert c.post(f"/api/events/{eid}/verdict", json={"verdict": "true"}).status_code == 403
    assert c.post(f"/api/events/{eid}/verdict", json={"verdict": "true"}, headers={"X-ARES": "1"}).json() == {"ok": True}
    assert c.get(f"/api/events/{eid}").json()["status"] == "ack"
    d = c.get(f"/api/sites/{sid}").json()
    assert d["edges"][0]["online"] and len(d["cameras"]) == 2 and d["series"]
    # tạo địa điểm + đổi khoá từ dashboard
    r = c.post("/api/sites", json={"name": "Phòng khám B"}, headers={"X-ARES": "1"}).json()
    k2 = r["key"]
    assert c.get("/api/v1/ping", headers={"Authorization": f"Bearer {k2}"}).status_code == 200
    k3 = c.post(f"/api/sites/{r['id']}/rotate-key", headers={"X-ARES": "1"}).json()["key"]
    assert c.get("/api/v1/ping", headers={"Authorization": f"Bearer {k2}"}).status_code == 401
    assert c.get("/api/v1/ping", headers={"Authorization": f"Bearer {k3}"}).status_code == 200
    assert c.get("/").status_code == 200


def test_accounts_setup_login_register_roles(tmp_path):
    from fastapi.testclient import TestClient

    X = {"X-ARES": "1"}
    cfg = ServerSettings(data_dir=tmp_path / "srv", host="0.0.0.0")
    check_exposure(cfg)  # không còn bắt buộc mật khẩu trong file cấu hình: tài khoản tạo trên dashboard
    app = create_app(cfg)
    remote = TestClient(app, client=("192.168.1.50", 5000))
    local = TestClient(app, client=("127.0.0.1", 5000))
    # chưa có tài khoản: máy khác KHÔNG tạo được quản trị viên, chỉ máy chủ tạo được
    assert remote.get("/api/me").json() | {"version": 0} == {"user": None, "need_setup": True, "can_setup": False,
                                                              "pending": 0, "version": 0}
    acc = {"username": "tri", "password": "mk-dai-va-kho", "name": "Trí"}
    assert remote.post("/api/setup", json=acc, headers=X).status_code == 403
    assert remote.post("/api/register", json=acc, headers=X).status_code == 409
    assert local.post("/api/setup", json=acc).status_code == 403  # thiếu X-ARES (chống CSRF)
    assert local.post("/api/setup", json=dict(acc, password="12345678"), headers=X).status_code == 400  # quá dễ
    r = local.post("/api/setup", json=acc, headers=X)
    assert r.status_code == 200 and r.json()["user"]["role"] == "admin"
    assert local.get("/api/overview").status_code == 200  # đã đăng nhập luôn
    assert local.post("/api/setup", json=dict(acc, username="khac"), headers=X).status_code == 409
    raw = app.state.db._q("SELECT pw_hash FROM users")[0]["pw_hash"]
    assert raw.startswith("pbkdf2_sha256$") and "mk-dai-va-kho" not in raw

    # máy khác tự đăng ký -> chờ duyệt -> chưa đăng nhập được
    reg = {"username": "baove1", "password": "bao-ve-ca-dem", "name": "Bảo vệ ca đêm"}
    assert remote.post("/api/register", json=reg, headers=X).json()["ok"]
    assert remote.post("/api/register", json=reg, headers=X).status_code == 400  # trùng tên
    r = remote.post("/api/login", json={"user": "baove1", "password": "bao-ve-ca-dem"}, headers=X)
    assert r.status_code == 401 and "chờ" in r.json()["detail"]
    users = local.get("/api/users").json()["items"]
    bv = next(u for u in users if u["username"] == "baove1")
    assert bv["status"] == "pending" and bv["role"] == "viewer" and local.get("/api/me").json()["pending"] == 1
    assert local.post(f"/api/users/{bv['id']}", json={"status": "active"}, headers=X).json()["ok"]
    assert remote.post("/api/login", json={"user": "BAOVE1", "password": "bao-ve-ca-dem"}, headers=X).status_code == 200

    # người xem: xem + xử lý cảnh báo được, không vào quản trị
    assert remote.get("/api/overview").status_code == 200
    assert remote.get("/api/users").status_code == 403
    assert remote.post("/api/sites", json={"name": "X"}, headers=X).status_code == 403
    assert remote.post("/api/me/password", json={"old": "sai-roi-nhe", "password": "mat-khau-moi-1"}, headers=X).status_code == 400
    assert remote.post("/api/me/password", json={"old": "bao-ve-ca-dem", "password": "mat-khau-moi-1"}, headers=X).json()["ok"]

    # khoá tài khoản: phiên đang mở mất hiệu lực ngay
    assert local.post(f"/api/users/{bv['id']}", json={"status": "disabled"}, headers=X).json()["ok"]
    assert remote.get("/api/overview").status_code == 401
    me_id = local.get("/api/me").json()["user"]["id"]
    assert local.post(f"/api/users/{me_id}", json={"role": "viewer"}, headers=X).status_code == 400  # không tự hạ quyền
    assert local.post(f"/api/users/{me_id}/delete", headers=X).status_code == 400
    assert local.post("/api/users", json={"username": "ql2", "password": "quan-ly-so-2", "role": "admin"}, headers=X).json()["ok"]
    assert local.post(f"/api/users/{bv['id']}/delete", headers=X).json()["ok"]
    remote.cookies.set("ares_srv", "gia.mao")
    assert remote.get("/api/overview").status_code == 401


def test_env_password_becomes_admin_account(tmp_path):
    cfg, app, c = make(tmp_path, host="0.0.0.0", admin_user="admin", admin_password="mk-dai-va-kho")
    assert c.get("/api/me").json()["need_setup"] is False
    assert c.post("/api/login", json={"user": "admin", "password": "sai-mat-khau"}, headers={"X-ARES": "1"}).status_code == 401
    assert c.post("/api/login", json={"user": "admin", "password": "mk-dai-va-kho"}, headers={"X-ARES": "1"}).status_code == 200
    assert c.get("/api/users").json()["items"][0]["role"] == "admin"


# ------------------------------------------------------------------ Edge -> server qua HTTP thật
def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def live_server(tmp_path):
    import uvicorn

    cfg = ServerSettings(data_dir=tmp_path / "srv", port=free_port())
    app = create_app(cfg)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=cfg.port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield cfg, app
    srv.should_exit = True
    th.join(timeout=5)


def fake_runtime(tmp_path):
    from ares.behavior.store import EventStore
    from ares.state import AcState, LightingState

    root = tmp_path / "edge"
    store = EventStore(root)
    d = root / "events" / "20261115" / "20261115-010203-abcdef"
    (d / "frames").mkdir(parents=True)
    (d / "snapshot.jpg").write_bytes(JPG)
    for i in (1, 2):
        (d / "frames" / f"{i:04d}.jpg").write_bytes(JPG)
    (d / "meta.json").write_text(json.dumps({"detail": {"still_s": 30}}), encoding="utf-8")
    store.add({"id": "20261115-010203-abcdef", "t": time.time(), "ts": "2026-11-15T01:02:03", "cam_id": "main",
               "cam_name": "Phòng 101", "kind": "LYING_STILL", "label": "Ngã gục / nằm bất động", "severity": "error",
               "message": "Nằm bất động", "track_ids": "[1]", "score": 1.0, "source": "rules", "status": "new",
               "verdict": "", "dir": "events/20261115/20261115-010203-abcdef", "clip_ready": 1, "n_frames": 2})
    beh = SimpleNamespace(store=store, root=root, loading=False, error="", classifier_info=None,
                          stats={"loop_fps": 9.0}, wants=lambda c: True)
    state = SimpleNamespace(get_lighting=lambda: LightingState(pwm=[40.0, 60.0]), get_ac=lambda: AcState(power=1, temp=26, known=True),
                            get_sensors=lambda: None, mqtt_connected=True, model_version="base")
    hub = SimpleNamespace(control_name="Phòng 101", overview=lambda: {"cameras": [
        {"id": "main", "name": "Phòng 101", "online": True, "n_people": 4, "fps": 10.0}]})
    clock = SimpleNamespace(time=time.time)
    return SimpleNamespace(behavior=beh, state=state, hub=hub, sim=False, started_at=time.time() - 60, clock=clock), store


def test_uplink_end_to_end_and_offline_queue(tmp_path, live_server):
    from ares.config import AresSettings
    from ares.uplink import Uplink

    scfg, app = live_server
    sid, key = app.state.db.create_site("Trường A")
    rt, store = fake_runtime(tmp_path)
    emitted = []

    # 1) server tắt -> Edge không lỗi, sự kiện vẫn chờ, có cảnh báo mất kết nối
    cfg_off = AresSettings(storage_root=tmp_path / "e", uplink_url=f"http://127.0.0.1:{free_port()}", uplink_key=key,
                           uplink_edge_id="may-1", uplink_timeout_s=2)
    up = Uplink(cfg_off, rt, emit=emitted.append)
    up.tick()
    assert up.last_error and store.get("20261115-010203-abcdef")["uplinked"] == 0
    assert [e.kind for e in emitted] == ["UPLINK_DOWN"]

    # 2) có server -> heartbeat + gửi bù sự kiện kèm ảnh và clip
    cfg = AresSettings(storage_root=tmp_path / "e", uplink_url=f"http://127.0.0.1:{scfg.port}", uplink_key=key,
                       uplink_edge_id="may-1")
    up = Uplink(cfg, rt, emit=emitted.append)
    up._was_down = True
    up.tick()
    assert up.last_error == "", up.last_error
    assert store.get("20261115-010203-abcdef")["uplinked"] == 1
    assert emitted[-1].kind == "UPLINK_UP"
    db = app.state.db
    ev = db.events()[0]
    assert ev["kind"] == "LYING_STILL" and ev["edge_id"] == "may-1" and ev["snapshot"] and ev["n_frames"] == 2
    cams = db.cameras(sid)
    assert cams[0]["n_people"] == 4
    info = db.edges(sid)[0]["info"]
    assert info["integrations"]["lighting"]["avg_pwm"] == 50.0 and info["integrations"]["ac"]["temp"] == 26

    # 2b) người trực đánh giá trên server -> lần heartbeat sau Edge nhận lại (dữ liệu train)
    db.handle_event(ev["id"], "admin", "false")
    up.send_heartbeat()
    row = store.get("20261115-010203-abcdef")
    assert row["verdict"] == "false" and row["status"] == "ack"
    assert db.pop_updates(sid, "may-1") == []  # đã đồng bộ, không gửi lại

    # 3) chế độ không gửi ảnh: chỉ thông tin
    store.add({"id": "20261115-010300-bbbbbb", "t": time.time(), "cam_id": "main", "cam_name": "Phòng 101", "kind": "FALL",
               "label": "Té ngã", "severity": "error", "message": "Té ngã", "track_ids": "[2]", "score": 0.9,
               "source": "rules", "dir": "events/20261115/20261115-010203-abcdef", "clip_ready": 1})
    up2 = Uplink(AresSettings(storage_root=tmp_path / "e", uplink_url=cfg.uplink_url, uplink_key=key, uplink_media="none"),
                 rt)
    assert up2.send_events() == 1
    e2 = [e for e in db.events() if e["kind"] == "FALL"][0]
    assert e2["snapshot"] == "" and e2["n_frames"] == 0

    # 4) khoá sai -> lỗi rõ ràng, không đánh dấu đã gửi
    store.add({"id": "20261115-010400-cccccc", "t": time.time(), "cam_id": "main", "kind": "FIGHT", "track_ids": "[1,2]",
               "dir": "x", "clip_ready": 1})
    bad = Uplink(AresSettings(storage_root=tmp_path / "e", uplink_url=cfg.uplink_url, uplink_key="ares_sai"), rt,
                 emit=emitted.append)
    bad.tick()
    assert "401" in bad.last_error and store.get("20261115-010400-cccccc")["uplinked"] == 0
    assert "ARES_UPLINK_KEY" in emitted[-1].message


# ------------------------------------------------------------------ chế độ máy chủ: hình trực tiếp, camera từ xa, app xem
def test_live_commands_and_viewer_download(tmp_path):
    cfg, app, c = make(tmp_path, host="0.0.0.0", admin_password="mk-dai-va-kho", discovery_port=free_port())
    db = app.state.db
    sid, key = db.create_site("Máy chủ")
    H = {"Authorization": f"Bearer {key}", "X-Edge-Id": "may-chu"}
    hb = {"edge": {"name": "Máy chủ"}, "cameras": [{"id": "main", "name": "Phòng 101", "online": True, "n_people": 2}]}
    with c:
        # app xem: tải không cần đăng nhập, file cấu hình chứa đúng địa chỉ máy khách đã dùng để vào
        assert c.get("/tai-app").status_code == 200
        info = c.get("/api/app-info").json()
        assert info["exposed"] is True and "windows" in info["available"]
        r = c.get("/tai-app/ARES-Viewer-Windows.zip", headers={"Host": "192.168.1.10:8200"})
        if info["available"]["windows"]:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            assert json.loads(z.read("ARES Viewer/ares_viewer.json"))["server"] == "http://192.168.1.10:8200"
            assert z.getinfo("ARES Viewer/ARES Viewer.exe").file_size > 100_000
        assert c.get("/tai-app/ARES-Viewer-Plan9.zip").status_code == 404

        # dò tìm máy chủ qua UDP (app ARES Viewer dùng khi máy chủ đổi IP)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(3)
        for _ in range(30):
            s.sendto(b"ARES_DISCOVER", ("127.0.0.1", cfg.discovery_port))
            try:
                data, _ = s.recvfrom(512)
                break
            except socket.timeout:
                continue
        assert json.loads(data)["port"] == cfg.port
        s.close()

        assert c.post("/api/v1/heartbeat", headers=H, content=json.dumps(hb)).json()["commands"] == []
        # hình trực tiếp: Edge gửi, dashboard (đã đăng nhập) xem
        assert c.post("/api/v1/live", headers=H, files={"cam_main": ("main.jpg", JPG, "image/jpeg"),
                                                         "cam_bad": ("x.jpg", b"not-a-jpeg", "image/jpeg")}).json()["n"] == 1
        assert c.get("/api/live").status_code == 401
        c.post("/api/login", json={"user": "admin", "password": "mk-dai-va-kho"}, headers={"X-ARES": "1"})
        items = c.get("/api/live").json()["items"]
        assert [(i["cam_id"], i["name"]) for i in items] == [("main", "Phòng 101")]
        assert c.get(f"/api/live/{sid}/may-chu/main.jpg").content == JPG

        # thêm camera RTSP từ dashboard -> lệnh xuống Edge qua heartbeat -> kết quả quay về
        X = {"X-ARES": "1"}
        assert c.post(f"/api/sites/{sid}/edges/may-chu/cameras", json={"name": "Cổng", "source": "file:///etc/passwd"},
                      headers=X).status_code == 400
        assert c.post(f"/api/sites/{sid}/edges/khong-co/cameras", json={"name": "Cổng", "source": "0"},
                      headers=X).status_code == 404
        assert c.post(f"/api/sites/{sid}/edges/may-chu/cameras",
                      json={"name": "Cổng trường", "source": "rtsp://u:p@192.168.1.64:554/s1"}, headers=X).json()["ok"]
        cmds = c.post("/api/v1/heartbeat", headers=H, content=json.dumps(hb)).json()["commands"]
        assert [x["type"] for x in cmds] == ["add_camera"]
        assert c.post("/api/v1/heartbeat", headers=H, content=json.dumps(hb)).json()["commands"] == []  # không giao trùng
        res = dict(hb, command_results=[{"id": cmds[0]["id"], "ok": True, "message": "Đã thêm camera 'Cổng trường'"}])
        c.post("/api/v1/heartbeat", headers=H, content=json.dumps(res))
        d = c.get(f"/api/sites/{sid}").json()
        assert d["commands"][0]["ok"] == 1 and d["live"] == ["main"]


def test_uplink_runs_camera_commands_and_live(tmp_path):
    from ares.cameras import CameraSpec
    from ares.config import AresSettings
    from ares.uplink import Uplink

    rt, store = fake_runtime(tmp_path)
    added, removed = [], []

    def add(spec: CameraSpec):
        added.append(spec)
        return spec.source != "rtsp://hong"
    rt.hub.add, rt.hub.remove, rt.hub.specs = add, removed.append, (lambda: added)
    cfg = AresSettings(storage_root=tmp_path / "e", cameras_file=tmp_path / "cams.json", uplink_url="http://x", uplink_key="k")
    up = Uplink(cfg, rt)
    r = up.run_command({"id": 1, "type": "add_camera", "name": "Cổng trường", "source": "rtsp://u:p@1.2.3.4/s"})
    assert r["ok"] and added[0].kind == "rtsp"
    saved = json.loads((tmp_path / "cams.json").read_text(encoding="utf-8"))
    assert saved["cameras"][0]["source"] == "rtsp://u:p@1.2.3.4/s"
    cam_id = saved["cameras"][0]["id"]
    assert cam_id == "Congtruong"
    assert not up.run_command({"id": 2, "type": "add_camera", "name": "Hỏng", "source": "rtsp://hong"})["ok"]
    assert not up.run_command({"id": 3, "type": "remove_camera", "cam_id": "main"})["ok"]
    assert up.run_command({"id": 4, "type": "remove_camera", "cam_id": cam_id})["ok"] and removed == [cam_id]
    assert json.loads((tmp_path / "cams.json").read_text(encoding="utf-8"))["cameras"] == []
    off = Uplink(AresSettings(storage_root=tmp_path / "e", uplink_remote_cameras=False, uplink_url="http://x", uplink_key="k"), rt)
    assert not off.run_command({"id": 5, "type": "add_camera", "name": "a", "source": "0"})["ok"]

    # hình trực tiếp: lấy ảnh camera, thu nhỏ
    import cv2
    big = np.zeros((720, 1280, 3), np.uint8)
    jpg = cv2.imencode(".jpg", big)[1].tobytes()
    rt.behavior.frame = lambda cid: (jpg, time.time())
    frames = Uplink(AresSettings(storage_root=tmp_path / "e", uplink_live_width=320, uplink_url="http://x", uplink_key="k"),
                    rt).live_frames()
    img = cv2.imdecode(np.frombuffer(frames["main"], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 320
    # khung không đổi thì không gửi lại (đỡ tốn mạng)
    up2 = Uplink(AresSettings(storage_root=tmp_path / "e", uplink_live_width=640, uplink_url="http://x", uplink_key="k"), rt)
    t0 = time.time()
    rt.behavior.frame = lambda cid: (jpg, t0)
    assert "main" in up2.live_frames() and "main" not in up2.live_frames()
    # không có ảnh AI hành vi -> ảnh khung người từ camera chính, KHÔNG vẽ vùng đèn (vùng chia khung)
    from ares.detector import Det
    rt.behavior.frame = lambda cid: (None, 0.0)
    raw = np.zeros((360, 640, 3), np.uint8)
    rt.tracker = SimpleNamespace(last_raw=(raw, [Det(10, 10, 100, 200, 0.9)], time.time()))
    calls = []
    import ares.cameras as camsmod
    orig = camsmod.render_boxes
    camsmod.render_boxes = lambda img_, dets, cfg_: (calls.append(len(dets)), orig(img_, dets, cfg_))[1]
    try:
        out = up2.live_frames()
    finally:
        camsmod.render_boxes = orig
    assert calls == [1] and out["main"][:3] == b"\xff\xd8\xff"


# ------------------------------------------------------------------ máy khác = app: camera của họ, AI trên máy chủ, chỉ thấy của mình
def test_client_camera_flow_and_scoping(tmp_path, live_server):
    import cv2
    from ares.cameras import CameraSpec, guess_kind
    from ares.config import AresSettings
    from ares.remote_cam import AresPullSource

    scfg, app = live_server
    db = app.state.db
    X = {"X-ARES": "1"}
    sid, key = db.create_site("Máy chủ")
    H = {"Authorization": f"Bearer {key}", "X-Edge-Id": "may-chu"}
    base = f"http://127.0.0.1:{scfg.port}"
    from fastapi.testclient import TestClient
    admin, khach, khac = (TestClient(app, client=(ip, 5000)) for ip in ("127.0.0.1", "192.168.1.20", "192.168.1.30"))
    db.create_user("tri", "mk-dai-va-kho", "Trí", "admin")
    db.create_user("an.nv", "mat-khau-cua-an", "An", "viewer")
    db.create_user("binh", "mat-khau-cua-binh", "Bình", "viewer")
    for c, u, p in ((admin, "tri", "mk-dai-va-kho"), (khach, "an.nv", "mat-khau-cua-an"), (khac, "binh", "mat-khau-cua-binh")):
        assert c.post("/api/login", json={"user": u, "password": p}, headers=X).status_code == 200

    # chưa có máy chủ AI trực tuyến -> báo rõ
    assert khach.post("/api/my/cameras", json={"name": "Phòng khách"}, headers=X).status_code == 409
    hb = {"edge": {"name": "Máy chủ"}, "cameras": [{"id": "main", "name": "Phòng 101", "online": True, "n_people": 3}]}
    TestClient(app).post("/api/v1/heartbeat", headers=H, content=json.dumps(hb))
    r = khach.post("/api/my/cameras", json={"name": "Phòng khách"}, headers=X).json()
    cam_id, cid = r["cam_id"], r["id"]
    cmds = TestClient(app).post("/api/v1/heartbeat", headers=H, content=json.dumps(hb)).json()["commands"]
    assert cmds[0]["type"] == "add_camera" and cmds[0]["cam_id"] == cam_id and cmds[0]["source"].startswith("ares://")
    assert guess_kind(cmds[0]["source"]) == "remote"

    # máy khác gửi webcam -> Edge kéo về bằng khoá thiết bị (qua HTTP thật)
    img = np.full((240, 320, 3), 128, np.uint8)
    jpg = cv2.imencode(".jpg", img)[1].tobytes()
    assert khac.post(f"/api/my/cameras/{cid}/frame", content=jpg, headers=X).status_code == 404  # không phải camera của Bình
    assert khach.post(f"/api/my/cameras/{cid}/frame", content=b"xx", headers=X).status_code == 415
    assert khach.post(f"/api/my/cameras/{cid}/frame", content=jpg, headers=X).json()["ok"]
    ecfg = AresSettings(storage_root=tmp_path / "e", uplink_url=base, uplink_key=key, uplink_edge_id="may-chu")
    src = AresPullSource(ecfg, CameraSpec(cam_id, "Phòng khách", cmds[0]["source"], "remote"), fps=20)
    got = None
    for _ in range(50):
        got = src.read()
        if got is not None:
            break
        time.sleep(0.05)
    assert got is not None and got.shape == (240, 320, 3)
    assert src.read() is None  # không trả lại khung cũ
    src.close()
    bad = urllib_get(f"{base}/api/v1/frame/{cmds[0]['source'][7:]}", {"Authorization": "Bearer sai", "X-Edge-Id": "may-chu"})
    assert bad == 401

    # Edge báo camera đã chạy AI + gửi hình kết quả + sự kiện -> chỉ An (và quản trị) thấy
    hb2 = dict(hb, cameras=hb["cameras"] + [{"id": cam_id, "name": "Phòng khách · An", "online": True, "n_people": 2}])
    TestClient(app).post("/api/v1/heartbeat", headers=H, content=json.dumps(hb2))
    TestClient(app).post("/api/v1/live", headers=H, files={f"cam_{cam_id}": ("a.jpg", JPG, "image/jpeg"),
                                                           "cam_main": ("b.jpg", JPG, "image/jpeg")})
    for i, cam in enumerate((cam_id, "main")):
        TestClient(app).post("/api/v1/events", headers=H, data={"meta": json.dumps(
            {"id": f"20261115-11000{i}-aaaaa{i}", "t": time.time(), "cam_id": cam, "cam_name": cam, "kind": "FALL",
             "label": "Té ngã", "severity": "error", "message": "Té ngã"})})
    mine = khach.get("/api/my/cameras").json()["items"]
    assert mine[0]["sending"] and mine[0]["ai_online"] and mine[0]["n_people"] == 2 and mine[0]["live"]
    o = khach.get("/api/overview").json()
    assert o["totals"]["people"] == 2 and o["totals"]["new_alerts"] == 1 and o["ai"]["online"]
    assert [p["name"] for p in o["places"]] == ["An"] and o["places"][0]["cams_online"] == 1  # chỉ địa điểm của mình
    assert [e["cam_id"] for e in khach.get("/api/events").json()["items"]] == [cam_id]
    assert khac.get("/api/events").json()["items"] == [] and khac.get("/api/live").json()["items"] == []
    assert [i["cam_id"] for i in khach.get("/api/live").json()["items"]] == [cam_id]
    main_ev = next(e for e in admin.get("/api/events").json()["items"] if e["cam_id"] == "main")
    assert khach.get(f"/api/events/{main_ev['id']}").status_code == 404
    assert khach.post(f"/api/events/{main_ev['id']}/ack", headers=X).status_code == 404
    assert khach.get(f"/api/live/{sid}/may-chu/main.jpg").status_code == 404
    assert khach.get(f"/api/live/{sid}/may-chu/{cam_id}.jpg").content == JPG
    for path in ("/api/sites", f"/api/sites/{sid}", "/api/users"):
        assert khach.get(path).status_code == 403
    assert len(admin.get("/api/events").json()["items"]) == 2  # quản trị thấy tất cả
    an = next(u for u in admin.get("/api/users").json()["items"] if u["username"] == "an.nv")
    assert an["n_places"] == 1 and an["cams_online"] == 1 and an["people"] == 2 and an["new_alerts"] == 1
    assert khach.post("/api/events/ack-all", headers=X).json()["acked"] == 1  # chỉ cảnh báo của mình
    assert any(e["status"] == "new" for e in admin.get("/api/events").json()["items"])

    # gỡ camera -> lệnh gỡ xuống Edge
    assert khach.post(f"/api/my/cameras/{cid}/delete", headers=X).json()["ok"]
    cmds = TestClient(app).post("/api/v1/heartbeat", headers=H, content=json.dumps(hb)).json()["commands"]
    assert [c["type"] for c in cmds] == ["remove_camera"] and cmds[0]["cam_id"] == cam_id


def urllib_get(url, headers):
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=3) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


# ------------------------------------------------------------------ địa điểm chia theo người dùng
def test_places_per_user(tmp_path):
    from fastapi.testclient import TestClient

    from ares_server.app import create_app
    from ares_server.config import ServerSettings
    from ares_server.db import ServerDB

    scfg = ServerSettings(data_dir=tmp_path / "srv")
    db = ServerDB(scfg.data_dir / "s.db")
    app = create_app(scfg, db)
    X = {"X-ARES": "1"}
    sid, key = db.create_site("Máy chủ")
    H = {"Authorization": f"Bearer {key}", "X-Edge-Id": "may-chu"}
    admin, an, binh = (TestClient(app, client=(ip, 5000)) for ip in ("127.0.0.1", "192.168.1.20", "192.168.1.30"))
    db.create_user("tri", "mk-dai-va-kho", "Trí", "admin")
    db.create_user("an.nv", "mat-khau-cua-an", "An", "viewer")
    db.create_user("binh", "mat-khau-cua-binh", "Bình", "viewer")
    for c, u, p in ((admin, "tri", "mk-dai-va-kho"), (an, "an.nv", "mat-khau-cua-an"), (binh, "binh", "mat-khau-cua-binh")):
        assert c.post("/api/login", json={"user": u, "password": p}, headers=X).status_code == 200
    hb = {"edge": {"name": "Máy chủ"}, "cameras": [{"id": "main", "name": "Phòng 101", "online": True, "n_people": 3}]}
    TestClient(app).post("/api/v1/heartbeat", headers=H, content=json.dumps(hb))

    # mỗi tài khoản có sẵn 1 địa điểm mang tên mình; người dùng chỉ thấy địa điểm của mình
    assert [p["name"] for p in an.get("/api/places").json()["items"]] == ["An"]
    assert len(admin.get("/api/places").json()["items"]) == 3
    # An tạo thêm "Nhà bà ngoại" và gắn camera vào đó; mở lại app cùng tên camera -> dùng lại, không tạo trùng
    pid = an.post("/api/places", json={"name": "Nhà bà ngoại"}, headers=X).json()["id"]
    r1 = an.post("/api/my/cameras", json={"name": "Phòng khách", "place_id": pid}, headers=X).json()
    r2 = an.post("/api/my/cameras", json={"name": "phòng khách"}, headers=X).json()
    assert r2["id"] == r1["id"] and r2["reused"] and len(db.client_cams()) == 1
    assert binh.post("/api/my/cameras", json={"name": "X", "place_id": pid}, headers=X).status_code == 404  # không phải của Bình
    # camera của máy chủ chưa gán -> địa điểm của quản trị viên, không lẫn vào của An
    o = an.get("/api/overview").json()
    assert {p["name"]: p["cams_total"] for p in o["places"]} == {"An": 0, "Nhà bà ngoại": 1}
    ov = admin.get("/api/overview").json()
    tri = next(p for p in ov["places"] if p["owner_username"] == "tri")
    assert tri["people"] == 3 and tri["cams_online"] == 1 and ov["totals"]["users"] == 3
    # quản trị viên chuyển camera Phòng 101 sang địa điểm của Bình -> Bình thấy, An không
    pb = binh.get("/api/places").json()["items"][0]["id"]
    assert an.post("/api/places/assign", json={"site_id": sid, "edge_id": "may-chu", "cam_id": "main", "place_id": pid},
                   headers=X).status_code == 404
    assert admin.post("/api/places/assign", json={"site_id": sid, "edge_id": "may-chu", "cam_id": "main", "place_id": pb},
                      headers=X).json()["ok"]
    assert binh.get("/api/overview").json()["totals"]["people"] == 3
    TestClient(app).post("/api/v1/live", headers=H, files={"cam_main": ("a.jpg", b"\xff\xd8\xff\xe0xx", "image/jpeg")})
    assert binh.get(f"/api/live/{sid}/may-chu/main.jpg").status_code == 200
    assert an.get(f"/api/live/{sid}/may-chu/main.jpg").status_code == 404
    lv = admin.get("/api/live").json()["items"]
    assert lv[0]["place_name"] == "Bình" and lv[0]["owner"] == "Bình" and lv[0]["name"] == "Phòng 101"
    # đổi tên / xoá địa điểm: camera chuyển sang địa điểm còn lại; không xoá được địa điểm cuối cùng
    assert an.post(f"/api/places/{pid}", json={"name": "Nhà ngoại"}, headers=X).json()["ok"]
    assert binh.post(f"/api/places/{pid}", json={"name": "hack"}, headers=X).status_code == 404
    assert an.post(f"/api/places/{pid}/delete", headers=X).json()["ok"]
    o = an.get("/api/overview").json()
    assert [(p["name"], p["cams_total"]) for p in o["places"]] == [("An", 1)]
    assert an.post(f"/api/places/{o['places'][0]['id']}/delete", headers=X).status_code == 400
    # tài khoản: đổi tên hiển thị; POST /api/me phải đăng nhập + có X-ARES
    assert TestClient(app).post("/api/me", json={"name": "x"}, headers=X).status_code == 401
    assert an.post("/api/me", json={"name": "Nguyễn Văn An"}).status_code == 403
    assert an.post("/api/me", json={"name": "Nguyễn Văn An"}, headers=X).json()["ok"]
    assert an.get("/api/me").json()["user"]["name"] == "Nguyễn Văn An"
    # quản trị viên tạo địa điểm cho người khác; xoá tài khoản thì xoá địa điểm của họ
    assert admin.post("/api/places", json={"name": "Chi nhánh 2", "user_id": db.user_by_name("binh")["id"]}
                      if hasattr(db, "user_by_name") else {"name": "Chi nhánh 2", "user_id": pb and binh.get("/api/me").json()["user"]["id"]},
                      headers=X).json()["ok"]
    assert len(binh.get("/api/places").json()["items"]) == 2
    bid = binh.get("/api/me").json()["user"]["id"]
    assert admin.post(f"/api/users/{bid}/delete", headers=X).json()["ok"]
    assert all(p["user_id"] != bid for p in db.places())


# ------------------------------------------------------------------ thử video: máy khác tải video lên, máy AI xử lý, trả kết quả
def test_video_upload_job_roundtrip(tmp_path, live_server):
    import zipfile as zf_

    from fastapi.testclient import TestClient

    from ares.config import AresSettings
    from ares.uplink import Uplink
    from ares.video_jobs import VideoJobWorker

    scfg, app = live_server
    db = app.state.db
    X = {"X-ARES": "1"}
    sid, key = db.create_site("Máy chủ")
    db.create_user("an.nv", "mat-khau-cua-an", "An", "viewer")
    db.create_user("binh", "mat-khau-cua-binh", "Bình", "viewer")
    an, binh = (TestClient(app, client=(ip, 5000)) for ip in ("192.168.1.20", "192.168.1.30"))
    for c, u, p in ((an, "an.nv", "mat-khau-cua-an"), (binh, "binh", "mat-khau-cua-binh")):
        assert c.post("/api/login", json={"user": u, "password": p}, headers=X).status_code == 200

    video = b"\x00\x00\x00\x18ftypmp42" + b"x" * 5000
    assert an.post("/api/videos?name=a.exe", content=video, headers=X).status_code == 415
    assert an.post("/api/videos?name=nho.mp4", content=b"abc", headers=X).status_code == 400
    jid = an.post("/api/videos?name=" + "té ngã.mp4", content=video, headers=X).json()["id"]
    lst = an.get("/api/videos").json()["items"]
    assert lst[0]["status"] == "queued" and lst[0]["queue_pos"] == 1 and lst[0]["size"] == len(video)
    assert binh.get("/api/videos").json()["items"] == [] and binh.get(f"/api/videos/{jid}").status_code == 404

    seen = {}

    def fake_analyze(cfg, pose, clf, src, out_mp4, fps, frames_zip=None, max_w=0, progress=None, should_stop=None):
        seen["bytes"] = src.read_bytes()
        progress(0.5, 3)
        with zf_.ZipFile(frames_zip, "w") as z:
            for i in range(1, 4):
                z.writestr(f"{i:04d}.jpg", JPG)
        return {"events": [{"t": 0.2, "time": "00:00.20", "kind": "FALL", "label": "Té ngã", "tracks": [1], "frame": 3}],
                "frames": 3, "fps": fps, "duration": 0.3, "people_max": 1, "people_series": [[0, 1]], "processing_s": 0.1}

    cfg = AresSettings(storage_root=tmp_path / "e", uplink_url=f"http://127.0.0.1:{scfg.port}", uplink_key=key,
                       uplink_edge_id="may-chu")
    up = Uplink(cfg, SimpleNamespace())
    w = VideoJobWorker(cfg, up, models_factory=lambda: (SimpleNamespace(device="cpu"), None, None), analyze_fn=fake_analyze)
    assert w.run_once() is True and seen["bytes"] == video
    assert w.run_once() is False  # hết job
    j = an.get(f"/api/videos/{jid}").json()
    assert j["status"] == "done" and j["n_frames"] == 3 and j["result"]["events"][0]["kind"] == "FALL"
    assert j["result"]["device"] == "cpu" and j["name"] == "té ngã.mp4"
    assert an.get(f"/api/videos/{jid}/frames/2.jpg").content == JPG
    assert binh.get(f"/api/videos/{jid}/frames/2.jpg").status_code == 404
    assert an.get("/api/videos").json()["items"][0]["n_events"] == 1
    assert not (tmp_path / "e" / "behavior" / "video_jobs" / str(jid)).exists()  # máy AI không giữ video người dùng

    # xoá giữa chừng -> máy AI dừng; máy AI tắt giữa chừng -> job được nhận lại
    jid2 = an.post("/api/videos?name=b.mp4", content=video, headers=X).json()["id"]

    def slow(cfg, pose, clf, src, out_mp4, fps, frames_zip=None, max_w=0, progress=None, should_stop=None):
        assert an.post(f"/api/videos/{jid2}/delete", headers=X).json()["ok"]
        w._progress(jid2, 0.3, "x")  # server trả 404 -> InterruptedError
        raise AssertionError("không tới đây")
    w._analyze = slow
    assert w.run_once() is True and an.get(f"/api/videos/{jid2}").status_code == 404
    jid3 = an.post("/api/videos?name=c.mp4", content=video, headers=X).json()["id"]
    assert db.claim_video_job(sid, "may-chu")["id"] == jid3
    assert db.claim_video_job(sid, "may-chu") is None
    db._x("UPDATE video_jobs SET updated=? WHERE id=?", (time.time() - 3600, jid3))  # máy AI im lặng 1 giờ
    assert db.claim_video_job(sid, "may-chu")["id"] == jid3


def test_analyze_writes_frames_zip(tmp_path):
    """analyze() thật trên video tổng hợp (pose giả không thấy ai): ra đủ khung JPEG + chuỗi số người."""
    import zipfile as zf_

    import cv2

    from ares.behavior.run_video import analyze
    from ares.config import AresSettings

    p = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 20, (320, 240))
    for i in range(40):
        img = np.full((240, 320, 3), i * 5, np.uint8)
        vw.write(img)
    vw.release()

    class NoPose:
        device = "cpu"

        def infer(self, images):
            return [(np.zeros((0, 17, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)) for _ in images]

    calls = []
    r = analyze(AresSettings(storage_root=tmp_path / "e"), NoPose(), None, p, None, 10.0, frames_zip=tmp_path / "f.zip",
                progress=lambda pr, n: calls.append(n))
    with zf_.ZipFile(tmp_path / "f.zip") as z:
        names = z.namelist()
    assert r["frames"] == 20 and len(names) == 20 and names[0] == "0001.jpg"
    assert r["events"] == [] and r["people_max"] == 0 and r["people_series"][0] == [0, 0] and abs(r["duration"] - 2.0) < 0.1


# ------------------------------------------------------------------ dashboard trên GitHub Pages: CORS + token, an toàn sau đường hầm
def test_remote_dashboard_token_cors_and_tunnel(tmp_path):
    from fastapi.testclient import TestClient

    from ares_server.__main__ import build_web
    from ares_server.app import create_app
    from ares_server.config import ServerSettings
    from ares_server.db import ServerDB

    scfg = ServerSettings(data_dir=tmp_path / "srv")
    db = ServerDB(scfg.data_dir / "s.db")
    app = create_app(scfg, db)
    c = TestClient(app, client=("8.8.8.8", 5000))
    X = {"X-ARES": "1"}
    pages = {"Origin": "https://blm2006.github.io"}
    # preflight từ GitHub Pages được phép; trang lạ thì không
    r = c.options("/api/overview", headers={**pages, "Access-Control-Request-Method": "GET",
                                            "Access-Control-Request-Headers": "authorization"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "https://blm2006.github.io"
    assert "access-control-allow-credentials" not in r.headers  # không dùng cookie khác trang
    r = c.options("/api/overview", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers
    # 401 cũng mang header CORS (để trang biết mà hiện màn đăng nhập)
    r = c.get("/api/overview", headers=pages)
    assert r.status_code == 401 and r.headers["access-control-allow-origin"] == "https://blm2006.github.io"

    db.create_user("an.nv", "mat-khau-cua-an", "An", "viewer")
    tok = c.post("/api/login", json={"user": "an.nv", "password": "mat-khau-cua-an"}, headers={**X, **pages}).json()["token"]
    nc = TestClient(app, client=("8.8.8.8", 5000))  # không cookie, chỉ token
    assert nc.get("/api/overview").status_code == 401
    assert nc.get("/api/overview", headers={"Authorization": "Bearer " + tok}).status_code == 200
    assert nc.get("/api/me", headers={"Authorization": "Bearer " + tok}).json()["user"]["username"] == "an.nv"
    assert nc.get("/api/overview?t=" + tok).status_code == 200  # <img src> dùng ?t=
    assert nc.post("/api/me?t=" + tok, json={"name": "x"}, headers=X).status_code == 401  # ?t= chỉ cho GET
    assert nc.post("/api/me", json={"name": "Ân"}, headers={"Authorization": "Bearer " + tok}).status_code == 403  # vẫn cần X-ARES
    assert nc.get("/api/overview", headers={"Authorization": "Bearer " + tok[:-2] + "00"}).status_code == 401

    # sau Tailscale Funnel / Cloudflare: kết nối từ 127.0.0.1 nhưng KHÔNG được coi là người ngồi ở máy chủ
    db2 = ServerDB(tmp_path / "s2.db")
    app2 = create_app(ServerSettings(data_dir=tmp_path / "srv2"), db2)
    loc = TestClient(app2, client=("127.0.0.1", 5000))
    assert loc.get("/api/me").json()["can_setup"] is True
    assert loc.get("/api/me", headers={"Tailscale-Funnel-Request": "?1", "X-Forwarded-For": "1.2.3.4"}).json()["can_setup"] is False
    assert loc.post("/api/setup", json={"username": "admin", "password": "mat-khau-dai-1"},
                    headers={**X, "X-Forwarded-For": "1.2.3.4"}).status_code == 403

    out = build_web(str(tmp_path / "web"), "https://may-chu.abc.ts.net/")
    html = (tmp_path / "web" / "index.html").read_text(encoding="utf-8")
    assert "window.ARES_REMOTE=true" in html and '"https://may-chu.abc.ts.net"' in html and (tmp_path / "web" / ".nojekyll").exists()
    assert "web" in out


# ------------------------------------------------------------------ mạch cảm biến: ESP32/ESP8266 (Wi-Fi) + Arduino (USB qua trình duyệt)
def test_sensor_devices_wifi_usb_and_scope(tmp_path):
    from fastapi.testclient import TestClient

    from ares_server.app import clean_sensor_data, create_app
    from ares_server.config import ServerSettings
    from ares_server.db import ServerDB

    db = ServerDB(tmp_path / "s.db")
    app = create_app(ServerSettings(data_dir=tmp_path / "srv"), db)
    X = {"X-ARES": "1"}
    db.create_user("an.nv", "mat-khau-cua-an", "An", "viewer")
    db.create_user("binh", "mat-khau-cua-binh", "Bình", "viewer")
    an, binh = (TestClient(app, client=(ip, 5000)) for ip in ("192.168.1.20", "192.168.1.30"))
    for c, u, p in ((an, "an.nv", "mat-khau-cua-an"), (binh, "binh", "mat-khau-cua-binh")):
        assert c.post("/api/login", json={"user": u, "password": p}, headers=X).status_code == 200

    assert clean_sensor_data({"temp": 28.46, "hum": 120, "motion": True, "x y": 1, "gas": float("nan"), "abc": "1"}) == \
        {"temp": 28.46, "motion": 1.0}
    wid = an.post("/api/devices", json={"name": "Phòng khách", "kind": "esp32"}, headers=X).json()["id"]
    uid = an.post("/api/devices", json={"name": "Uno bàn học", "kind": "usb"}, headers=X).json()["id"]
    assert an.post("/api/devices", json={"name": "x", "kind": "pic16"}, headers=X).status_code == 422
    # code nạp: đã ghi khoá + địa chỉ máy chủ; Wi-Fi để trống cho trình duyệt điền
    ino = an.get(f"/api/devices/{wid}/firmware.ino?server=http://192.168.1.5:8200").text
    tok = db.device(wid)["token"]
    assert f'"{tok}"' in ino and '"http://192.168.1.5:8200"' in ino and "{{WIFI_SSID}}" in ino and "#define USE_DHT" in ino
    assert "{{SENSORS}}" not in ino and "WiFiClientSecure" in ino
    assert "Serial.read()" in an.get(f"/api/devices/{uid}/firmware.ino").text  # mạch USB: code đọc lệnh qua cổng USB
    assert an.get(f"/api/devices/{wid}/firmware.ino?server=javascript:alert(1)").status_code == 400
    assert binh.get(f"/api/devices/{wid}/firmware.ino").status_code == 404

    # ESP32 gửi qua Wi-Fi bằng khoá thiết bị (không cần đăng nhập)
    dev = TestClient(app, client=("192.168.1.77", 5000))
    assert dev.post("/api/v1/sensor", json={"temp": 30}, headers={"X-Device-Token": "sai"}).status_code == 401
    r = dev.post("/api/v1/sensor", json={"temp": 30.5, "hum": 65, "motion": 1, "relay": 0, "rssi": -60},
                 headers={"X-Device-Token": tok}).json()
    assert r["ok"] and r["cmd"] == {} and r["saved"] == 5
    # bật relay trên dashboard -> mạch nhận trong câu trả lời lần gửi sau
    assert an.post(f"/api/devices/{wid}/cmd", json={"relay": 1, "led": 250, "evil": 1}, headers=X).json()["desired"] == \
        {"relay": 1, "led": 100}
    db._x("UPDATE devices SET last_seen=last_seen-5 WHERE id=?", (wid,))
    r = dev.post(f"/api/v1/sensor?token={tok}", json={"temp": 31, "relay": 1}).json()
    assert r["cmd"] == {"relay": 1, "led": 100}
    assert binh.post(f"/api/devices/{wid}/cmd", json={"relay": 0}, headers=X).status_code == 404

    # Arduino cắm USB: trình duyệt gửi bằng phiên đăng nhập
    assert an.post(f"/api/devices/{uid}/data", json={"light": 40, "gas": 12}, headers=X).json()["ok"]
    items = {d["id"]: d for d in an.get("/api/devices").json()["items"]}
    assert items[wid]["online"] and items[wid]["data"]["temp"] == 31 and items[wid]["via"] == "wifi"
    assert items[uid]["via"] == "usb" and items[uid]["data"] == {"light": 40.0, "gas": 12.0}
    assert binh.get("/api/devices").json()["items"] == []
    assert len(an.get(f"/api/devices/{wid}/series").json()["items"]) == 1  # lịch sử tối đa 1 bản / 30 s

    # số đo hiện trên thẻ địa điểm; chuyển mạch sang địa điểm khác
    pl = an.get("/api/overview").json()["places"][0]
    assert {s["name"] for s in pl["sensors"]} == {"Phòng khách", "Uno bàn học"}
    pid = an.post("/api/places", json={"name": "Cửa hàng"}, headers=X).json()["id"]
    assert an.post(f"/api/devices/{wid}", json={"place_id": pid}, headers=X).json()["ok"]
    pb = binh.get("/api/places").json()["items"][0]["id"]
    assert an.post(f"/api/devices/{wid}", json={"place_id": pb}, headers=X).status_code == 404
    ov = {p["name"]: p for p in an.get("/api/overview").json()["places"]}
    assert [s["name"] for s in ov["Cửa hàng"]["sensors"]] == ["Phòng khách"]

    # xoá mạch -> khoá cũ hết hiệu lực
    assert an.post(f"/api/devices/{wid}/delete", headers=X).json()["ok"]
    assert dev.post("/api/v1/sensor", json={"temp": 1}, headers={"X-Device-Token": tok}).status_code == 401
