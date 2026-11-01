import json
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ares.active_learner import _Job
from ares.clock import ManualClock
from ares.config import AresSettings
from ares.dashboard.app import create_app
from ares.detector import Det
from ares.model_registry import BASE
from ares.runtime import AresRuntime

H = {"X-ARES": "1"}


def build(tmp_path, **over):
    cfg = AresSettings(_env_file=None, storage_root=tmp_path / "a", simulation_mode=True, overlay_interval_s=0.0,
                       al_min_gap_s=0.0, al_hourly_cap=100, **{"pca_enabled": False, **over})
    cfg.ensure_dirs()
    clock = ManualClock(datetime(2026, 11, 2, 7, 5))
    return AresRuntime(cfg, clock=clock, with_dashboard=False), cfg, clock


@pytest.fixture
def rt(tmp_path):
    r, *_ = build(tmp_path)
    for _ in range(20):  # chạy tay vài chục giây mô phỏng
        r.clock.advance(5)
        r.world.advance(r.clock.time())
        r.tracker.tick()
    r.engine.lighting_tick()
    r.engine.hvac_tick()
    yield r
    r.al.drain(5); r.al.stop(); r.db.close()


@pytest.fixture
def client(rt):
    return TestClient(create_app(rt))


def make_pending(rt, n=1):
    ids = []
    for i in range(n):
        img = np.random.default_rng(i).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        iid = rt.al.save(_Job(img, [Det(16, 12, 80, 108, 0.45)], rt.clock.now() + timedelta(seconds=i)))
        ids.append(iid)
    return ids


def test_health_status_and_frame(client, rt):
    assert client.get("/healthz").json() == {"ok": True}
    s = client.get("/api/status").json()
    assert s["simulation"] is True and s["room_id"] == "P101" and len(s["zones"]) == 4 and s["tz"] == "Asia/Ho_Chi_Minh"
    assert s["vision"]["n_total"] >= 0 and len(s["lighting"]["pwm"]) == 4 and "sim" in s
    assert s["schedule"]["current"]["name"] == "Tiết 1-3"
    fr = client.get("/api/frame.jpg")
    assert fr.status_code == 200 and fr.headers["content-type"] == "image/jpeg" and fr.content[:2] == b"\xff\xd8"
    assert fr.headers["cache-control"] == "no-store"


def test_index_page_is_vietnamese_and_self_contained(client):
    r = client.get("/")
    assert r.status_code == 200 and "Giám sát" in r.text and "Duyệt ảnh" in r.text
    assert "http://" not in r.text.replace("http://localhost", "") and "https://" not in r.text  # không phụ thuộc CDN


def test_history_events_habits(client, rt):
    rt.log_telemetry(); rt.db.flush()
    h = client.get("/api/history?hours=1").json()
    assert h["rows"] and h["bucket_s"] >= 30
    assert client.get("/api/history?hours=99999&bucket=1").status_code == 200  # tham số bị kẹp, không nổ
    assert "events" in client.get("/api/events?limit=5").json()
    assert client.get("/api/habits").json()["p_threshold"] == 0.75


def test_review_flow_via_api(client, rt):
    a, b, c = make_pending(rt, 3)
    r = client.get("/api/al/items?status=PENDING").json()
    assert {x["image_id"] for x in r["items"]} == {a, b, c} and r["counts"]["PENDING"] == 3
    img = client.get(f"/api/al/items/{a}/image")
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    assert client.post(f"/api/al/items/{a}/approve", headers=H).json()["status"] == "APPROVED"
    e = client.post(f"/api/al/items/{b}/edit", headers=H, json={"boxes": [{"bbox": [5, 5, 60, 100], "class": 0}]})
    assert e.json() == {"image_id": b, "status": "APPROVED", "n_boxes": 1}
    assert client.post(f"/api/al/items/{c}/reject", headers=H, json={"reason": "Ảnh mờ"}).json()["status"] == "REJECTED"
    counts = client.get("/api/al/stats").json()["counts"]
    assert (counts["PENDING"], counts["APPROVED"], counts["REJECTED"]) == (0, 2, 1)


def test_review_error_codes(client, rt):
    (a,) = make_pending(rt)
    assert client.post(f"/api/al/items/{a}/approve", headers=H).status_code == 200
    r = client.post(f"/api/al/items/{a}/approve", headers=H)
    assert r.status_code == 409 and "PENDING" in r.json()["detail"]
    assert client.post("/api/al/items/khong_co/approve", headers=H).status_code == 404
    assert client.post("/api/al/items/..%2Fetc%2Fpasswd/approve", headers=H).status_code in (400, 404)
    assert client.get("/api/al/items/a b/image").status_code in (400, 404)
    assert client.get("/api/al/items?status=BAY").status_code == 400
    assert client.post(f"/api/al/items/{a}/edit", headers=H, json={"boxes": [{"bbox": [1, 2, 3]}]}).status_code == 422


def test_posts_require_csrf_header(client, rt):
    (a,) = make_pending(rt)
    r = client.post(f"/api/al/items/{a}/approve")  # như một trang web lạ gửi chéo origin: không có header tuỳ biến
    assert r.status_code == 403
    assert client.get("/api/status").status_code == 200  # GET không bị ảnh hưởng
    assert rt.al.get_item(a)["status"] == "PENDING"


def test_model_management(client, rt, tmp_path):
    m = client.get("/api/models").json()
    assert m["production"] == BASE and m["versions"][0]["version"] == BASE
    w = tmp_path / "w.pt"; w.write_bytes(b"w")
    rt.registry.add(w, BASE, "midnight", 3, 9, 20, {}, {"passed": None, "reason": "cần duyệt"}, status="candidate")
    assert client.post("/api/models/v0001/promote", headers=H).json() == {"production": "v0001"}
    assert client.post("/api/models/v0001/promote", headers=H).status_code == 409  # đã là production
    assert client.post("/api/models/v7777/promote", headers=H).status_code == 404
    assert client.post("/api/models/rollback", headers=H).json() == {"production": BASE}
    assert client.post("/api/models/rollback", headers=H).status_code == 409
    assert "MODEL_PROMOTED" in [e.kind for e in rt.state.events] and "MODEL_ROLLBACK" in [e.kind for e in rt.state.events]


def test_retrain_endpoint_respects_config(client, rt):
    r = client.post("/api/retrain", headers=H)
    assert r.status_code == 409 and "tắt" in r.json()["detail"]  # chế độ mô phỏng mặc định không train


def test_token_protection(tmp_path):
    r, cfg, _ = build(tmp_path, dashboard_token="bi-mat")
    c = TestClient(create_app(r))
    assert c.get("/api/status").status_code == 401
    assert c.get("/healthz").status_code == 200
    assert c.get("/api/status", headers={"X-ARES-Token": "sai"}).status_code == 401
    assert c.get("/api/status", headers={"X-ARES-Token": "bi-mat"}).status_code == 200
    first = c.get("/?token=bi-mat")
    assert first.status_code == 200 and "ares_token" in first.headers.get("set-cookie", "") and "HttpOnly" in first.headers["set-cookie"]
    assert c.get("/api/status").status_code == 200  # cookie được nhớ
    r.al.stop(); r.db.close()


def test_refuses_to_expose_dashboard_without_token(tmp_path):
    r, cfg, _ = build(tmp_path, dashboard_host="0.0.0.0")
    with pytest.raises(RuntimeError, match="ARES_DASHBOARD_TOKEN"):
        r._check_dashboard_exposure()
    cfg.dashboard_token = "x"
    r._check_dashboard_exposure()
    r.al.stop(); r.db.close()


def test_runtime_threads_start_and_stop_cleanly(tmp_path):
    from ares.clock import SimClock

    cfg = AresSettings(_env_file=None, storage_root=tmp_path / "t", simulation_mode=True, sim_speed=600.0,
                       sim_start="2026-11-02T06:50:00", telemetry_interval_s=60.0)
    cfg.ensure_dirs()
    rt = AresRuntime(cfg, with_dashboard=False)
    assert isinstance(rt.clock, SimClock)
    rt.start()
    import time
    time.sleep(2.5)  # ~25 phút mô phỏng: có kế hoạch pre-cool và lớp bắt đầu
    snap = rt.status()
    names = set(snap["health"])
    rt.stop()
    assert {"vision", "lighting", "hvac", "telemetry", "sim_world"} <= names
    assert all(h["errors"] == 0 for h in snap["health"].values()), snap["health"]
    assert snap["sim"]["true_people"] >= 0 and any(e["kind"] == "SYSTEM_START" for e in snap["events"])
    assert not any(t.is_alive() for t in rt._threads)


def test_dashboard_boots_over_real_http(tmp_path):
    import socket
    import time
    import urllib.request

    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    cfg = AresSettings(_env_file=None, storage_root=tmp_path / "h", simulation_mode=True, sim_speed=60.0,
                       sim_start="2026-11-02T07:00:00", dashboard_port=port)
    cfg.ensure_dirs()
    rt = AresRuntime(cfg)
    rt.start()
    try:
        body = None
        for _ in range(50):
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2).read()
                break
            except Exception:
                time.sleep(0.2)
        assert body and json.loads(body)["room_id"] == "P101"
    finally:
        rt.stop()
