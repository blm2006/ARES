"""Dashboard (FastAPI), công cụ khai báo vùng, dòng lệnh, chạy mô phỏng đầu-cuối."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from traffic_safety_ai.core.config import AppPaths, SiteConfig, load_site_config, save_site_config
from traffic_safety_ai.evidence.store import Store
from traffic_safety_ai.tools.roi_calibrator import CalibState, preview_scene


# ----------------------------------------------------------------- dashboard
@pytest.fixture
def app_env(tmp_path):
    from fastapi.testclient import TestClient

    from traffic_safety_ai.dashboard.app import create_app

    paths = AppPaths(tmp_path / "data").ensure()
    st = Store(paths.db)
    folder = paths.evidence / "2026-11-04" / "VP20261104-CAM01-00001"
    folder.mkdir(parents=True)
    (folder / "composite.jpg").write_bytes(cv2.imencode(".jpg", np.zeros((10, 10, 3), np.uint8))[1].tobytes())
    (folder / "record.html").write_text("<p>phiếu</p>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("bí mật", encoding="utf-8")
    st.insert_violation({"id": "VP20261104-CAM01-00001", "rule_code": "RB-05", "rule_name": "Vượt đèn đỏ",
                         "severity": "vi_pham", "camera_id": "CAM-01", "occurred_at": "2026-11-04T07:10:00",
                         "vehicle_cls": "motorcycle", "plate_text": "60B598765", "plate_display": "60-B5 987.65",
                         "status": "cho_xac_minh", "evidence_dir": str(folder),
                         "files": {"composite.jpg": {"sha256": "x", "size": 1}}, "measurements": {}, "notes": []})
    st.close()
    return paths, create_app, TestClient


def test_dashboard_list_detail_review_and_csrf(app_env):
    paths, create_app, TestClient = app_env
    c = TestClient(create_app(paths))
    assert c.get("/").status_code == 200 and "Giám sát" in c.get("/").text
    r = c.get("/api/violations").json()
    assert r["total"] == 1 and r["items"][0]["plate_display"] == "60-B5 987.65"
    assert "evidence_dir" not in r["items"][0]                        # không lộ đường dẫn máy chủ
    vid = r["items"][0]["id"]
    assert c.get(f"/api/violations/{vid}").json()["has"]["composite.jpg"]
    assert c.get(f"/api/violations/{vid}/file/composite.jpg").status_code == 200
    assert c.get(f"/api/violations/{vid}/file/..%2F..%2F..%2Fsecret.txt").status_code in (400, 404)
    assert c.get(f"/api/violations/{vid}/file/.hidden").status_code == 400
    # thiếu header chống CSRF → 403
    assert c.post(f"/api/violations/{vid}/review", json={"action": "approve", "reviewer": "Trí"}).status_code == 403
    h = {"X-TSA": "1"}
    assert c.post(f"/api/violations/{vid}/review", json={"action": "approve"}, headers=h).status_code == 400
    ok = c.post(f"/api/violations/{vid}/review", json={"action": "approve", "reviewer": "Trí"}, headers=h).json()
    assert ok["ok"] and ok["item"]["status"] == "da_xac_minh"
    ed = c.post(f"/api/violations/{vid}/review", json={"action": "edit_plate", "reviewer": "Trí",
                                                        "plate": "60B5-98766"}, headers=h).json()
    assert ed["item"]["plate_display"] == "60-B5 987.66" and ed["item"]["status"] == "da_xac_minh"
    assert c.get("/api/audit").json()["items"][0]["action"] == "review:edit_plate"
    lk = c.get("/api/lookup", params={"plate": "60B598766"}).json()
    assert lk["total"] == 1 and lk["valid"]
    assert c.get("/api/lookup", params={"plate": "60"}).status_code == 400
    assert c.get("/api/stats").json()["total"] == 1
    assert "counts" in c.get("/api/traffic").json()
    assert c.get("/api/live.jpg").status_code == 404
    rep = c.post("/api/report", json={}, headers=h).json()
    assert c.get(rep["url"]).status_code == 200
    assert c.get("/evidence/2026-11-04/VP20261104-CAM01-00001/record.html").status_code == 200


def test_dashboard_token(app_env):
    paths, create_app, TestClient = app_env
    app = create_app(paths, token="abc123")
    c = TestClient(app)
    assert c.get("/api/violations").status_code == 401
    assert c.get("/").status_code == 401
    assert c.get("/api/violations", headers={"Authorization": "Bearer abc123"}).status_code == 200
    r = c.get("/?token=abc123", follow_redirects=False)
    assert r.status_code == 303 and "tsa_token" in r.headers.get("set-cookie", "")
    c.cookies.set("tsa_token", "abc123")
    assert c.get("/api/violations").status_code == 200
    # mở ra mạng mà không khai báo token → tự sinh
    assert create_app(paths, require_token=True).state.token


# ----------------------------------------------------------------- công cụ khai báo
def test_calibrator_state_to_config(tmp_path):
    answers = iter(["", "", "", "", "VD-1", "Vạch dừng chính", "", "", "", "", "4", "12"])
    st = CalibState(ask=lambda q, d: next(answers, "") or d)
    st.set_mode("1")
    for p in [(0, 0), (100, 0), (100, 100)]:
        st.click(*p)
    st.next_step()                                     # đa giác vỉa hè xong (tên/mã mặc định)
    st.set_mode("3")
    for p in [(200, 0), (300, 0), (300, 300), (200, 300)]:
        st.click(*p)
    st.next_step()
    st.click(250, 250)
    st.click(250, 50)                                  # mũi tên hướng đi (2 điểm cố định → tự sang bước sau)
    st.set_mode("s")
    st.click(200, 150)
    st.click(300, 150)
    st.click(250, 200)
    st.click(250, 100)
    st.next_step()                                     # bỏ qua vạch xác nhận
    st.set_mode("l")
    st.click(500, 10)
    st.click(520, 70)
    st.set_mode("h")
    for p in [(0, 400), (400, 400), (350, 200), (50, 200)]:
        st.click(*p)
    assert [s.mode for s in st.shapes] == ["1", "3", "s", "l", "h"]
    cfg = st.apply_to(SiteConfig())
    assert cfg.zones.sidewalks[0].polygon[1] == (100.0, 0.0)
    assert cfg.zones.lanes[0].direction == [(250.0, 250.0), (250.0, 50.0)]
    sl = cfg.lines.stop_lines[0]
    assert sl.id == "VD-1" and sl.confirm_points is None and sl.light_id == cfg.traffic_lights[0].id
    assert cfg.traffic_lights[0].roi == (500.0, 10.0, 520.0, 70.0)
    assert cfg.calibration.enabled and cfg.calibration.world_points[2] == (4.0, 12.0)
    out = tmp_path / "a.yaml"
    save_site_config(cfg, out)
    again = load_site_config(out)
    assert again.lines.stop_lines[0].name == "Vạch dừng chính"
    st.delete_last()
    assert len(st.shapes) == 4


def test_preview_scene(tmp_path):
    vid = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for i in range(20):
        w.write(np.full((240, 320, 3), i * 10, np.uint8))
    w.release()
    cfg = load_site_config(Path(__file__).resolve().parents[1] / "configs" / "cctv_config.yaml")
    out = preview_scene(str(vid), cfg, str(tmp_path / "p.jpg"), at_s=0.5)
    img = cv2.imread(out)
    assert img is not None and img.shape[:2] == (240, 320)


# ----------------------------------------------------------------- dòng lệnh + mô phỏng đầu-cuối
def test_cli_check_and_lookup(tmp_path, capsys):
    from traffic_safety_ai.cli import main

    assert main(["check", "--no-models"]) == 0
    out = capsys.readouterr().out
    assert "RB-05" in out and "BẬT" in out
    assert main(["--data-dir", str(tmp_path / "d"), "lookup", "51F67890"]) == 0
    assert "0 hồ sơ" in capsys.readouterr().out


def test_sim_pipeline_end_to_end(tmp_path):
    from traffic_safety_ai.sim.demo import run_demo

    s = run_demo("cctv", data_dir=tmp_path / "d", write_video=False, duration=14.0, async_io=False,
                 start_time=datetime(2026, 11, 4, 7, 0, 0))
    by = s["by_rule"]
    assert by.get("RB-10") == 1 and by.get("RB-06") == 2 and by.get("RB-03") == 1 and by.get("RB-04") == 1
    assert sum(by.values()) == 5                                  # không báo giả trong 14 giây đầu
    st = Store(Path(s["db"]))
    rows, _ = st.list_violations(limit=50)
    plates = {r["rule_code"] + ":" + (r["plate_display"] or "") for r in rows}
    assert "RB-10:51F-678.90" in plates and "RB-03:51-K3 456.78" in plates   # hợp nhất nhiều lần đọc lỗi
    one = next(r for r in rows if r["rule_code"] == "RB-10")
    folder = Path(one["evidence_dir"])
    for f in ("composite.jpg", "record.html", "record.json", "snapshot_raw.jpg"):
        assert (folder / f).exists(), f
    assert one["files"]["composite.jpg"]["sha256"]
    st.close()
    assert Path(s["report"]).exists()


def test_sim_pov_end_to_end(tmp_path):
    from traffic_safety_ai.sim.demo import run_demo

    s = run_demo("pov", data_dir=tmp_path / "d", write_video=False, async_io=False, make_report=False)
    assert s["by_rule"] == {"RB-03": 1, "RB-08": 1, "HZ-02": 1, "HZ-03": 1, "RB-07": 1, "HZ-01": 1}
    hz01 = next(e for e in s["events"] if e["rule"] == "HZ-01")
    assert hz01["t"] > 25.0                                       # xe bật xi-nhan không bị báo
