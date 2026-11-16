"""Studio: thả video, lưu hình vẽ theo từng video, kiểm tra tình huống, chạy nền."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from traffic_safety_ai.core.config import AppPaths, load_site_config
from traffic_safety_ai.dashboard import studio as st
from traffic_safety_ai.dashboard.app import create_app

H = {"X-TSA": "1"}


def _video_bytes(tmp_path, n=12, size=(320, 240)) -> bytes:
    f = tmp_path / "goc.avi"
    w = cv2.VideoWriter(str(f), cv2.VideoWriter_fourcc(*"MJPG"), 10, size)
    for i in range(n):
        img = np.full((size[1], size[0], 3), 40 + i * 5, np.uint8)
        w.write(img)
    w.release()
    return f.read_bytes()


@pytest.fixture()
def client(tmp_path):
    app = create_app(AppPaths(tmp_path / "data"))
    with TestClient(app) as c:
        c.tmp = tmp_path
        yield c


def _upload(c, name="cổng trường 1.avi"):
    r = c.put("/api/studio/upload", params={"name": name}, content=_video_bytes(c.tmp), headers=H)
    assert r.status_code == 200, r.text
    return r.json()["name"]


def test_trang_studio_va_tai_len(client):
    assert "Studio" in client.get("/studio").text
    name = _upload(client)
    assert name == "cong truong 1.avi"            # bỏ dấu để OpenCV/Windows đọc chắc chắn
    vids = client.get("/api/studio/videos").json()["videos"]
    assert [v["name"] for v in vids if v["kind"] == "đã thả vào"] == [name]
    info = client.get("/api/studio/info", params={"video": name}).json()
    assert (info["width"], info["height"], info["frames"]) == (320, 240, 12)
    for pos in (0, 0.5, 1):
        r = client.get("/api/studio/frame", params={"video": name, "pos": pos})
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_chan_csrf_va_ten_tep_xau(client):
    body = _video_bytes(client.tmp)
    assert client.put("/api/studio/upload", params={"name": "a.avi"}, content=body).status_code == 403
    assert client.put("/api/studio/upload", params={"name": "virus.exe"}, content=body, headers=H).status_code == 400
    r = client.put("/api/studio/upload", params={"name": "..\\..\\thoat ra.avi"}, content=body, headers=H)
    assert r.status_code == 200 and r.json()["name"] == "thoat ra.avi"
    assert (client.tmp / "data" / "studio" / "videos" / "thoat ra.avi").is_file()
    # tệp không phải video thật → bị từ chối và không để lại rác
    r = client.put("/api/studio/upload", params={"name": "gia.mp4"}, content=b"khong phai video", headers=H)
    assert r.status_code == 415
    assert not (client.tmp / "data" / "studio" / "videos" / "gia.mp4").exists()
    assert client.get("/api/studio/frame", params={"video": "khong_co.mp4"}).status_code == 404


def test_luu_hinh_ve_rieng_cho_tung_video(client):
    name = _upload(client)
    d = client.get("/api/studio/scene", params={"video": name}).json()
    sc = d["scene"]
    assert d["saved"] is False and sc["camera"]["frame_size"] == [320, 240]
    assert not sc["zones"]["sidewalks"] and not sc["lines"]["stop_lines"]      # video mới: không kế thừa hình vẽ mẫu
    rules0 = {r["code"]: r["enabled"] for r in d["rules"]}
    assert rules0["RB-05"] is False and rules0["RB-04"] is False

    sc["zones"]["sidewalks"].append({"id": "via-he-1", "polygon": [[0, 150], [320, 150], [320, 240], [0, 240]]})
    sc["traffic_lights"].append({"id": "den-1", "roi": [10, 10, 30, 60]})
    sc["lines"]["stop_lines"].append({"id": "vach-dung-1", "points": [[40, 120], [280, 120]],
                                      "direction": [[160, 140], [160, 100]], "light_id": "den-1"})
    body = {"scene": {**{k: sc[k] for k in ("zones", "lines", "traffic_lights", "calibration")},
                      "camera": {"name": "Cổng trường A"}, "speed": {"limit_kmh": 30}}}
    chk = client.post("/api/studio/check", params={"video": name}, json=body, headers=H).json()
    rules1 = {r["code"]: r["enabled"] for r in chk["rules"]}
    assert rules1["RB-04"] is True and rules1["RB-05"] is True
    assert client.get("/api/studio/scene", params={"video": name}).json()["saved"] is False   # check không ghi tệp

    r = client.put("/api/studio/scene", params={"video": name}, json=body, headers=H)
    assert r.status_code == 200, r.text
    cfg = load_site_config(r.json()["file"])
    assert cfg.camera.name == "Cổng trường A" and cfg.speed.limit_kmh == 30
    assert cfg.zones.sidewalks[0].id == "via-he-1" and cfg.lines.stop_lines[0].light_id == "den-1"
    d2 = client.get("/api/studio/scene", params={"video": name}).json()
    assert d2["saved"] is True and d2["scene"]["traffic_lights"][0]["roi"] == [10, 10, 30, 60]
    assert next(v for v in client.get("/api/studio/videos").json()["videos"] if v["name"] == name)["has_scene"]


def test_hinh_ve_sai_bao_loi_tieng_viet(client):
    name = _upload(client)
    sc = client.get("/api/studio/scene", params={"video": name}).json()["scene"]
    sc["zones"]["sidewalks"].append({"id": "x", "polygon": [[0, 0], [1, 1]]})
    r = client.put("/api/studio/scene", params={"video": name}, json={"scene": {"zones": sc["zones"]}}, headers=H)
    assert r.status_code == 422 and "ít nhất 3 điểm" in r.json()["detail"]
    assert not st.StudioPaths(AppPaths(client.tmp / "data")).scene_file(name).exists()


def test_chay_nen_mot_viec_mot_luc(client, monkeypatch):
    name = _upload(client)
    r = client.post("/api/studio/run", json={"video": name}, headers=H)
    assert r.status_code == 400 and "Lưu" in r.json()["detail"]
    client.put("/api/studio/scene", params={"video": name}, json={"scene": {}}, headers=H)

    gate = {"go": False}

    def fake(self, job):
        job.state, job.total = "dang_chay", 12
        while not gate["go"] and not job.stop_flag.is_set():
            time.sleep(0.01)
        job.frame = 12
        job.state, job.message = ("da_dung", "Đã dừng") if job.stop_flag.is_set() else ("xong", "Xong")

    monkeypatch.setattr(st.JobRunner, "_run", fake)
    assert client.post("/api/studio/run", json={"video": name, "options": {"stride": 2}}, headers=H).status_code == 200
    assert client.post("/api/studio/run", json={"video": name}, headers=H).status_code == 409
    assert client.delete(f"/api/studio/videos/{name}", headers=H).status_code == 409
    assert client.get("/api/studio/job").json()["job"]["active"] is True
    client.post("/api/studio/stop", headers=H)
    for _ in range(200):
        j = client.get("/api/studio/job").json()["job"]
        if not j["active"]:
            break
        time.sleep(0.01)
    assert j["state"] == "da_dung"
    assert client.delete(f"/api/studio/videos/{name}", headers=H).status_code == 200
    assert not any(v["name"] == name for v in client.get("/api/studio/videos").json()["videos"])


def test_loi_khi_chay_duoc_bao_len_giao_dien(client, monkeypatch):
    name = _upload(client)
    client.put("/api/studio/scene", params={"video": name}, json={"scene": {}}, headers=H)

    def boom(self, job):
        raise RuntimeError("thiếu model")

    monkeypatch.setattr(st.JobRunner, "_run", boom)
    client.post("/api/studio/run", json={"video": name}, headers=H)
    for _ in range(200):
        j = client.get("/api/studio/job").json()["job"]
        if not j["active"]:
            break
        time.sleep(0.01)
    assert j["state"] == "loi" and "thiếu model" in j["message"]


def test_them_theo_duong_dan(client):
    assert st.safe_name("C:\\x\\Đường Số 5 (sáng).MP4") == "Duong So 5 (sang).MP4"
    f = client.tmp / "ngoai" / "xa lo.avi"
    f.parent.mkdir()
    f.write_bytes(_video_bytes(client.tmp))
    r = client.post("/api/studio/link", json={"path": str(f)}, headers=H)
    assert r.status_code == 200 and r.json()["name"] == "xa lo.avi"
    assert client.get("/api/studio/frame", params={"video": "xa lo.avi"}).status_code == 200
    assert client.post("/api/studio/link", json={"path": str(f.parent / "khong.avi")}, headers=H).status_code == 400
    client.delete("/api/studio/videos/xa lo.avi", headers=H)
    assert f.is_file()                                      # gỡ liên kết không xoá tệp gốc của người dùng


# ----------------------------------------------------------------- trang "Xử lý vi phạm" + nhãn gọn trên video
def _rec(i, rule, cls, status, day, plate):
    return {"id": f"VP-{i:03d}", "rule_code": rule, "rule_name": f"Lỗi {rule}", "severity": "vi_pham",
            "camera_id": "CAM-01", "camera_name": "Cổng A", "location": "Đường X", "occurred_at": f"{day}T07:10:0{i}",
            "vehicle_cls": cls, "plate_text": plate, "plate_display": plate, "status": status, "run_id": f"run{i % 2}",
            "files": {}, "measurements": {}, "notes": []}


def test_trang_vi_pham_loc_va_dem_theo_trang_thai(tmp_path):
    from traffic_safety_ai.evidence.store import Store

    paths = AppPaths(tmp_path / "data").ensure()
    s = Store(paths.db)
    s.insert_violation(_rec(1, "RB-05", "motorcycle", "cho_xac_minh", "2026-11-01", "59X112345"))
    s.insert_violation(_rec(2, "RB-05", "car", "cho_xac_minh", "2026-11-03", "51F67890"))
    s.insert_violation(_rec(3, "RB-04", "motorcycle", "da_xac_minh", "2026-11-05", "60B598765"))
    s.insert_violation(_rec(4, "RB-04", "motorcycle", "loai_bo", "2026-11-05", ""))
    s.close()
    with TestClient(create_app(paths)) as c:
        assert "Xử lý vi phạm" in c.get("/vi-pham").text
        f = c.get("/api/violations-facets").json()
        assert f["by_status"] == {"cho_xac_minh": 2, "da_xac_minh": 1, "loai_bo": 1}
        assert {v["id"] for v in f["vehicles"]} == {"motorcycle", "car"} and len(f["rules"]) == 2
        assert (f["first_day"], f["last_day"]) == ("2026-11-01", "2026-11-05")
        f2 = c.get("/api/violations-facets", params={"vehicle": "motorcycle", "day_from": "2026-11-02"}).json()
        assert f2["by_status"] == {"da_xac_minh": 1, "loai_bo": 1}

        def ids(**kw):
            return sorted(v["id"] for v in c.get("/api/violations", params=kw).json()["items"])

        assert ids(vehicle="car") == ["VP-002"]
        assert ids(day_from="2026-11-02", day_to="2026-11-04") == ["VP-002"]
        assert ids(q="598") == ["VP-003"] and ids(q="VP-004") == ["VP-004"]
        assert ids(run="run1", status="cho_xac_minh") == ["VP-001"]
        assert ids(rule="RB-04", status="loai_bo") == ["VP-004"]


def test_video_chu_thich_gon_chi_ghi_chu_len_xe_vi_pham():
    from datetime import datetime

    from traffic_safety_ai.core.config import SiteConfig
    from traffic_safety_ai.core.types import FrameContext
    from traffic_safety_ai.utils.visualizer import Visualizer

    class Kin:
        speed_kmh = 31.0

    class Tr:
        def __init__(self, tid, box):
            self.id, self.cls, self.xyxy, self.kin = tid, "car", np.array(box, float), Kin()

    frame = np.zeros((360, 640, 3), np.uint8)
    tracks = [Tr(1, (100, 150, 200, 220)), Tr(2, (400, 150, 500, 220))]
    ctx = FrameContext(0, 0.0, datetime(2026, 11, 16, 7, 0), frame, "cctv", "CAM", tracks, {}, SiteConfig())

    def ink(img, box):          # số điểm ảnh có chữ/nền nhãn ngay phía trên hộp
        x1, y1, x2, _ = box
        return int((img[y1 - 22:y1 - 2, x1:x2 + 80].sum(axis=2) > 0).sum())

    gon = Visualizer().draw_frame(frame, ctx, [], suspects={2}, plates={2: "51F-678.90"}, draw_scene=False)
    assert ink(gon, (100, 150, 200, 220)) == 0            # xe bình thường: không chữ
    assert ink(gon, (400, 150, 500, 220)) > 100           # xe vi phạm: có nhãn (biển số)
    full = Visualizer(labels="day_du").draw_frame(frame, ctx, [], suspects={2}, draw_scene=False)
    assert ink(full, (100, 150, 200, 220)) > 100
