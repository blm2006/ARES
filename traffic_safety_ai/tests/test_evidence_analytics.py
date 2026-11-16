"""CSDL, phiếu bằng chứng, thống kê lưu lượng/tốc độ, báo cáo."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from traffic_safety_ai.analytics.counting import LineCounter, percentile, summarize_counts, summarize_speeds
from traffic_safety_ai.analytics.reports import build_report
from traffic_safety_ai.core.config import AppPaths, CountingLineCfg, SiteConfig
from traffic_safety_ai.evidence.privacy import blur_bystanders, pixelate
from traffic_safety_ai.evidence.report import compose_sheet, fmt_value, meas_lines, record_fields, record_html
from traffic_safety_ai.evidence.store import Store
from traffic_safety_ai.tests.helpers import World, box_at


def _rec(store: Store, i: int, plate: str = "51F67890", rule: str = "RB-10", when: str = "2026-11-04T07:05:00",
         sev: str = "vi_pham", cls: str = "car") -> str:
    vid = store.next_violation_id(datetime.fromisoformat(when), "CAM-01")
    store.insert_violation({"id": vid, "rule_code": rule, "rule_name": "Chạy quá tốc độ", "severity": sev,
                            "camera_id": "CAM-01", "occurred_at": when, "vehicle_cls": cls,
                            "plate_text": plate, "plate_display": "51F-678.90", "status": "cho_xac_minh",
                            "measurements": {"toc_do_kmh": 61.7 + i}, "notes": ["n"], "files": {}})
    return vid


def test_store_ids_review_lookup_stats(tmp_path):
    st = Store(tmp_path / "t.db")
    a = _rec(st, 0)
    b = _rec(st, 1, plate="29AB56789", rule="RB-06", sev="nghi_van", cls="motorcycle")
    assert a.endswith("-00001") and b.endswith("-00002") and a.startswith("VP20261104-CAM01-")
    rows, total = st.list_violations(plate="51F678")
    assert total == 1 and rows[0]["measurements"]["toc_do_kmh"] == 61.7
    assert st.review(a, "approve", "Trí", "đúng") and st.get_violation(a)["status"] == "da_xac_minh"
    assert st.review(b, "reset", "Trí", plate="29-AB 567.89")
    r = st.get_violation(b)
    assert r["plate_display"] == "29-AB 567.89" and r["plate_valid"] == 1
    assert not st.review("khong-co", "approve", "x")
    assert len(st.lookup_plate("51F-678.90")) == 1 and st.lookup_plate("51") == []
    st.review(a, "reject", "Trí")
    assert st.lookup_plate("51F67890") == []                 # hồ sơ bị loại không hiện khi tra cứu
    s = st.stats()
    assert s["total"] == 1 and {x["rule_code"] for x in s["by_rule"]} == {"RB-06"}
    st.audit("Trí", "review:approve", a, "{}", "127.0.0.1")
    assert st.audit_log()[0]["actor"] == "Trí"
    old = (datetime.now() - timedelta(days=200)).isoformat(timespec="seconds")
    c = _rec(st, 2, when=old)
    assert [x["id"] for x in st.expired(90)] == [c]
    st.close()


def test_counts_speeds_hazards(tmp_path):
    st = Store(tmp_path / "t.db")
    rows = [("L1", "Mặt cắt", f"2026-11-04T07:{m:02d}:00", "motorcycle", "Đi ra", 10 + m) for m in (0, 15, 30, 45)]
    st.add_counts("r1", "CAM-01", rows)
    st.add_counts("r1", "CAM-01", [("L1", "Mặt cắt", "2026-11-04T07:00:00", "motorcycle", "Đi ra", 5)])
    got = st.counts(camera_id="CAM-01")
    assert sum(r["n"] for r in got) == 10 + 25 + 40 + 55 + 5     # cộng dồn khi trùng khoá
    summ = summarize_counts(got)
    line = summ["L1"]
    assert line["total"] == sum(r["n"] for r in got)
    assert line["peak"] is not None and 0 < line["peak"]["phf"] <= 1.0
    for v in (20.0, 25.0, 30.0, 45.0):
        st.add_speed("r1", "CAM-01", 1, "car", "2026-11-04T07:00:00", v, "homography")
    ss = summarize_speeds(st.speeds(), limit_kmh=40)
    assert ss["car"]["n"] == 4 and ss["car"]["pct_over"] == 25.0 and ss["Tất cả"]["max"] == 45.0
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    st.add_hazard_point("r1", "CAM-01", "RB-07", "2026-11-04T07:00:00", 10.0, 20.0)
    assert st.hazard_points()[0]["rule_code"] == "RB-07"
    st.close()


def test_line_counter_counts_vehicle_once_and_skips_riders():
    scene = SiteConfig()
    scene.lines.counting.append(CountingLineCfg(id="L", name="L", points=[(0, 200), (400, 200)],
                                                direction=[(200, 100), (200, 300)]))
    w = World(scene, calib=False)
    lc = LineCounter(scene)

    def frames(i):
        y = 5 + i * 0.1
        return {1: ("car", box_at(5, y, 40, 30)), 2: ("person", box_at(15, 30 - i * 0.05, 10, 30))}, {}

    for i in range(120):
        ctx = w.step(*frames(i))
        lc.update(ctx)
    rows = lc.flush_rows()
    cars = [r for r in rows if r[3] == "car"]
    assert len(cars) == 1 and cars[0][5] == 1 and cars[0][4] == "Chiều thuận"


def test_sheet_and_html(tmp_path):
    rec = {"id": "VP20261104-CAM01-00001", "camera_id": "CAM-01", "camera_name": "Cổng A", "severity": "vi_pham",
           "rule_code": "RB-10", "rule_name": "Chạy quá tốc độ", "occurred_at": "2026-11-04T07:05:02",
           "location": "Cổng trường", "vehicle_cls": "car", "vehicle_label": "Ô tô", "plate_display": "51F-678.90",
           "plate_conf": 0.93, "plate_valid": 1, "legal_ref": "Điểm a khoản 6 Điều 6 NĐ 168/2024/NĐ-CP",
           "legal_verified": 1, "fine": "6.000.000 – 8.000.000 đồng", "status": "cho_xac_minh",
           "measurements": {"toc_do_kmh": 61.7, "phuong_phap_do": "homography", "khoa_la": 1}}
    img = np.full((720, 1280, 3), 90, np.uint8)
    sheet = compose_sheet(rec, img, img, None, img[:100, :100], None, "abc")
    assert sheet.shape == (1250, 1920, 3)
    f = dict(record_fields(rec))
    assert f["Biển kiểm soát"].startswith("51F-678.90") and f["Loại phương tiện"] == "Ô tô"
    labels = dict(meas_lines(rec["measurements"]))
    assert labels["Tốc độ ước lượng (km/h)"] == "61,7"
    assert labels["Phương pháp đo"] == "hiệu chuẩn mặt đường (homography)"
    assert "Khoa la" in labels                           # khoá chưa có nhãn vẫn hiển thị được
    assert fmt_value(None) == "—" and fmt_value(["car", "bus"]) == "ô tô, xe buýt/khách"
    # sự kiện không gắn phương tiện → biển số "Không áp dụng"
    hz = {**rec, "vehicle_cls": None, "plate_display": None, "rule_code": "HZ-04"}
    assert "Không áp dụng" in dict(record_fields(hz))["Biển kiểm soát"]
    compose_sheet(hz, img, None, None, None, None)
    (tmp_path / "composite.jpg").write_bytes(b"x")
    html = record_html({**rec, "files": {"composite.jpg": {"sha256": "ff", "size": 1}}}, tmp_path)
    assert "51F-678.90" in html and "Chờ xác minh" in html and "<script" not in html.lower()


def test_privacy_blur():
    img = np.random.default_rng(0).integers(0, 255, (200, 200, 3), np.uint8)
    before = img.copy()
    pixelate(img, [50, 50, 100, 100])
    assert not np.array_equal(img[50:100, 50:100], before[50:100, 50:100])
    assert np.array_equal(img[:40, :40], before[:40, :40])
    out = blur_bystanders(before.copy(), [{"id": 5, "cls": "person", "xyxy": [10, 10, 60, 150]}], False, [])
    assert not np.array_equal(out[10:40, 10:60], before[10:40, 10:60])


def test_build_report(tmp_path):
    paths = AppPaths(tmp_path / "data").ensure()
    st = Store(paths.db)
    _rec(st, 0)
    st.add_counts("r1", "CAM-01", [("L1", "Mặt cắt", "2026-11-04T07:00:00", "car", "Đi ra", 3)])
    st.add_speed("r1", "CAM-01", 1, "car", "2026-11-04T07:00:00", 33.0, "homography")
    out = build_report(st, paths, camera_id="CAM-01", speed_limit=40)
    html = out.read_text(encoding="utf-8")
    assert "BÁO CÁO" in html and "51F-678.90" in html
    for name in ("vi_pham.csv", "luu_luong.csv", "toc_do.csv"):
        assert (out.parent / name).exists()
    assert (out.parent / "vi_pham.csv").read_bytes().startswith(b"\xef\xbb\xbf")   # UTF-8 BOM cho Excel
    st.close()


def test_evidence_folder_survives_moved_data(tmp_path):
    from traffic_safety_ai.evidence.store import evidence_folder

    root = tmp_path / "evidence"
    (root / "2026-11-04" / "VP20261104-CAM01-00001").mkdir(parents=True)
    rec = {"id": "VP20261104-CAM01-00001", "occurred_at": "2026-11-04T07:05:00",
           "evidence_dir": "/sessions/may-ao/mnt/yolo-al/traffic_safety_ai/data/evidence/2026-11-04/VP20261104-CAM01-00001"}
    assert evidence_folder(rec, root) == root / "2026-11-04" / "VP20261104-CAM01-00001"
    assert evidence_folder({**rec, "occurred_at": None}, root) == root / "2026-11-04" / "VP20261104-CAM01-00001"
    assert evidence_folder({"id": "../../etc", "evidence_dir": None}, root) is None
    assert evidence_folder({"id": "VP-khong-co"}, root) is None


def test_line_counter_ignores_camera_shift():
    """Camera trôi xuống 30 px/khung: xe đứng yên không được đếm là cắt vạch."""
    import numpy as np

    scene = SiteConfig()
    scene.lines.counting.append(CountingLineCfg(id="L", name="L", points=[(0, 200), (400, 200)],
                                                direction=[(200, 100), (200, 300)]))
    w = World(scene, calib=False)
    lc = LineCounter(scene)
    shift = 0.0
    for i in range(20):
        shift += 30.0 if i > 0 else 0.0
        b = box_at(5, 9, 40, 30).astype(float)
        b[[1, 3]] += shift                       # cả khung trôi → hộp xe trôi theo
        ctx = w.step({1: ("car", b)}, {})
        ctx.aux["camera_warp"] = np.array([[1.0, 0, 0], [0, 1.0, 30.0]]) if i > 0 else None
        lc.update(ctx)
    assert lc.flush_rows() == []
