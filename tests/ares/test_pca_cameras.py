"""Kiểm tra lọc PCA hai giai đoạn và lưới nhiều camera (dashboard kiểu CCTV)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ares.active_learner import _Job
from ares.cameras import grid_dims
from ares.clock import ManualClock
from ares.config import AresSettings
from ares.dashboard.app import create_app
from ares.detector import Det
from ares.pca_filter import select_diverse
from ares.runtime import AresRuntime

H = {"X-ARES": "1"}


def scene(seed: int, noise: float = 0.0) -> np.ndarray:
    """Khung hình giả: vài khối màu ở vị trí ngẫu nhiên (seed khác -> bố cục khác). noise nhỏ -> gần như cùng một cảnh."""
    r = np.random.default_rng(seed)
    img = np.full((120, 160, 3), r.integers(30, 220, 3), dtype=np.uint8)
    for _ in range(4):
        x, y = r.integers(0, 120), r.integers(0, 80)
        img[y:y + 40, x:x + 40] = r.integers(0, 255, 3)
    if noise:
        img = np.clip(img.astype(np.int16) + np.random.default_rng(seed + 999).integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    return img


def build(tmp_path, **over):
    base = dict(storage_root=tmp_path / "a", simulation_mode=True, overlay_interval_s=0.0, al_min_gap_s=0.0, al_hourly_cap=1000,
                pca_fit_min=60, pca_fit_min_manual=40, pca_train_skip_below=4, pca_triage_delay_s=0.0, pca_corpus_interval_s=1.0)
    cfg = AresSettings(_env_file=None, **{**base, **over})
    cfg.ensure_dirs()
    clock = ManualClock(datetime(2026, 11, 2, 7, 5))
    return AresRuntime(cfg, clock=clock, with_dashboard=False), clock


@pytest.fixture
def rt(tmp_path):
    r, clock = build(tmp_path)
    yield r
    r.al.drain(5); r.al.stop(); r.pca.close(); r.hub.close(); r.db.close()


def fit_pca(rt, n=50):
    for i in range(n):
        rt.clock.advance(2)
        assert rt.pca.offer_corpus("main", scene(1000 + i))
    assert rt.pca.fit(manual=True) is not None


def save(rt, seed, noise=0.0, k=0):
    return rt.al.save(_Job(scene(seed, noise), [Det(16, 12, 80, 108, 0.45)], rt.clock.now() + timedelta(seconds=k)))


# ------------------------------------------------------------------ thuật toán
@pytest.mark.parametrize("n,dims", [(0, (1, 1)), (1, (1, 1)), (2, (2, 1)), (3, (2, 2)), (4, (2, 2)), (6, (3, 2)), (9, (3, 3)),
                                     (12, (4, 3)), (16, (4, 4)), (17, (5, 4))])
def test_grid_dims(n, dims):
    assert grid_dims(n) == dims


def unit(rows):
    z = np.asarray(rows, dtype=np.float32)
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def test_select_diverse_collapses_duplicates_and_keeps_forced():
    z = unit([[1, 0, 0], [1, 0.01, 0], [1, 0, 0.01], [0, 1, 0], [0, 1, 0.01], [0, 0, 1]])
    picked, dist, ref = select_diverse(z, None, radius=0.05)
    assert len(picked) == 3  # 3 cụm -> 3 đại diện
    picked, *_ = select_diverse(z, None, radius=0.05, forced=[2])
    assert 2 in picked and len(picked) == 3  # ảnh bắt buộc làm đại diện cụm của nó
    picked, *_ = select_diverse(z, None, radius=0.05, min_keep=5)
    assert len(picked) == 5


def test_select_diverse_seeds_suppress_known_scenes():
    z = unit([[1, 0, 0], [0, 1, 0]])
    picked, *_ = select_diverse(z, unit([[1, 0.001, 0]]), radius=0.05)
    assert picked == [1]  # cảnh 0 đã có trong seeds (đã train) -> không cần chọn lại


# ------------------------------------------------------------------ giai đoạn 1
@pytest.fixture
def rt_keep(tmp_path):
    """Giữ ảnh trùng 3 ngày (khôi phục được) thay vì xoá ngay như mặc định."""
    r, clock = build(tmp_path, duplicate_ttl_days=3)
    yield r
    r.al.drain(5); r.al.stop(); r.pca.close(); r.hub.close(); r.db.close()


def test_default_deletes_duplicates_immediately(rt):
    ids = [save(rt, 1 + s, noise=0.5 if d else 0.0, k=s * 4 + d) for s in range(5) for d in range(4)]
    paths = {i: rt.al.image_path(i) for i in ids}
    fit_pca(rt)
    res = rt.pca.triage(force=True)
    c = rt.al.counts()
    assert res["status"] == "done" and res["dropped"] >= 8
    assert c["DUPLICATE"] == 0 and c["PENDING"] == res["kept"]  # không còn dòng DUPLICATE nào
    kept = {it["image_id"] for it in rt.al.list_items("PENDING", 50, 0)}
    for i, pth in paths.items():
        assert pth.exists() == (i in kept)  # ảnh trùng đã bị xoá khỏi đĩa, ảnh giữ lại còn nguyên
        assert not pth.with_suffix(".txt").exists() or i in kept
    st = rt.pca.status()
    assert st["duplicates_deleted_now"] and st["duplicates_total"] == res["dropped"]


def test_items_wait_for_pca_then_duplicates_are_filtered_and_restorable(rt_keep):
    rt = rt_keep
    ids = [save(rt, 1 + s, noise=0.5 if d else 0.0, k=s * 4 + d) for s in range(5) for d in range(4)]
    assert all(rt.al.get_item(i)["status"] == "CANDIDATE" for i in ids)  # chưa có PCA -> giữ lại, người chưa thấy
    assert rt.pca.triage(force=True)["status"] == "waiting_pca"
    assert rt.al.counts()["PENDING"] == 0
    fit_pca(rt)
    res = rt.pca.triage(force=True)
    c = rt.al.counts()
    assert res["status"] == "done" and c["CANDIDATE"] == 0
    assert c["PENDING"] + c["DUPLICATE"] == 20
    assert 3 <= c["PENDING"] <= 12 and c["DUPLICATE"] >= 8  # 5 cảnh x 4 bản gần giống -> còn ít hơn nhiều
    dup = rt.al.list_items("DUPLICATE", 50, 0)[0]
    assert dup["dup_of"] and rt.al.restore(dup["image_id"])["status"] == "PENDING"
    with pytest.raises(ValueError):
        rt.al.restore(dup["image_id"])  # đã khôi phục rồi


def test_release_all_skips_filter(rt):
    ids = [save(rt, s, k=s) for s in range(3)]
    assert rt.pca.release_all() == 3
    assert {rt.al.get_item(i)["status"] for i in ids} == {"PENDING"}


# ------------------------------------------------------------------ giai đoạn 2
def test_stage2_reduces_but_keeps_edited_and_excludes_dropped_from_training(rt):
    fit_pca(rt)
    ids = [save(rt, 1 + s, noise=0.5 if d else 0.0, k=s * 4 + d) for s in range(5) for d in range(4)]
    rt.pca.release_all()
    for i in ids:
        rt.al.approve(i)
    edited = ids[1]  # bản "trùng" nhưng người đã sửa bbox
    rt.al.edit(edited, [{"bbox": [10, 10, 60, 100], "class": 0}])
    rep = rt.pca.select_for_training(rt.al.items_for_training(), commit=False)
    assert rep["n_new"] == 20 and rep["n_picked"] < 20 and rep["reduction_pct"] > 30
    assert edited in rep["picked_ids"]
    assert rt.al.get_item(ids[0]).get("train_pick") in (None, True)  # commit=False không ghi gì
    rep = rt.pca.select_for_training(rt.al.items_for_training(), commit=True)
    assert rt.al.get_item(edited)["train_pick"] is True
    assert sum(1 for i in ids if rt.al.get_item(i)["train_pick"] is False) == rep["n_dropped"] > 0


def test_stage2_small_batch_is_kept_whole(rt):
    fit_pca(rt)
    ids = [save(rt, 5 + s, k=s) for s in range(3)]
    rt.pca.release_all()
    for i in ids:
        rt.al.approve(i)
    rep = rt.pca.select_for_training(rt.al.items_for_training(), commit=False)
    assert rep["n_picked"] == 3 and rep["n_dropped"] == 0


# ------------------------------------------------------------------ nhiều camera
@pytest.fixture
def cams_rt(tmp_path):
    r, clock = build(tmp_path / "c", sim_cameras=4, pca_enabled=False)
    for _ in range(40):
        r.clock.advance(5)
        r.world.advance(r.clock.time())
        r.tracker.tick()
        r.hub.tick()
    yield r
    r.al.drain(5); r.al.stop(); r.hub.close(); r.db.close()


def test_hub_lists_all_sim_cameras_with_people_counts(cams_rt):
    o = cams_rt.hub.overview()
    assert o["count"] == 4 and o["grid"] == {"cols": 2, "rows": 2}
    assert [c["control"] for c in o["cameras"]] == [True, False, False, False]
    assert all(c["online"] and c["has_frame"] and isinstance(c["n_people"], int) for c in o["cameras"])
    assert o["total_people"] == sum(c["n_people"] for c in o["cameras"])
    jpg, _ = cams_rt.hub.frame(o["cameras"][1]["id"])
    assert jpg[:2] == b"\xff\xd8"
    with pytest.raises(KeyError):
        cams_rt.hub.frame("khongco")


def test_camera_api(cams_rt):
    c = TestClient(create_app(cams_rt))
    o = c.get("/api/cameras").json()
    assert o["count"] == 4
    assert c.get(f"/api/cameras/{o['cameras'][2]['id']}/frame.jpg").headers["content-type"] == "image/jpeg"
    assert c.get("/api/cameras/main/frame.jpg").status_code in (200, 404)
    assert c.get("/api/cameras/..%2Fx/frame.jpg").status_code in (400, 404)
    assert c.get("/api/cameras/khongco/frame.jpg").status_code == 404
    assert c.post("/api/cameras/rescan").status_code == 403  # thiếu header chống CSRF
    assert c.post("/api/cameras/rescan", headers=H).status_code == 200


def test_pca_api_when_disabled(cams_rt):
    c = TestClient(create_app(cams_rt))
    assert c.get("/api/pca/status").json() == {"enabled": False}
    assert c.post("/api/pca/fit", headers=H).status_code == 409
    assert c.get("/api/pca/train-preview").status_code == 409


def test_pca_api_flow(rt_keep):
    rt = rt_keep
    c = TestClient(create_app(rt))
    st = c.get("/api/pca/status").json()
    assert st["enabled"] and not st["fitted"]
    assert c.post("/api/pca/fit", headers=H).status_code == 409  # chưa đủ ảnh mẫu
    fit_pca(rt)
    ids = [save(rt, 1 + s, noise=0.5 if d else 0.0, k=s * 4 + d) for s in range(5) for d in range(4)]
    r = c.post("/api/pca/triage", headers=H).json()
    assert r["triage"]["status"] == "done" and r["status"]["fitted"]
    items = c.get("/api/al/items?status=DUPLICATE&limit=50").json()["items"]
    assert items and c.post(f"/api/al/items/{items[0]['image_id']}/restore", headers=H).status_code == 200
    assert c.get("/api/al/items?status=NOPE").status_code == 400
    assert len(ids) == 20
    assert c.get("/api/pca/train-preview").status_code == 200
