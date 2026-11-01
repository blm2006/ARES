from datetime import datetime, timedelta

import numpy as np
import pytest

from ares.active_learner import ActiveLearner, ahash, hamming, yolo_label_text
from ares.config import AresSettings
from ares.detector import Det

W, H = 160, 120


def frame(seed):
    return np.random.default_rng(seed).integers(0, 255, (H, W, 3), dtype=np.uint8)


def band_det(conf=0.45):
    return [Det(10, 10, 60, 110, conf)]


def when(clock, minutes=0):
    return clock.now() + timedelta(minutes=minutes)


def pending(al):
    return al.list_items("PENDING", 100)


def test_only_uncertain_band_is_collected(al, clock):
    assert al.offer(frame(1), band_det(0.34), when(clock)) is False  # dưới dải
    assert al.offer(frame(2), band_det(0.56), when(clock)) is False  # chắc chắn
    assert al.offer(frame(3), [], when(clock)) is False
    assert al.offer(frame(4), band_det(0.35), when(clock)) is True  # biên dưới thuộc dải
    assert al.offer(frame(5), band_det(0.55), when(clock, 1)) is True  # biên trên thuộc dải
    al.drain()
    assert len(pending(al)) == 2


def test_full_frame_original_and_label_file_saved(al, cfg, clock):
    img = frame(7)
    assert al.offer(img, [Det(16, 12, 80, 108, 0.40)], when(clock))
    al.drain()
    it = pending(al)[0]
    assert (it["width"], it["height"]) == (W, H) and it["status"] == "PENDING"
    jpg = cfg.storage_root / "storage" / "pending" / f"{it['image_id']}.jpg"
    assert jpg.exists() and jpg.with_suffix(".txt").exists()
    assert it["pred_bboxes"][0]["conf"] == pytest.approx(0.4)
    import cv2
    got = cv2.imdecode(np.fromfile(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert got.shape == img.shape  # không resize/crop
    cx = float(jpg.with_suffix(".txt").read_text().split()[1])
    assert cx == pytest.approx((16 + 80) / 2 / W, abs=1e-4)


def test_offer_does_not_modify_frame_and_is_cheap(al, clock):
    img = frame(8)
    before = img.copy()
    al.offer(img, band_det(), when(clock))
    assert np.array_equal(img, before)


def test_daily_cap(mk_cfg, clock, db, events):
    cfg = mk_cfg(al_daily_cap=3, al_hourly_cap=100, al_min_gap_s=0.0)
    a = ActiveLearner(cfg, clock, db, emit=events)
    a.start()
    got = [a.offer(frame(i), band_det(), when(clock, i)) for i in range(6)]
    a.drain()
    assert got == [True, True, True, False, False, False] and a.skipped.get("cap_ngày") == 3
    assert events.kinds().count("AL_DAILY_CAP") == 1  # báo một lần, không spam
    # sang ngày mới: cap được đặt lại
    assert a.offer(frame(99), band_det(), when(clock, 24 * 60 + 5)) is True
    a.stop()


def test_cap_counts_images_already_in_db_after_restart(mk_cfg, clock, db):
    cfg = mk_cfg(al_daily_cap=2, al_hourly_cap=100, al_min_gap_s=0.0)
    a = ActiveLearner(cfg, clock, db)
    a.start()
    a.offer(frame(1), band_det(), when(clock)); a.offer(frame(2), band_det(), when(clock, 1)); a.drain(); a.stop()
    b = ActiveLearner(cfg, clock, db)  # khởi động lại giữa ngày
    b.start()
    assert b.offer(frame(3), band_det(), when(clock, 2)) is False
    b.stop()


def test_hourly_cap_and_min_gap(mk_cfg, clock, db):
    cfg = mk_cfg(al_daily_cap=50, al_hourly_cap=2, al_min_gap_s=60.0)
    a = ActiveLearner(cfg, clock, db)
    a.start()
    t = when(clock)
    assert a.offer(frame(1), band_det(), t)
    assert not a.offer(frame(2), band_det(), t + timedelta(seconds=30)) and a.skipped["giãn_cách"] == 1
    assert a.offer(frame(3), band_det(), t + timedelta(seconds=90))
    assert not a.offer(frame(4), band_det(), t + timedelta(seconds=200)) and a.skipped["cap_giờ"] == 1
    a.stop()


def test_near_duplicate_frames_are_dropped(al, clock):
    img = frame(11)
    assert al.offer(img, band_det(), when(clock))
    noisy = np.clip(img.astype(int) + np.random.default_rng(0).integers(-2, 3, img.shape), 0, 255).astype(np.uint8)
    assert not al.offer(noisy, band_det(), when(clock, 5)) and al.skipped["ảnh_trùng"] == 1
    assert al.offer(frame(12), band_det(), when(clock, 6))


def test_ahash_distance_properties():
    a = frame(1)
    assert hamming(ahash(a), ahash(a)) == 0
    assert hamming(ahash(a), ahash(frame(2))) > 6


def test_non_ascii_storage_path_on_windows_style(tmp_path, clock):
    from ares.db_manager import DBManager

    cfg = AresSettings(_env_file=None, storage_root=tmp_path / "Ảnh phòng Việt Nam", al_min_gap_s=0.0)
    cfg.ensure_dirs()
    db = DBManager(cfg.database_url, cfg.room_id, clock)
    a = ActiveLearner(cfg, clock, db)
    a.start()
    assert a.offer(frame(3), band_det(), clock.now())
    a.drain()
    item = a.list_items()[0]
    assert a.image_path(item["image_id"]).exists()
    a.stop(); db.close()


# --------------------------------------------------------------------------- duyệt ảnh
def one(al, clock, seed=21):
    al.offer(frame(seed), [Det(16, 12, 80, 108, 0.45), Det(100, 20, 150, 110, 0.40)], when(clock, seed))
    al.drain()
    return pending(al)[0]["image_id"]


def test_approve_moves_to_approved_with_label(al, cfg, clock):
    iid = one(al, clock)
    assert al.approve(iid)["status"] == "APPROVED"
    a = cfg.approved_dir / f"{iid}.jpg"
    assert a.exists() and len(a.with_suffix(".txt").read_text().strip().splitlines()) == 2
    assert not (cfg.pending_dir / f"{iid}.jpg").exists() and not (cfg.pending_dir / f"{iid}.txt").exists()
    assert al.counts()["APPROVED"] == 1 and al.counts()["PENDING"] == 0


def test_edit_replaces_boxes_and_clamps(al, cfg, clock):
    iid = one(al, clock)
    al.edit(iid, [{"bbox": [-20, -5, 70, 300], "class": 0}, {"bbox": [50, 50, 51, 51]}])  # hộp 2 bị loại (suy biến)
    lab = (cfg.approved_dir / f"{iid}.txt").read_text().strip().splitlines()
    assert len(lab) == 1
    cx, cy, w, h = map(float, lab[0].split()[1:])
    assert 0 <= cx <= 1 and 0 <= cy <= 1 and w <= 1 and h <= 1
    assert al.get_item(iid)["final_bboxes"][0]["bbox"] == [0.0, 0.0, 70.0, 120.0]


def test_edit_with_no_boxes_is_a_valid_negative_sample(al, cfg, clock):
    iid = one(al, clock)
    al.edit(iid, [])
    assert (cfg.approved_dir / f"{iid}.txt").exists() and (cfg.approved_dir / f"{iid}.txt").read_text() == ""


def test_re_edit_of_approved_image_works(al, cfg, clock):
    iid = one(al, clock)
    al.approve(iid)
    al.edit(iid, [{"bbox": [10, 10, 40, 100]}])  # sửa lại ảnh đã duyệt: không được lỗi "same file"
    assert (cfg.approved_dir / f"{iid}.jpg").exists() and al.get_item(iid)["status"] == "APPROVED"


def test_reject_moves_to_rejected_and_keeps_reason(al, cfg, clock):
    iid = one(al, clock)
    al.reject(iid, "Ảnh mờ")
    assert (cfg.rejected_dir / f"{iid}.jpg").exists() and not (cfg.pending_dir / f"{iid}.jpg").exists()
    assert al.get_item(iid)["reject_reason"] == "Ảnh mờ" and al.get_item(iid)["status"] == "REJECTED"


def test_invalid_transitions_and_unknown_ids(al, clock):
    iid = one(al, clock)
    al.approve(iid)
    with pytest.raises(ValueError):
        al.approve(iid)
    with pytest.raises(ValueError):
        al.reject(iid)
    with pytest.raises(KeyError):
        al.approve("khong_ton_tai")
    with pytest.raises(ValueError):
        al.edit(iid, [{"bbox": "hỏng"}])


def test_stats_and_training_selection(al, clock):
    a = one(al, clock, 31)
    al.approve(a)
    assert [r.image_id for r in al.items_for_training()] == [a]
    assert al.mark_trained([a], "v0001") == 1 and al.get_item(a)["status"] == "TRAINED" and al.get_item(a)["trained_in"] == "v0001"
    assert al.stats()["today"]["cap"] is None  # không giới hạn: việc giảm số lượng do PCA đảm nhiệm


def test_queue_full_drops_instead_of_blocking(mk_cfg, clock, db):
    cfg = mk_cfg(al_queue_size=1, al_min_gap_s=0.0, al_hourly_cap=100)
    a = ActiveLearner(cfg, clock, db)  # CHƯA start luồng ghi => hàng đợi không rỗng
    assert a.offer(frame(1), band_det(), clock.now())
    assert not a.offer(frame(2), band_det(), clock.now() + timedelta(seconds=1)) and a.skipped["hàng_đợi_đầy"] == 1


def test_ttl_purge(al, cfg, clock, db):
    iid = one(al, clock)
    al.reject(iid, "x")
    clock.advance(86400 * (cfg.rejected_ttl_days + 1))
    assert al.purge()["rejected"] == 1 and not (cfg.rejected_dir / f"{iid}.jpg").exists()


def test_yolo_label_text_normalised():
    t = yolo_label_text([{"class": 0, "bbox": [0, 0, 100, 50]}], 200, 100)
    assert t == "0 0.250000 0.250000 0.500000 0.500000\n"
