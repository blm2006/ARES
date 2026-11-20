"""Cổng sinh người mới của tracker: không cho người "mọc" giữa khung hình sau mấy giây đầu."""

import numpy as np

from ares.behavior.tracker import Tracker

W, H = 640, 480


def det(*boxes):
    b = np.array(boxes, np.float32).reshape(-1, 4)
    return np.zeros((len(b), 17, 3), np.float32), b, np.full(len(b), 0.9, np.float32)


def run(tr, t, *boxes):
    return tr.update(t, *det(*boxes), frame_wh=(W, H))


def ids(tr, t):
    return sorted(k for k, v in tr.tracks.items() if abs(v.last_t - t) < 1e-6)


def test_warmup_accepts_anyone():
    tr = Tracker(gate=True, warmup_s=2.0)
    run(tr, 0.0, (300, 150, 360, 330))
    run(tr, 1.0, (300, 150, 360, 330), (100, 150, 160, 330))  # người thứ 2 thấy ở giây 1: vẫn trong warmup
    assert len(ids(tr, 1.0)) == 2


def test_duplicate_box_inside_person_is_dropped():
    tr = Tracker(gate=True, warmup_s=1.0)
    for k in range(30):
        t = k * 0.1
        boxes = [(300, 150, 360, 330)]
        if t > 2.0:
            boxes.append((305, 160, 355, 250))  # khung "nửa người" chồng lên đúng người đó
        run(tr, t, *boxes)
    assert ids(tr, 2.9) == [1]
    assert tr.stats["dup_dropped"] > 0


def test_mid_frame_ghost_needs_confirmation():
    tr = Tracker(gate=True, warmup_s=1.0, confirm_s=4.0)
    for k in range(25):
        run(tr, k * 0.1, (100, 150, 160, 330))
    # 1 khung lạ ở giữa (không chồng ai) xuất hiện chập chờn: không bao giờ thành người
    for k in range(25, 60):
        t = k * 0.1
        run(tr, t, (100, 150, 160, 330), *([(380, 160, 440, 340)] if k % 4 else []))
        assert ids(tr, t) == [1]
    # nhưng nếu đứng đó liên tục đủ 4 giây (người khuất từ đầu rồi đứng dậy) thì được nhận
    for k in range(60, 110):
        run(tr, k * 0.1, (100, 150, 160, 330), (380, 160, 440, 340))
    assert len(ids(tr, 10.9)) == 2


def test_confirm_zero_never_accepts_mid_frame():
    tr = Tracker(gate=True, warmup_s=1.0, confirm_s=0.0)
    for k in range(100):
        run(tr, k * 0.1, (100, 150, 160, 330), *([(380, 160, 440, 340)] if k > 20 else []))
    assert ids(tr, 9.9) == [1]


def test_person_entering_from_edge_is_born_at_once():
    tr = Tracker(gate=True, warmup_s=1.0)
    for k in range(20):
        run(tr, k * 0.1, (300, 150, 360, 330))
    run(tr, 2.0, (300, 150, 360, 330), (0, 150, 50, 330))  # chạm mép trái
    assert len(ids(tr, 2.0)) == 2


def test_lost_person_reappears_with_same_id():
    tr = Tracker(gate=True, warmup_s=1.0, recover_s=6.0)
    for k in range(20):
        run(tr, k * 0.1, (300, 150, 360, 330))
    for k in range(20, 50):  # bị che 3 giây
        run(tr, k * 0.1)
    run(tr, 5.0, (320, 150, 380, 330))
    assert ids(tr, 5.0) == [1]


def test_entry_zone_allows_birth_in_middle():
    door = [(350, 100), (480, 100), (480, 360), (350, 360)]
    tr = Tracker(gate=True, warmup_s=1.0, entry_zones=[door])
    for k in range(20):
        run(tr, k * 0.1, (100, 150, 160, 330))
    run(tr, 2.0, (100, 150, 160, 330), (380, 160, 440, 340))
    assert len(ids(tr, 2.0)) == 2


def test_gate_off_keeps_old_behaviour():
    tr = Tracker()
    run(tr, 0.0, (100, 150, 160, 330))
    run(tr, 5.0, (380, 160, 440, 340))
    assert len(tr.tracks) >= 1 and max(tr.tracks) == 2
