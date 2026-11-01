import pytest

from ares.buffers import OccupancyBuffer
from ares.config import RoomSpec, ZoneSpec, default_room, load_room
from ares.detector import Det
from ares.zones import ZoneMap, foot_point, point_in_polygon


def box(x1, y1, x2, y2):
    return Det(x1, y1, x2, y2, 0.9)


def test_foot_point_is_bottom_center():
    assert foot_point(box(100, 50, 200, 300), 400, 400) == (0.375, 0.75)


def test_zone_assignment_uses_foot_not_box_center():
    zm = ZoneMap(default_room())
    # bbox cao, đầu ở nửa trên nhưng CHÂN ở nửa dưới-trái => Zone 3 (chỉ số 2), không phải Zone 1
    c = zm.count([box(50, 10, 150, 380)], 400, 400)
    assert c.zones == [0, 0, 1, 0] and c.total == 1 and c.unzoned == 0


def test_counts_and_unzoned():
    room = RoomSpec("X", 10, (ZoneSpec(1, "a", ((0, 0), (0.5, 0), (0.5, 1), (0, 1)), 5, 3.0),))
    zm = ZoneMap(room)
    c = zm.count([box(10, 10, 50, 100), box(300, 10, 340, 100)], 400, 200)
    assert c.zones == [1] and c.unzoned == 1 and c.total == 2 and c.zoned_total == 1


def test_point_on_edge_counts_as_inside():
    sq = ((0, 0), (1, 0), (1, 1), (0, 1))
    assert point_in_polygon(1.0, 0.5, sq) and point_in_polygon(0.0, 0.0, sq)
    assert not point_in_polygon(1.01, 0.5, sq)


def test_concave_polygon():
    L = ((0, 0), (1, 0), (1, 0.4), (0.4, 0.4), (0.4, 1), (0, 1))
    assert point_in_polygon(0.2, 0.9, L) and not point_in_polygon(0.8, 0.8, L)


def test_load_room_validations(tmp_path):
    import json

    f = tmp_path / "z.json"
    f.write_text(json.dumps({"zones": [{"id": 1, "polygon": [[0, 0], [1, 0]]}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="3 điểm"):
        load_room(f)
    f.write_text(json.dumps({"zones": [{"id": 1, "polygon": [[0, 0], [5, 0], [5, 5]]}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="chuẩn hoá"):
        load_room(f)
    poly = [[0, 0], [1, 0], [1, 1]]
    f.write_text(json.dumps({"zones": [{"id": 1, "polygon": poly}, {"id": 1, "polygon": poly}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="trùng"):
        load_room(f)
    f.write_text(json.dumps({"zones": [{"id": 1, "k": 99, "polygon": poly}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="k="):
        load_room(f)
    assert load_room(tmp_path / "missing.json").n_zones == 4  # thiếu file => lưới 2x2 mặc định


def test_shipped_zone_file_is_valid():
    from ares.config import CONFIG_DIR

    room = load_room(CONFIG_DIR / "zones.json")
    assert room.n_zones == 4 and all(3.0 <= z.k <= 3.5 for z in room.zones)


# ------------------------------------------------------------------------ bộ đệm
def test_constant_signal_mean_and_full_coverage():
    b = OccupancyBuffer(30, 2, max_gap_s=20)
    for t in range(0, 31, 5):
        b.push(t, 3, [2, 1])
    s = b.stats(30)
    assert s.mean_total == pytest.approx(3) and s.mean_zones == [pytest.approx(2), pytest.approx(1)]
    assert s.coverage == pytest.approx(1.0)


def test_time_weighted_not_sample_weighted():
    b = OccupancyBuffer(30, 1, max_gap_s=100)
    b.push(0, 0, [0])
    for t in (28, 29):  # 2 mẫu "10 người" dồn ở cuối cửa sổ không được kéo trung bình lên
        b.push(t, 10, [10])
    assert b.stats(30).mean_total == pytest.approx((0 * 28 + 10 * 2) / 30)


def test_gap_longer_than_max_gap_is_missing_data_not_zero_people():
    b = OccupancyBuffer(30, 1, max_gap_s=20)
    b.push(0, 0, [0])
    b.push(25, 10, [10])
    s = b.stats(30)
    # mẫu t=0 chỉ "giữ" 20 s; t=20..25 là khoảng mất dữ liệu; t=25..30 giữ giá trị 10
    assert s.mean_total == pytest.approx(2.0) and s.coverage == pytest.approx(25 / 30)


def test_no_data_is_not_ok_and_old_data_expires():
    b = OccupancyBuffer(30, 1)
    assert not b.stats(10).ok
    b.push(0, 5, [5])
    assert not b.stats(100).ok  # camera chết lâu: KHÔNG được trả về 5, cũng không phải 0


def test_peak_and_since_nonzero():
    b = OccupancyBuffer(60, 2)
    b.push(0, 3, [3, 0])
    b.push(10, 0, [0, 0])
    s = b.stats(15)
    assert s.peak_total == 3 and s.peak_zones == [3, 0]
    assert b.seconds_since_nonzero(40) == 40  # lần cuối THẤY người là mẫu t=0
    assert b.seconds_since_nonzero(40, zone=1) is None


def test_late_sample_is_ignored_and_wrong_size_rejected():
    b = OccupancyBuffer(30, 2)
    b.push(10, 1, [1, 0])
    b.push(5, 99, [99, 0])
    assert b.stats(12).mean_total == pytest.approx(1.0)
    with pytest.raises(ValueError):
        b.push(11, 1, [1])
