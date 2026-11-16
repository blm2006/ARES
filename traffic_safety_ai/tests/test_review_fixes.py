"""Kiểm thử tái hiện các lỗi logic tìm thấy trong đợt rà soát 11/2026 (mỗi bài ứng với một lỗi đã sửa)."""

from datetime import datetime

import numpy as np
import pytest

from traffic_safety_ai.analytics.counting import SpeedTrapMeter
from traffic_safety_ai.core.config import (ConfigError, CountingLineCfg, SiteConfig, SolidLineCfg, SpeedTrapCfg,
                                           StopLineCfg, TrafficLightCfg, ZoneCfg)
from traffic_safety_ai.core.tracker import ByteTracker, TrackerConfig
from traffic_safety_ai.core.traffic_light import LightStateTracker
from traffic_safety_ai.core.types import Detection
from traffic_safety_ai.evidence.recorder import FrameRing
from traffic_safety_ai.evidence.store import evidence_folder
from traffic_safety_ai.rules.base_rule import zone_active
from traffic_safety_ai.rules.catalog import LegalCatalog
from traffic_safety_ai.rules.extra_rules import AbreastRule, ParkingRule
from traffic_safety_ai.rules.shared_rules import (LaneSidewalkRule, PedestrianConflictRule, RedLightRule,
                                                  WrongWayRule)
from traffic_safety_ai.tests.helpers import World, box_at, run

CAT = LegalCatalog()


# ------------------------------------------------------------------ giờ hiệu lực qua nửa đêm
def test_zone_active_hours_wrap_midnight():
    z = ZoneCfg(id="Z", polygon=[(0, 0), (1, 0), (1, 1)], active_hours=["22:00-05:00"])
    assert zone_active(z, datetime(2026, 11, 8, 23, 30))
    assert zone_active(z, datetime(2026, 11, 8, 3, 0))
    assert not zone_active(z, datetime(2026, 11, 8, 12, 0))
    day = ZoneCfg(id="D", polygon=[(0, 0), (1, 0), (1, 1)], active_hours=["06:30-07:30"])
    assert zone_active(day, datetime(2026, 11, 8, 7, 0)) and not zone_active(day, datetime(2026, 11, 8, 8, 0))


# ------------------------------------------------------------------ RB-11: tracker đổi ID không đếm lại từ 0
def _parking_scene():
    sc = SiteConfig()
    sc.zones.no_parking.append(ZoneCfg(id="NP1", kind="crosswalk",
                                       polygon=[(0, 300), (200, 300), (200, 360), (0, 360)]))
    return sc


def test_parking_timer_survives_track_id_change_and_reports_once():
    w = World(_parking_scene(), fps=10)
    rule = ParkingRule(w.scene, CAT)

    def frames(i):
        if 60 <= i < 80:                         # bị xe khác che 2 s
            return {}
        tid = 1 if i < 60 else 2                 # hiện lại với ID mới, đúng chỗ cũ
        return {tid: ("car", box_at(5, 17.0, 60, 50))}

    ev = run(rule, w, frames, n=125)             # 6 s + 2 s che + 4,5 s: tổng > 10 s chỉ khi đồng hồ được kế thừa
    assert len(ev) == 1 and ev[0].measurements["thoi_gian_dung_s"] >= 10
    more = run(rule, w, lambda i: {3: ("car", box_at(5, 17.0, 60, 50))}, n=60)   # đổi ID lần nữa → không báo lại
    assert more == []


def test_parking_new_car_elsewhere_starts_from_zero():
    w = World(_parking_scene(), fps=10)
    rule = ParkingRule(w.scene, CAT)

    def frames(i):
        return {1: ("car", box_at(2, 17.0, 60, 50))} if i < 70 else {2: ("car", box_at(8, 17.0, 60, 50))}

    assert run(rule, w, frames, n=125) == []     # xe thứ hai đỗ CHỖ KHÁC mới 5,5 s → chưa đủ 10 s


def test_parking_timer_kept_while_camera_shakes():
    w = World(_parking_scene(), fps=10)
    rule = ParkingRule(w.scene, CAT)

    def frames(i):
        shake = 40 <= i < 50
        dx = 0.4 * (i % 2) if shake else 0.0     # camera rung làm hộp "nhảy" → trông như đang chạy
        return ({1: ("car", box_at(5 + dx, 17.0, 60, 50))}, {"aux": {"camera_moving": shake}})

    assert len(run(rule, w, frames, n=115)) == 1


# ------------------------------------------------------------------ RB-12: nhóm thêm xe vẫn là một hồ sơ
def test_abreast_group_growing_is_one_event():
    w = World()
    rule = AbreastRule(w.scene, CAT)
    v = 12 / 3.6

    def frames(i):
        ids = (1, 2, 3) if i < 110 else (1, 2, 3, 4)
        return {k: ("bicycle", box_at(2 + k * 1.0, 1 + v * i / 30, 20, 40)) for k in ids}

    ev = run(rule, w, frames, n=260)
    assert len(ev) == 1
    rule.forget_tracks([1, 2, 3, 4])
    assert not rule._groups and not rule.fired and not rule.scores


# ------------------------------------------------------------------ RB-04: chạm vạch liền rồi quay lại
def _solid_scene():
    sc = SiteConfig()
    sc.lines.solid_lines.append(SolidLineCfg(id="V1", points=[(100, 0), (100, 600)]))   # vạch dọc tại x = 5 m
    return sc


def test_solid_line_touch_and_return_not_reported():
    w = World(_solid_scene())
    rule = LaneSidewalkRule(w.scene, CAT)

    def frames(i):
        x = 4.5 + 0.7 * np.sin(np.pi * min(i, 40) / 40)        # lấn sang bên kia tối đa 0,2 m rồi quay về làn cũ
        return {1: ("car", box_at(x - (0.6 if i > 40 else 0) * min(1, (i - 40) / 20), 2 + i * 0.3, 60, 50))}

    assert [e for e in run(rule, w, frames, n=80) if e.variant == "de_vach"] == []


def test_solid_line_real_crossing_reported():
    w = World(_solid_scene())
    rule = LaneSidewalkRule(w.scene, CAT)
    ev = run(rule, w, lambda i: {1: ("car", box_at(3.5 + min(i, 40) * 0.08, 2 + i * 0.3, 60, 50))}, n=60)
    assert len([e for e in ev if e.variant == "de_vach"]) == 1
    rule.forget_tracks([1])
    assert not rule._crossed and not rule._entry


# ------------------------------------------------------------------ RB-07: PET khi CHƯA hiệu chuẩn
def _cw_scene():
    sc = SiteConfig()
    sc.zones.crosswalks.append(ZoneCfg(id="CW", polygon=[(0, 300), (200, 300), (200, 360), (0, 360)]))
    return sc


def _pet_events(car_start_frame: int):
    w = World(_cw_scene(), calib=False)
    rule = PedestrianConflictRule(w.scene, CAT)

    def frames(i):
        objs = {1: ("person", box_at(3.0 + i * 0.05, 16.5, 20, 50))}       # người đi ngang trên vạch
        if i >= car_start_frame:
            k = i - car_start_frame
            objs[2] = ("car", box_at(3.5, 9.0 + k * 0.45, 60, 50))          # xe đi dọc, cắt qua vạch tại x = 3,5 m
        return objs

    return [e for e in run(rule, w, frames, n=car_start_frame + 30) if e.variant == "pet"]


def test_pet_detected_without_calibration():
    ev = _pet_events(car_start_frame=8)        # xe tới chỗ x = 3,5 m ngay sau khi người vừa đi qua
    assert len(ev) == 1 and 0 < ev[0].measurements["PET_s"] <= 1.0


def test_pet_not_reported_when_vehicle_comes_much_later():
    assert _pet_events(car_start_frame=70) == []


# ------------------------------------------------------------------ RB-03: đồng thuận theo cửa sổ thời gian
def _lane_scene():
    sc = SiteConfig()
    sc.zones.lanes.append(ZoneCfg(id="L1", name="Làn", polygon=[(0, 0), (200, 0), (200, 600), (0, 600)],
                                  direction=[(100, 50), (100, 550)]))
    return sc


def test_two_wrong_way_riders_in_empty_lane_are_reported():
    w = World(_lane_scene())
    rule = WrongWayRule(w.scene, CAT)
    ev = run(rule, w, lambda i: {1: ("motorcycle", box_at(3, 28 - i * 0.3, 30, 40)),
                                 2: ("motorcycle", box_at(7, 27 - i * 0.3, 30, 40))}, n=60)
    assert sorted(e.primary_track_id for e in ev) == [1, 2]


def test_everyone_against_arrow_means_misdeclared_lane():
    w = World(_lane_scene())
    rule = WrongWayRule(w.scene, CAT)
    ev = run(rule, w, lambda i: {k: ("motorcycle", box_at(1 + 2 * k, 28 - i * 0.3, 30, 40)) for k in (1, 2, 3, 4)},
             n=60)
    assert ev == []


def test_misdeclared_lane_detected_even_in_sparse_traffic():
    w = World(_lane_scene())
    rule = WrongWayRule(w.scene, CAT)

    def frames(i):                               # mỗi lúc chỉ MỘT xe trong làn, nối đuôi nhau, tất cả "ngược"
        k = i // 50
        return {10 + k: ("motorcycle", box_at(5, 28 - (i % 50) * 0.3, 30, 40))}

    ev = run(rule, w, frames, n=400)             # 8 xe
    assert len(ev) <= 2                          # chỉ vài xe đầu lọt, sau đó nhận ra khai báo sai và ngừng báo


# ------------------------------------------------------------------ RB-05: camera rung không làm mất ứng viên
def test_red_light_candidate_survives_camera_shake():
    sc = SiteConfig()
    sc.traffic_lights.append(TrafficLightCfg(id="D1", roi=(0, 0, 10, 30)))
    sc.lines.stop_lines.append(StopLineCfg(id="S1", points=[(0, 200), (200, 200)], direction=[(100, 150), (100, 250)],
                                           confirm_points=[(0, 360), (200, 360)], light_id="D1"))
    w = World(sc)
    rule = RedLightRule(w.scene, CAT)
    lt = LightStateTracker("D1")
    lt.state, lt.since_t = "red", 0.0
    lt.transitions.append((0.0, "red"))

    def frames(i):
        shake = 50 <= i < 62                     # camera rung đúng đoạn xe đi qua vạch xác nhận (y = 18 m, khung 56)
        return ({1: ("motorcycle", box_at(3, 4 + i * 0.25, 20, 30))},     # cắt vạch dừng (10 m) ở khung 24, t = 0,8 s
                {"aux": {"light_trackers": {"D1": lt}, "camera_moving": shake}})

    ev = run(rule, w, frames, n=90)
    assert [e.primary_track_id for e in ev] == [1]        # trước đây: mất luôn hồ sơ


def test_red_light_unknown_light_id_is_reported_by_check():
    sc = SiteConfig()
    sc.traffic_lights.append(TrafficLightCfg(id="D1", roi=(0, 0, 10, 30)))
    sc.lines.stop_lines.append(StopLineCfg(id="S1", points=[(0, 200), (200, 200)], direction=[(100, 150), (100, 250)],
                                           light_id="DEN-KHONG-CO"))
    ok, why = RedLightRule(sc, CAT).scene_ready("cctv")
    assert not ok and "DEN-KHONG-CO" in why


# ------------------------------------------------------------------ tracker: hạn giữ track theo thời gian thật
def test_tracker_drops_lost_track_by_time_when_frames_are_skipped():
    trk = ByteTracker(TrackerConfig(track_buffer_s=1.0), fps=30)
    box = np.array([100, 100, 160, 160], float)
    trk.update([Detection(box, 0.9, "car")], 0, 0.0)
    trk.update([Detection(box, 0.9, "car")], 1, 1 / 30)
    assert len(trk.tracks) == 1
    trk.update([], 2, 2 / 30)
    trk.update([], 3, 2 / 30 + 3.0)              # nguồn trực tiếp bỏ khung: chỉ 1 "khung" nhưng đã 3 giây
    assert len(trk.tracks) == 0


# ------------------------------------------------------------------ bằng chứng
def test_clip_fps_matches_frame_decimation():
    ring = FrameRing(2.0, fps=25.0, clip_fps=10.0)
    assert ring.step == 2 and ring.clip_fps == pytest.approx(12.5)
    assert FrameRing(2.0, fps=30.0, clip_fps=10.0).clip_fps == pytest.approx(10.0)


def test_privacy_mask_applied_to_clip_frames():
    import cv2

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    poly = [(0, 0), (80, 0), (80, 120), (0, 120)]
    jpg = FrameRing(2.0, 10.0, privacy_polygons=[poly]).push(0, 0.0, frame)[2]
    out = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert out[:, :60].std() < 0.25 * frame[:, :60].std()         # nửa trái đã bị làm mờ
    assert out[:, 100:].std() > 0.5 * frame[:, 100:].std()        # nửa phải giữ nguyên (chỉ mất chi tiết do nén JPEG)


def test_evidence_folder_ignores_foreign_directory(tmp_path):
    other = tmp_path / "khac"
    other.mkdir()
    root = tmp_path / "evidence"
    good = root / "2026-11-08" / "VP1"
    good.mkdir(parents=True)
    rec = {"id": "VP1", "occurred_at": "2026-11-08T07:00:00", "evidence_dir": str(other)}
    assert evidence_folder(rec, root) == good


# ------------------------------------------------------------------ cấu hình
def test_scaled_to_scales_rule_exempt_zones():
    sc = SiteConfig()
    sc.camera.frame_size = (1000, 500)
    sc.rules["RB-04"] = {"exempt_zones": [[(100, 100), (200, 100), (200, 200)]]}
    out = sc.scaled_to(2000, 1000)
    assert out.rules["RB-04"]["exempt_zones"][0][1] == (400.0, 200.0)
    assert sc.rules["RB-04"]["exempt_zones"][0][1] == (200, 100)       # bản gốc không bị đổi


def test_config_rejects_malformed_lines():
    with pytest.raises(ValueError):
        CountingLineCfg(id="L", points=[(0, 0)], direction=[(0, 0), (1, 1)])
    with pytest.raises(ValueError):
        StopLineCfg(id="S", points=[(0, 0), (1, 0)], direction=[(0, 0)])
    with pytest.raises(ValueError):
        SpeedTrapCfg(id="T", line_a=[(0, 0), (1, 0)], line_b=[(0, 5), (1, 5)], distance_m=0, direction=[(0, 0), (0, 5)])
    with pytest.raises(ValueError):
        TrafficLightCfg(id="D", roi=(10, 10, 5, 30))
    assert ConfigError is not None


# ------------------------------------------------------------------ vạch đo tốc độ
def test_speed_trap_ignores_b_before_a_and_camera_shake():
    sc = SiteConfig()
    sc.lines.speed_traps.append(SpeedTrapCfg(id="T", line_a=[(0, 200), (200, 200)], line_b=[(0, 400), (200, 400)],
                                             distance_m=10.0, direction=[(100, 150), (100, 250)]))
    w = World(sc, calib=False)
    meter = SpeedTrapMeter(sc)
    got = []
    for i in range(60):                           # 10 m (200 px) trong 20 khung = 0,667 s → 54 km/h
        ctx = w.step({1: ("car", box_at(5, 8 + i * 0.5, 60, 50))})
        got += meter.update(ctx)
    assert len(got) == 1 and got[0][1] == pytest.approx(54.0, rel=0.03)
    w2 = World(sc, calib=False)
    meter2 = SpeedTrapMeter(sc)
    got2 = []
    for i in range(60):
        ctx = w2.step({1: ("car", box_at(5, 8 + i * 0.5, 60, 50))}, aux={"camera_moving": True})
        got2 += meter2.update(ctx)
    assert got2 == []
    meter.forget(1)
    assert not meter._t
