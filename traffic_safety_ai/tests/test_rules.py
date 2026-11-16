from collections import deque

import numpy as np
import pytest

from traffic_safety_ai.core.config import SiteConfig, StopLineCfg, TrafficLightCfg, ZoneCfg
from traffic_safety_ai.core.ego_motion import EgoMotion
from traffic_safety_ai.core.traffic_light import LightStateTracker
from traffic_safety_ai.rules.catalog import LegalCatalog
from traffic_safety_ai.rules.cctv_rules import (ExcessPassengersRule, NoHelmetRule, OverloadRule, SpeedingRule,
                                                speed_variant)
from traffic_safety_ai.rules.extra_rules import AbreastRule, JaywalkRule, ParkingRule
from traffic_safety_ai.rules.pov_rules import SuddenEmergenceRule, TurnNoSignalRule
from traffic_safety_ai.rules.registry import build_rules
from traffic_safety_ai.rules.shared_rules import (FollowingRule, LaneSidewalkRule, PedestrianConflictRule,
                                                  PhoneUseRule, RedLightRule, WrongWayRule)
from traffic_safety_ai.tests.helpers import PX_PER_M, World, box_at, run

CAT = LegalCatalog()


def moto_and_riders(y_m, n_riders, x_m=5.0, child=False):
    objs = {1: ("motorcycle", box_at(x_m, y_m, 30, 40))}
    riders = []
    for k in range(n_riders):
        h = 70 if not (child and k == n_riders - 1) else 40
        objs[10 + k] = ("person", box_at(x_m, y_m - 0.3 * k, 26, h))
        riders.append(10 + k)
    return objs, riders


# ----------------------------------------------------------------- RB-06
@pytest.mark.parametrize("n,variant,sev", [(3, "3_nguoi", "nghi_van"), (4, "4_nguoi_tro_len", "vi_pham")])
def test_excess_passengers_tiers(n, variant, sev):
    w = World()
    rule = ExcessPassengersRule(w.scene, CAT)

    def frames(i):
        objs, riders = moto_and_riders(2 + i * 0.25, n)
        return objs, {"riders": {1: riders}}

    ev = run(rule, w, frames, n=60)
    assert len(ev) == 1
    assert ev[0].variant == variant and ev[0].severity == sev
    assert "Điều 7" in ev[0].legal["ref"]


def test_excess_passengers_ignores_stopped_moto():
    w = World()
    rule = ExcessPassengersRule(w.scene, CAT)
    objs, riders = moto_and_riders(5, 4)
    ev = run(rule, w, lambda i: (objs, {"riders": {1: riders}}), n=60)
    assert ev == []


# ----------------------------------------------------------------- RB-01
def test_no_helmet_driver_and_child_passenger():
    w = World()
    rule = NoHelmetRule(w.scene, CAT)
    obs_d = deque()
    obs_p = deque()

    def frames(i):
        objs, riders = moto_and_riders(2 + i * 0.25, 2, child=True)
        t = i / 30
        obs_d.append((t, "no_helmet", 0.9))
        obs_p.append((t, "no_helmet", 0.8))
        return objs, {"riders": {1: riders}, "attrs": {10: {"helmet_obs": obs_d}, 11: {"helmet_obs": obs_p}}}

    ev = run(rule, w, frames, n=40)
    variants = {e.variant: e for e in ev}
    assert set(variants) == {"driver", "passenger"}
    assert variants["driver"].legal["ref"].startswith("Điểm h khoản 2 Điều 7")
    assert variants["passenger"].severity == "nghi_van"     # dáng trẻ em → cần xác minh


def test_helmet_ok_no_event():
    w = World()
    rule = NoHelmetRule(w.scene, CAT)
    obs = deque()

    def frames(i):
        objs, riders = moto_and_riders(2 + i * 0.25, 1)
        obs.append((i / 30, "helmet", 0.9))
        return objs, {"riders": {1: riders}, "attrs": {10: {"helmet_obs": obs}}}

    assert run(rule, w, frames, n=40) == []


# ----------------------------------------------------------------- RB-10
def test_speed_variant_brackets():
    assert speed_variant("motorcycle", 4.9) is None
    assert speed_variant("motorcycle", 7) == "moto_5_10"
    assert speed_variant("motorcycle", 15) == "moto_10_20"
    assert speed_variant("motorcycle", 25) == "moto_20_plus"
    assert speed_variant("car", 30) == "car_20_35"
    assert speed_variant("truck", 40) == "car_35_plus"


@pytest.mark.parametrize("kmh,expect", [(44.0, None), (52.0, "moto_10_20"), (70.0, "moto_20_plus")])
def test_speeding_motorcycle(kmh, expect):
    scene = SiteConfig()
    scene.speed.limit_kmh = 40
    w = World(scene)
    rule = SpeedingRule(w.scene, CAT)
    v = kmh / 3.6
    ev = run(rule, w, lambda i: {1: ("motorcycle", box_at(5, 1 + v * i / 30, 30, 40))}, n=45)
    if expect is None:
        assert ev == []
    else:
        assert len(ev) == 1 and ev[0].variant == expect
        assert ev[0].measurements["toc_do_kmh"] == pytest.approx(kmh, abs=1.5)


# ----------------------------------------------------------------- RB-03
def test_wrong_way_cctv():
    scene = SiteConfig()
    lane = ZoneCfg(id="L1", name="Làn phải", polygon=[(0, 0), (200, 0), (200, 600), (0, 600)],
                   direction=[(100, 50), (100, 550)])          # hướng đúng: y tăng
    scene.zones.lanes.append(lane)
    w = World(scene)
    rule = WrongWayRule(w.scene, CAT)

    def frames(i):
        return {1: ("car", box_at(3, 2 + i * 0.3, 60, 50)),        # đúng chiều
                2: ("motorcycle", box_at(7, 28 - i * 0.3, 30, 40))}  # ngược chiều

    ev = run(rule, w, frames, n=60)
    assert [e.primary_track_id for e in ev] == [2]
    assert ev[0].legal["ref"].startswith("Điểm a khoản 7 Điều 7")


def test_wrong_way_unavailable_without_lane_direction():
    rule = WrongWayRule(SiteConfig(), CAT)
    ok, why = rule.check_available({"vehicles"}, "cctv")
    assert not ok and "mũi tên" in why


# ----------------------------------------------------------------- RB-04
def _sidewalk_scene():
    scene = SiteConfig()
    scene.zones.sidewalks.append(ZoneCfg(id="VH1", name="Vỉa hè trước cổng", polygon=[(0, 0), (40, 0), (40, 600), (0, 600)]))
    return scene


def test_sidewalk_riding_along():
    w = World(_sidewalk_scene())
    rule = LaneSidewalkRule(w.scene, CAT)
    ev = run(rule, w, lambda i: {1: ("motorcycle", box_at(1.0, 2 + i * 0.15, 20, 30))}, n=60)
    assert len(ev) == 1 and ev[0].variant == "via_he"


def test_sidewalk_parked_or_gate_crossing_not_reported():
    scene = _sidewalk_scene()
    scene.rules["RB-04"] = {"exempt_zones": [[(0, 280), (40, 280), (40, 330), (0, 330)]]}
    w = World(scene)
    rule = LaneSidewalkRule(w.scene, CAT)
    # 1 đỗ trên vỉa hè (RB-11 xử lý), 2 đi ngang qua cổng (vùng miễn trừ)
    ev = run(rule, w, lambda i: {1: ("motorcycle", box_at(1.0, 5, 20, 30)),
                                 2: ("motorcycle", box_at(0.1 + i * 0.05, 15.3, 20, 30))}, n=60)
    assert ev == []


# ----------------------------------------------------------------- RB-05
def _red_light_scene():
    scene = SiteConfig()
    scene.traffic_lights.append(TrafficLightCfg(id="D1", roi=(500, 10, 520, 60)))
    scene.lines.stop_lines.append(StopLineCfg(id="V1", points=[(0, 200), (200, 200)], direction=[(100, 100), (100, 300)],
                                              confirm_points=[(0, 320), (200, 320)]))
    return scene


def _light(red_from: float):
    lt = LightStateTracker("D1")
    lt.transitions.append((-30.0, "green"))
    lt.transitions.append((red_from, "red"))
    return lt


def test_red_light_two_line_logic():
    w = World(_red_light_scene())
    rule = RedLightRule(w.scene, CAT)
    lt = _light(0.0)
    aux = {"light_trackers": {"D1": lt}}

    def frames(i):
        return ({1: ("motorcycle", box_at(3, 6 + i * 0.25, 20, 30)),       # cắt vạch lúc đỏ rồi đi tiếp → vi phạm
                 2: ("car", box_at(7, 8 + min(i, 10) * 0.21, 50, 40))},     # nhích qua vạch rồi dừng → không báo
                {"aux": aux})

    ev = run(rule, w, frames, n=60)
    assert [e.primary_track_id for e in ev] == [1]
    assert ev[0].measurements["trang_thai_den_khi_cat_vach"] == "ĐỎ"
    assert ev[0].legal["ref"].startswith("Điểm c khoản 7 Điều 7")


def test_red_light_grace_and_green():
    w = World(_red_light_scene())
    rule = RedLightRule(w.scene, CAT)
    lt = _light(1.25)      # đỏ bật lúc 1.25 s, xe cắt vạch ~1.33 s (< 0.4 s grace)
    aux = {"light_trackers": {"D1": lt}}
    ev = run(rule, w, lambda i: ({1: ("motorcycle", box_at(3, 6 + i * 0.1, 20, 30))}, {"aux": aux}), n=90)
    assert ev == []


# ----------------------------------------------------------------- RB-07
def _crosswalk_scene():
    scene = SiteConfig()
    scene.zones.crosswalks.append(ZoneCfg(id="CW", polygon=[(0, 300), (200, 300), (200, 360), (0, 360)]))
    return scene


def test_pedestrian_conflict_not_yielding():
    w = World(_crosswalk_scene())
    rule = PedestrianConflictRule(w.scene, CAT)

    def frames(i):
        return {1: ("person", box_at(4 + i * 0.03, 16.5, 20, 50)), 2: ("car", box_at(4.5, 2 + i * 0.4, 60, 50))}

    ev = run(rule, w, frames, n=30)
    assert ev and ev[0].rule_code == "RB-07" and ev[0].measurements["TTC_s"] < 2.5


def test_pedestrian_conflict_decelerating_vehicle_ok():
    w = World(_crosswalk_scene())
    rule = PedestrianConflictRule(w.scene, CAT)
    pos = {"y": 2.0, "v": 12.0}

    def frames(i):
        if i > 0:
            pos["v"] = max(0.0, pos["v"] - 6.0 / 30)
            pos["y"] += pos["v"] / 30
        return {1: ("person", box_at(4 + i * 0.03, 16.5, 20, 50)), 2: ("car", box_at(4.5, pos["y"], 60, 50))}

    ev = [e for e in run(rule, w, frames, n=45) if e.variant != "pet"]
    assert ev == []


# ----------------------------------------------------------------- RB-08
def test_abrupt_braking():
    w = World()
    rule = FollowingRule(w.scene, CAT)
    st = {"y": 1.0, "v": 14.0}

    def frames(i):
        if i > 0:
            if i > 20:
                st["v"] = max(0.0, st["v"] - 7.0 / 30)
            st["y"] += st["v"] / 30
        return {1: ("car", box_at(5, st["y"], 60, 50))}

    ev = run(rule, w, frames, n=60)
    assert any(e.variant == "phanh_gap" for e in ev)


def test_tailgating():
    w = World()
    rule = FollowingRule(w.scene, CAT)
    v = 40 / 3.6
    ev = run(rule, w, lambda i: {1: ("car", box_at(5, 1 + v * i / 30, 60, 50)),
                                 2: ("car", box_at(5, 1 + 5.0 + v * i / 30, 60, 50))}, n=90)
    tg = [e for e in ev if e.variant == "bam_duoi"]
    assert tg and tg[0].primary_track_id == 1


# ----------------------------------------------------------------- RB-09 / RB-02
def test_overload_and_phone():
    w = World()
    ov = OverloadRule(w.scene, CAT)
    ph = PhoneUseRule(w.scene, CAT)
    cargo = deque()
    phone = deque()

    def frames(i):
        objs, riders = moto_and_riders(2 + i * 0.25, 1)
        t = i / 30
        cargo.append((t, 2.1, 1.6, "bgsub"))
        phone.append((t, True, False, 0.7))
        return objs, {"riders": {1: riders}, "attrs": {1: {"cargo_obs": cargo}, 10: {"phone_obs": phone}}}

    events = []
    for i in range(60):
        objs, extra = frames(i)
        ctx = w.step(objs, **extra)
        events += ov.evaluate(ctx) + ph.evaluate(ctx)
    codes = sorted(e.rule_code for e in events)
    assert codes == ["RB-02", "RB-09"]
    assert next(e for e in events if e.rule_code == "RB-02").severity == "nghi_van"


# ----------------------------------------------------------------- RB-11 / RB-12 / HZ-04
def test_parking_on_crosswalk():
    scene = _crosswalk_scene()
    scene.zones.no_parking.append(ZoneCfg(id="NP1", name="Vạch trước cổng", kind="crosswalk",
                                          polygon=[(0, 300), (200, 300), (200, 360), (0, 360)]))
    w = World(scene, fps=10)
    rule = ParkingRule(w.scene, CAT)
    ev = run(rule, w, lambda i: {1: ("car", box_at(5, 17.0, 60, 50))}, n=130)
    assert len(ev) == 1 and ev[0].variant == "crosswalk"
    assert ev[0].measurements["thoi_gian_dung_s"] >= 10


def test_abreast_three_bicycles():
    w = World()
    rule = AbreastRule(w.scene, CAT)
    v = 12 / 3.6
    ev = run(rule, w, lambda i: {k: ("bicycle", box_at(2 + k * 1.0, 1 + v * i / 30, 20, 40)) for k in (1, 2, 3)}, n=100)
    assert len(ev) == 1 and ev[0].measurements["so_xe_hang_ngang"] == 3 and ev[0].vehicle_cls == "bicycle"
    w2 = World()
    rule2 = AbreastRule(w2.scene, CAT)
    ev2 = run(rule2, w2, lambda i: {k: ("bicycle", box_at(2 + k * 1.0, 1 + v * i / 30, 20, 40)) for k in (1, 2)}, n=100)
    assert ev2 == []


def test_jaywalk_outside_crosswalk():
    scene = _crosswalk_scene()
    scene.zones.roadway.append(ZoneCfg(id="R", polygon=[(0, 0), (200, 0), (200, 600), (0, 600)]))
    w = World(scene)
    rule = JaywalkRule(w.scene, CAT)
    ev = run(rule, w, lambda i: {1: ("person", box_at(1 + i * 0.05, 8, 20, 50)),          # băng ngang ngoài vạch
                                 2: ("person", box_at(1 + i * 0.05, 16.5, 20, 50))}, n=80)  # đi trên vạch
    assert [e.track_ids[0] for e in ev] == [1]
    assert ev[0].severity == "nguy_co"


# ----------------------------------------------------------------- POV
def test_sudden_emergence_pov():
    scene = SiteConfig()
    w = World(scene, calib=False, channel="pov")
    rule = SuddenEmergenceRule(w.scene, CAT)
    truck = np.array([700, 250, 1000, 650.0])

    def frames(i):
        objs = {1: ("truck", truck)}
        if i >= 10:
            k = i - 10
            x = 700 - 20 - k * 12
            objs[2] = ("person", np.array([x - 30, 470, x, 620.0]))
        return objs

    ev = run(rule, w, frames, n=16)
    assert len(ev) == 1 and ev[0].rule_code == "HZ-03" and ev[0].track_ids[0] == 2


def test_turn_without_signal_pov():
    w = World(SiteConfig(), calib=False, channel="pov")
    rule = TurnNoSignalRule(w.scene, CAT)
    ego = EgoMotion(np.array([[1.0, 0, 0], [0, 1, 0]]), horizon_y=324)

    def frames(i):
        x = 600 + (0 if i < 40 else (i - 40) * 4)
        return {1: ("car", np.array([x, 400, x + 120, 480.0])),
                2: ("car", np.array([300 + (0 if i < 40 else (i - 40) * 4), 380, 420 + (0 if i < 40 else (i - 40) * 4), 460.0]))}, \
            {"ego": ego, "attrs": {2: {"blink": {"right": i / 30.0}}}}

    ev = run(rule, w, frames, n=70)
    assert [e.primary_track_id for e in ev] == [1]


# ----------------------------------------------------------------- registry / catalog
def test_registry_reports_reasons():
    scene = SiteConfig()
    rules, report = build_rules(scene, CAT, {"vehicles", "persons"})
    st = {r.code: r for r in report}
    assert not st["RB-01"].enabled and "mũ bảo hiểm" in st["RB-01"].reason
    assert not st["RB-10"].enabled
    assert st["RB-06"].enabled
    assert not st["HZ-01"].enabled and "POV" in st["HZ-01"].reason
    assert len(report) == 16


def test_catalog_lookup():
    r = CAT.lookup("RB-05", "car")
    assert r.ref.startswith("Điểm b khoản 9 Điều 6") and "18.000.000" in r.fine
    assert CAT.lookup("RB-06", "motorcycle", "4_nguoi_tro_len").ref.startswith("Điểm b khoản 3 Điều 7")
    assert CAT.lookup("RB-10", "car", "car_35_plus").fine.startswith("12.000.000")
    assert CAT.lookup("RB-03", "bicycle").verified is False
    assert PX_PER_M == 20.0
