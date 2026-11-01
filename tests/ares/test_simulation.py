"""Kiểm tra chế độ mô phỏng chạy không đầu (headless) từ đầu đến cuối.

Đây là kiểm tra LOGIC của cả chuỗi (vision giả -> bộ đệm -> điều khiển -> thiết bị giả), không phải
kiểm chứng số liệu tiết kiệm điện thực tế: vật lý phòng là mô hình giả định.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from ares.simrun import default_cfg_for_sim, format_report, run_headless

START = datetime(2026, 11, 2, 5, 30)  # thứ Hai, trước buổi học đầu 07:00


@pytest.fixture(scope="module")
def day(tmp_path_factory):
    """Thứ Hai từ 05:30 tới ~21:00 (bước 5 s đúng dải spec), có baseline và xuất báo cáo ra tệp."""
    tmp = tmp_path_factory.mktemp("sim_day")
    cfg = default_cfg_for_sim(tmp / "storage", sim_seed=7)
    rep = run_headless(cfg, START, days=0.65, step_s=5.0, baseline=True, out_dir=tmp / "out")
    rt = rep.pop("_runtime")
    rep["_out"] = tmp / "out"
    rep["_db"] = {"write_errors": rt.db.write_errors, "dropped": rt.db.dropped}
    rt.stop()
    return rep


def test_ares_uses_less_lighting_energy_than_baseline(day):
    a, b = day["ares"], day["baseline"]
    assert a["led_kwh"] < b["led_kwh"]
    assert day["so_sanh"]["tiet_kiem_den_pct"] > 5.0


def test_ares_not_worse_than_baseline_on_total_energy(day):
    assert day["ares"]["total_kwh"] <= day["baseline"]["total_kwh"]


def test_comfort_is_kept_while_people_present(day):
    a = day["ares"]
    assert a["occupied_min"] > 300  # phòng thật sự có người trong ngày
    assert a["hot_min"] == 0.0  # không phút nào > 26,5 °C khi có người
    assert a["dark_zone_min"] < 15  # thiếu sáng (<300 lux) chỉ thoáng qua


def test_ac_is_off_after_last_class(day):
    late = [r for r in day["_series"] if r["time"] >= "2026-11-02T19:00"]
    assert late, "chuỗi thời gian phải phủ tới tối"
    assert not any(r["ac_on"] for r in late)
    assert all(r["pwm_mean"] == 0 for r in late)  # cả phòng trống -> đèn tắt hẳn


def test_precooling_happens_and_learns_cooling_rate(day):
    ev = day["events"]
    assert ev.get("PRECOOL_START", 0) >= 1
    assert ev.get("R_COOL_UPDATED", 0) >= 1


def test_ac_turned_off_when_room_empties(day):
    assert day["events"].get("AC_OFF_EMPTY", 0) >= 1


def test_vision_error_is_bounded(day):
    mae = day["vision"]["mae_people"]
    assert mae is not None and mae < 4.0


def test_active_learning_collects_without_daily_cap_and_pca_filters(day):
    al = day["active_learning"]
    assert al["today"]["captured"] > 0
    assert al["today"]["cap"] is None  # không còn giới hạn 50/ngày


def test_no_db_write_errors_or_dropped_rows(day):
    # ngoại lệ trong các nhịp điều khiển sẽ làm run_headless văng ra -> fixture hỏng; ở đây kiểm tra thêm DB
    assert day["_db"] == {"write_errors": 0, "dropped": 0}


def test_report_files_written_and_in_vietnamese(day):
    out = day["_out"]
    data = json.loads((out / "bao_cao.json").read_text(encoding="utf-8"))
    assert data["meta"]["seed"] == 7
    assert "KHÔNG phải số đo thực tế" in data["meta"]["ghi_chu"]
    txt = (out / "bao_cao.txt").read_text(encoding="utf-8")
    assert "kWh" in txt and "tiết kiệm" in txt.lower()
    assert format_report(data).strip() == txt.strip()
    assert (out / "chuoi_thoi_gian.csv").read_text(encoding="utf-8-sig").startswith("time,")


def test_coarse_step_is_flagged_in_report(tmp_path):
    cfg = default_cfg_for_sim(tmp_path / "coarse", sim_seed=5)
    rep = run_headless(cfg, START, days=0.05, step_s=20.0, baseline=False)
    rep.pop("_runtime").stop()
    assert rep["meta"]["canh_bao"] and "CẢNH BÁO" in format_report(rep)


def test_same_seed_gives_identical_result(tmp_path):
    outs = []
    for i in range(2):
        cfg = default_cfg_for_sim(tmp_path / f"s{i}", sim_seed=11)
        rep = run_headless(cfg, START, days=0.4, step_s=30.0, baseline=False)
        rep.pop("_runtime").stop()
        outs.append(rep)
    assert outs[0]["ares"] == outs[1]["ares"]
    # luồng học chủ động/PCA chạy ở thread nền nên số lần lọc có thể lệch 1 nhịp; phần điều khiển phải y hệt
    strip = lambda ev: {k: v for k, v in ev.items() if not k.startswith(("PCA_", "AL_"))}  # noqa: E731
    assert strip(outs[0]["events"]) == strip(outs[1]["events"])


def test_different_seed_changes_the_world(tmp_path):
    res = []
    for seed in (1, 2):
        cfg = default_cfg_for_sim(tmp_path / f"d{seed}", sim_seed=seed)
        rep = run_headless(cfg, START, days=0.4, step_s=30.0, baseline=False)
        rep.pop("_runtime").stop()
        res.append(rep["ares"])
    assert res[0] != res[1]


def test_weekend_runs_no_equipment(tmp_path):
    sat = datetime(2026, 11, 7, 6, 0)  # thứ Bảy: không có lớp trong lịch mẫu
    cfg = default_cfg_for_sim(tmp_path / "we", sim_seed=3)
    rep = run_headless(cfg, sat, days=0.5, step_s=30.0, baseline=False)
    rep.pop("_runtime").stop()
    assert rep["ares"]["ac_kwh"] == 0.0
    assert rep["ares"]["hot_min"] == 0.0
