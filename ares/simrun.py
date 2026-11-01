"""Chạy mô phỏng KHÔNG ĐẦU (headless): quay thời gian bằng ManualClock, không luồng, xác định theo seed.

Dùng chính `AresRuntime` (cùng dây nối với bản chạy thật), chỉ khác là các vòng tick được gọi tay theo
đồng hồ giả. Mỗi ngày mô phỏng chạy trong vài chục giây thay vì 24 giờ.

So sánh với một "baseline giả định" (đèn 100 % và điều hòa 25 °C suốt khung giờ có lớp +-30 phút, không cần
cảm biến) trên CÙNG kịch bản người, CÙNG thời tiết. Đây là mô hình để kiểm tra logic, KHÔNG phải số đo thực địa.

    python -m ares simulate --days 3 --start 2026-11-02T05:30 --out sim_out
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

from ares.clock import ManualClock
from ares.config import AresSettings
from ares.runtime import AresRuntime
from ares.simulation import SimMetrics, SimWorld

log = logging.getLogger(__name__)


class BaselineController:
    """Vận hành thủ công điển hình: bật hết đèn + điều hòa 25 °C trong khung giờ có lớp (-30/+30 phút)."""

    def __init__(self, world: SimWorld, schedule, tz, lead_min: float = 30.0, tail_min: float = 30.0):
        self.world, self.schedule, self.tz = world, schedule, tz
        self.lead, self.tail = lead_min, tail_min
        self.on: bool | None = None

    def _active(self, now: datetime) -> bool:
        ss = self.schedule.sessions_on(now.date())
        if not ss:
            return False
        a = ss[0].start - timedelta(minutes=self.lead)
        b = ss[-1].end + timedelta(minutes=self.tail)
        return a <= now < b

    def step(self, now: datetime) -> None:
        want = self._active(now)
        if want == self.on:
            return
        self.on = want
        n = self.world.n
        self.world.apply_led([100.0 if want else 0.0] * n)
        self.world.apply_ac({"power": 1 if want else 0, "temp": 25, "fan": 2})


@dataclass
class SessionProbe:
    key: str
    name: str
    start: str
    t_in_ares: float | None = None
    t_in_base: float | None = None
    people_true: int | None = None
    people_est: float | None = None


def _metrics_dict(m: SimMetrics) -> dict:
    d = asdict(m)
    d["led_kwh"] = round(m.led_wh / 1000.0, 3)
    d["ac_kwh"] = round(m.ac_wh / 1000.0, 3)
    d["total_kwh"] = round((m.led_wh + m.ac_wh) / 1000.0, 3)
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def run_headless(cfg: AresSettings, start: datetime, days: float = 1.0, *, step_s: float | None = None,
                 baseline: bool = True, out_dir: Path | None = None, progress: Callable[[float], None] | None = None,
                 schedule=None, scenario=None, quiet_logs: bool = True) -> dict:
    """Chạy `days` ngày mô phỏng từ `start`. Trả về báo cáo (dict, JSON được)."""
    step = float(step_s or cfg.vision_interval_s)
    cfg.simulation_mode = True
    cfg.retrain_enabled = False  # không train YOLO trong mô phỏng (ảnh tổng hợp)
    cfg.overlay_interval_s = max(cfg.overlay_interval_s, 900.0)  # không dựng ảnh minh hoạ liên tục
    if quiet_logs:
        logging.getLogger("ares").setLevel(logging.WARNING)

    clock = ManualClock(start, cfg.tz)
    rt = AresRuntime(cfg, clock=clock, schedule=schedule, scenario=scenario, with_dashboard=False)
    world, scen, engine = rt.world, rt.scenario, rt.engine
    rt.al.start()

    base_world = base_ctl = None
    if baseline:
        base_world = SimWorld(cfg, clock, rt.zonemap, scen, seed=cfg.sim_seed)
        base_ctl = BaselineController(base_world, rt.schedule, cfg.tz)

    t0 = clock.time()
    t_end = t0 + days * 86400.0
    nxt_light = nxt_hvac = nxt_tel = nxt_series = t0
    series: list[dict] = []
    err_abs: list[float] = []
    probes: dict[str, SessionProbe] = {}
    ac_cmds = 0
    led_msgs = 0
    pwm_occ_sum = pwm_occ_n = 0

    while clock.time() < t_end:
        clock.advance(step)
        t = clock.time()
        now = clock.now()
        world.advance(t)
        if base_world is not None:
            base_ctl.step(now)
            base_world.advance(t)
        rt.tracker.tick()
        if t >= nxt_light:
            nxt_light = t + cfg.lighting_interval_s
            engine.lighting_tick()
        if t >= nxt_hvac:
            nxt_hvac = t + cfg.hvac_tick_s
            engine.hvac_tick()
        if t >= nxt_tel:
            nxt_tel = t + cfg.telemetry_interval_s
            rt.log_telemetry()
            rt._housekeep()
            rt.nightly.tick()  # tự chạy khi qua nửa đêm: gom thói quen hôm đó cho pre-cooling hôm sau
        if rt.al.q.qsize() >= max(1, cfg.al_queue_size // 2):
            rt.al.drain(30)

        # ---- thu số liệu ----------------------------------------------------------------------
        people = scen.people_at(now)
        n_true = len(people)
        vis = rt.state.get_vision()
        if vis is not None and n_true + vis.n_total > 0:
            sm = rt.state.vision_smoothed.get("mean_total")
            if sm is not None:
                err_abs.append(abs(sm - n_true))
        if n_true:
            pwm_occ_sum += sum(world.pwm) / max(1, len(world.pwm))
            pwm_occ_n += 1

        for s in rt.schedule.sessions_on(now.date()):
            if s.key not in probes and now >= s.start:
                est = rt.state.vision_smoothed.get("mean_total")
                probes[s.key] = SessionProbe(
                    s.key, s.name, s.start.isoformat(timespec="minutes"),
                    t_in_ares=round(world.t_in, 2),
                    t_in_base=None if base_world is None else round(base_world.t_in, 2),
                    people_true=n_true, people_est=None if est is None else round(est, 1))

        if t >= nxt_series:
            nxt_series = t + 300.0
            ls = rt.state.get_lighting()
            series.append({
                "time": now.isoformat(timespec="minutes"), "people_true": n_true,
                "people_est": round(rt.state.vision_smoothed.get("mean_total") or 0.0, 2),
                "t_in": round(world.t_in, 2), "t_out": round(world.t_out(now), 1),
                "t_in_base": None if base_world is None else round(base_world.t_in, 2),
                "ac_on": world.ac["power"], "ac_set": world.ac["temp"],
                "pwm_mean": round(sum(world.pwm) / len(world.pwm), 1),
                "lux_desk_mean": round(sum(world.last_desk) / len(world.last_desk)),
                "daylight_lux_mean": round(sum(world.daylight_desk(i, now) for i in range(world.n)) / world.n),
                "mode": rt.state.hvac.mode,
            })
        if progress is not None:
            progress((t - t0) / (t_end - t0))

    rt.al.drain(60)
    rt.db.flush(30)

    # ---- tổng hợp ------------------------------------------------------------------------------
    ev_counts = Counter(e["kind"] for e in rt.db.recent_events(limit=100000))
    ac_cmds = ev_counts.get("AC_ON_OCCUPANCY", 0) + ev_counts.get("AC_SETPOINT", 0) + ev_counts.get("PRECOOL_START", 0) + ev_counts.get("PRECOOL_KEEP_ON", 0) \
        + ev_counts.get("FAILSAFE_EMPTY_CLASS", 0) + ev_counts.get("AC_OFF_EMPTY", 0)
    ares_m = _metrics_dict(world.metrics)
    rep: dict = {
        "meta": {
            "start": start.isoformat(), "days": days, "step_s": step, "seed": cfg.sim_seed, "room": cfg.room_id,
            "ghi_chu": "Kết quả từ mô hình vật lý giả định để kiểm tra logic, KHÔNG phải số đo thực tế.",
            "canh_bao": ([f"Bước mô phỏng {step:g} giây thô hơn dải vision 1-5 giây của spec: mỗi lần nhận nhầm của detector giả "
                          "bị giữ nguyên cả bước nên đèn/điều hòa nhấp nháy nhiều hơn thực tế. Dùng --step 5 trở xuống."]
                         if step > 5.0 else []),
        },
        "ares": ares_m,
        "events": dict(sorted(ev_counts.items())),
        "ac_commands_estimated": ac_cmds,
        "vision": {
            "mae_people": round(statistics.fmean(err_abs), 2) if err_abs else None,
            "ghi_chu": "Sai số tuyệt đối trung bình giữa số người thật và số người làm mượt 30 giây (detector giả bỏ sót ~10-20%, dương tính giả ~12%/khung).",
        },
        "active_learning": rt.al.stats(),
        "sessions": [asdict(p) for p in probes.values()],
        "habits_learned": len(rt.db.habit_matrix()),
        "model_calls": {"led_msgs": len(world.led_log), "ac_msgs": len(world.ac_log)},
    }
    if base_world is not None:
        bm = _metrics_dict(base_world.metrics)
        rep["baseline"] = bm
        tot_a, tot_b = ares_m["total_kwh"], bm["total_kwh"]
        rep["so_sanh"] = {
            "tiet_kiem_den_pct": _pct(bm["led_kwh"], ares_m["led_kwh"]),
            "tiet_kiem_dieu_hoa_pct": _pct(bm["ac_kwh"], ares_m["ac_kwh"]),
            "tiet_kiem_tong_pct": _pct(tot_b, tot_a),
            "baseline": "Đèn 100 % + điều hòa 25 °C từ 30 phút trước buổi đầu đến 30 phút sau buổi cuối mỗi ngày có lớp.",
        }
    comfort_ok = [p for p in probes.values() if p.t_in_ares is not None and p.people_true]
    if comfort_ok:
        rep["nhiet_do_luc_vao_lop"] = {
            "ares_tb": round(statistics.fmean(p.t_in_ares for p in comfort_ok), 2),
            "ares_max": max(p.t_in_ares for p in comfort_ok),
            "baseline_tb": round(statistics.fmean(p.t_in_base for p in comfort_ok if p.t_in_base is not None), 2) if base_world else None,
        }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rep["nhat_ky_su_kien"] = rt.db.recent_events(limit=2000)[::-1]
        (out_dir / "bao_cao.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        if series:
            with open(out_dir / "chuoi_thoi_gian.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(series[0].keys()))
                w.writeheader()
                w.writerows(series)
        (out_dir / "bao_cao.txt").write_text(format_report(rep), encoding="utf-8")
    rep["_series"] = series
    rep["_runtime"] = rt  # để test/đoạn mã gọi tự đóng
    return rep


def _pct(base: float, ours: float) -> float | None:
    return None if base <= 0 else round((1 - ours / base) * 100, 1)


def format_report(rep: dict) -> str:
    m = rep["meta"]
    a = rep["ares"]
    L = [
        "=" * 70,
        f"BÁO CÁO MÔ PHỎNG ARES — phòng {m['room']}, {m['days']:g} ngày từ {m['start']}",
        "=" * 70,
        "LƯU Ý: " + m["ghi_chu"],
        *[f"CẢNH BÁO: {w}" for w in m.get("canh_bao", [])],
        "",
        "Điện năng (kWh, mô hình): ",
        f"  ARES      đèn {a['led_kwh']:.2f} | điều hòa {a['ac_kwh']:.2f} | tổng {a['total_kwh']:.2f}",
    ]
    if "baseline" in rep:
        b = rep["baseline"]
        s = rep["so_sanh"]
        L += [
            f"  Baseline  đèn {b['led_kwh']:.2f} | điều hòa {b['ac_kwh']:.2f} | tổng {b['total_kwh']:.2f}",
            f"  Tiết kiệm đèn {s['tiet_kiem_den_pct']}% | điều hòa {s['tiet_kiem_dieu_hoa_pct']}% | tổng {s['tiet_kiem_tong_pct']}%",
            f"  (baseline = {s['baseline']})",
        ]
    L += [
        "",
        "Tiện nghi khi có người:",
        f"  ARES: nóng >27 °C {a['hot_min']:.0f} phút, >26.5 °C {a['warm_min']:.0f} phút; "
        f"vùng có người mà <300 lux: {a['dark_zone_min']:.0f} phút-vùng (trong {a['occupied_min']:.0f} phút có người)",
    ]
    if "baseline" in rep:
        b = rep["baseline"]
        L.append(f"  Baseline: nóng >27 °C {b['hot_min']:.0f} phút, >26.5 °C {b['warm_min']:.0f} phút; "
                 f"<300 lux: {b['dark_zone_min']:.0f} phút-vùng")
    if "nhiet_do_luc_vao_lop" in rep:
        n = rep["nhiet_do_luc_vao_lop"]
        L.append(f"  Nhiệt độ phòng lúc vào lớp: ARES TB {n['ares_tb']} °C (max {n['ares_max']}), baseline TB {n['baseline_tb']} °C")
    L += ["", f"Đếm người: sai số tuyệt đối TB {rep['vision']['mae_people']} người", "", "Sự kiện:"]
    L += [f"  {k}: {v}" for k, v in rep["events"].items()]
    al = rep["active_learning"]
    L += ["", f"Học chủ động: {al['counts']} | hôm nay {al['today']} | bỏ qua {al['skipped']}"]
    return "\n".join(L) + "\n"


def default_cfg_for_sim(storage: Path | None = None, **over) -> AresSettings:
    root = Path(storage) if storage else Path(tempfile.mkdtemp(prefix="ares_sim_"))
    cfg = AresSettings(_env_file=None, storage_root=root, simulation_mode=True, **over)
    cfg.ensure_dirs()
    return cfg
