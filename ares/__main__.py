"""Cổng vào dòng lệnh:  python -m ares <lệnh>

    run        Chạy hệ thống thật (camera + MQTT + ESP32 + dashboard).
    demo       Chạy MÔ PHỎNG theo thời gian thực (đồng hồ chạy nhanh) kèm dashboard — không cần phần cứng.
    simulate   Mô phỏng nhanh không đầu, in báo cáo tiết kiệm điện/tiện nghi so với baseline.
    check      Kiểm tra cấu hình, vùng, lịch và thư viện (không mở camera/MQTT).
    zones      Vẽ các vùng lên một ảnh chụp để căn lại đa giác.
    retrain    Chạy retrain ngay (cần ảnh đã duyệt), không mở camera.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _utf8() -> None:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")  # console Windows mặc định cp1252 sẽ làm hỏng tiếng Việt
        except Exception:
            pass


def _logging(cfg, to_file: bool = True) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(cfg.log_level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if to_file:
        logd = cfg.storage_root / "logs"
        logd.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(logd / "ares.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(fh)
    for noisy in ("httpx", "uvicorn.access", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _settings(**over):
    from ares.config import AresSettings

    cfg = AresSettings(**{k: v for k, v in over.items() if v is not None})
    cfg.ensure_dirs()
    return cfg


def cmd_run(a) -> int:
    from ares.runtime import AresRuntime

    cfg = _settings(simulation_mode=True if a.sim else None, dashboard_host=a.host, dashboard_port=a.port)
    _logging(cfg)
    AresRuntime(cfg, with_dashboard=not a.no_dashboard).run_forever()
    return 0


def cmd_demo(a) -> int:
    from ares.runtime import AresRuntime

    start = a.start
    if not start:
        d = datetime.now()
        d = d + timedelta(days=(7 - d.weekday()) % 7 or 0)  # thứ Hai gần nhất (hôm nay nếu đang là thứ Hai)
        start = d.replace(hour=6, minute=15, second=0, microsecond=0).isoformat()
    cfg = _settings(simulation_mode=True, sim_speed=a.speed, sim_start=start, dashboard_host=a.host, dashboard_port=a.port,
                    sim_seed=a.seed, sim_cameras=a.cams)
    _logging(cfg)
    print(f"\nDEMO mô phỏng bắt đầu {start}, tốc độ ×{a.speed:g}. Mở http://localhost:{cfg.dashboard_port}  (Ctrl+C để dừng)\n"
          f"Dữ liệu demo nằm riêng ở {cfg.storage_root} — không lẫn vào dữ liệu thật.\n")
    AresRuntime(cfg).run_forever()
    return 0


def cmd_simulate(a) -> int:
    from ares.simrun import default_cfg_for_sim, format_report, run_headless

    over = {"sim_seed": a.seed}
    cfg = default_cfg_for_sim(Path(a.storage) if a.storage else None, **over)
    start = datetime.fromisoformat(a.start)
    last = [-1]

    def prog(f: float) -> None:
        p = int(f * 100)
        if p != last[0] and p % 5 == 0:
            last[0] = p
            print(f"\r  tiến độ {p:3d}%", end="", file=sys.stderr, flush=True)

    rep = run_headless(cfg, start, days=a.days, step_s=a.step, baseline=not a.no_baseline,
                       out_dir=Path(a.out) if a.out else None, progress=prog)
    print("", file=sys.stderr)
    rt = rep.pop("_runtime", None)
    if rt is not None:
        rt.stop()  # đóng luồng Active Learning và DB để tiến trình thoát gọn
    print(format_report(rep))
    if a.out:
        print(f"Đã ghi báo cáo vào {a.out}/ (bao_cao.txt, bao_cao.json, chuoi_thoi_gian.csv)")
    return 0


def cmd_check(a) -> int:
    import importlib

    from ares.config import load_room
    from ares.schedule import StaticSchedule

    cfg = _settings()
    ok = True

    def line(good: bool | None, msg: str) -> None:
        nonlocal ok
        mark = {True: "[OK] ", False: "[LỖI]", None: "[ ! ] "}[good]
        if good is False:
            ok = False
        print(f"{mark} {msg}")

    print(f"Phòng {cfg.room_id} · múi giờ {cfg.timezone} · lưu dữ liệu tại {cfg.storage_root}")
    for mod, why in (("ultralytics", "YOLO"), ("cv2", "OpenCV"), ("paho.mqtt.client", "MQTT"), ("fastapi", "dashboard"),
                     ("uvicorn", "dashboard"), ("sqlalchemy", "DB")):
        try:
            importlib.import_module(mod)
            line(True, f"thư viện {mod} ({why})")
        except Exception as e:  # noqa: BLE001
            line(False, f"thiếu thư viện {mod} ({why}): {e}")
    try:
        import paho.mqtt.client as m
        line(hasattr(m, "CallbackAPIVersion") or False, "paho-mqtt >= 2.0")
    except Exception:
        pass
    try:
        import torch
        line(None if not torch.cuda.is_available() else True,
             "CUDA sẵn sàng" if torch.cuda.is_available() else "KHÔNG có CUDA — YOLO sẽ chạy bằng CPU (chậm, nên giảm imgsz/tần suất)")
    except Exception:
        line(None, "chưa cài torch")
    try:
        room = load_room(cfg.zones_file, cfg.room_id)
        line(True, f"zones.json: {room.n_zones} vùng, n_max={room.n_max}, k={[z.k for z in room.zones]}")
        if len(cfg.led_watts) != room.n_zones:
            line(None, f"led_watts có {len(cfg.led_watts)} giá trị nhưng có {room.n_zones} vùng (chỉ dùng cho ước tính điện năng)")
    except Exception as e:  # noqa: BLE001
        line(False, f"zones.json: {e}")
    try:
        sch = StaticSchedule.from_file(cfg.schedule_file, cfg.tz)
        n = len(sch.rules)
        line(True if n else None, f"schedule.json: {n} khung giờ" if n else "schedule.json không có buổi nào — chỉ chạy theo hiện diện")
    except Exception as e:  # noqa: BLE001
        line(False, f"schedule.json: {e}")
    w = cfg.weights_path()
    line(True if w.exists() else None, f"weights {w}" + ("" if w.exists() else " chưa có — ultralytics sẽ tự tải nếu là tên chuẩn (vd yolov8n.pt)"))
    host_open = cfg.dashboard_host not in ("127.0.0.1", "localhost", "::1")
    line(False if host_open and not cfg.dashboard_token else True,
         f"dashboard {cfg.dashboard_host}:{cfg.dashboard_port}" + (" (mở ra mạng, có token)" if host_open and cfg.dashboard_token else ""))
    print("\nKết quả:", "SẴN SÀNG" if ok else "CÓ LỖI CẦN SỬA")
    return 0 if ok else 1


def cmd_zones(a) -> int:
    import cv2

    from ares.config import load_room
    from ares.zones import ZoneMap

    cfg = _settings()
    zm = ZoneMap(load_room(cfg.zones_file, cfg.room_id))
    img = cv2.imdecode(__import__("numpy").fromfile(a.snapshot, dtype="uint8"), cv2.IMREAD_COLOR)  # đường dẫn Unicode trên Windows
    if img is None:
        print(f"Không đọc được ảnh {a.snapshot}")
        return 1
    zm.draw(img)
    out = a.out or str(Path(a.snapshot).with_name(Path(a.snapshot).stem + "_zones.jpg"))
    ok, buf = cv2.imencode(".jpg", img)
    Path(out).write_bytes(buf.tobytes())
    print(f"Đã vẽ {zm.n} vùng lên ảnh -> {out}")
    return 0


def cmd_retrain(a) -> int:
    from ares.active_learner import ActiveLearner
    from ares.clock import RealClock
    from ares.db_manager import DBManager
    from ares.model_registry import ModelRegistry
    from ares.retrain import Retrainer

    cfg = _settings()
    _logging(cfg)
    clock = RealClock(cfg.tz)
    db = DBManager(cfg.database_url, cfg.room_id, clock)
    al = ActiveLearner(cfg, clock, db)
    rt = Retrainer(cfg, clock, al, ModelRegistry(cfg.models_dir, cfg.weights_path()),
                   emit=lambda e: print(f"[{e.kind}] {e.message}"))
    res = rt.run("manual", force=True)
    db.close()
    print(res)
    return 0 if res.get("status") in ("done", "skipped") else 1


def main(argv: list[str] | None = None) -> int:
    _utf8()
    ap = argparse.ArgumentParser(prog="python -m ares", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="chạy hệ thống thật")
    p.add_argument("--sim", action="store_true", help="bật SimulationMode (tương đương ARES_SIMULATION_MODE=true)")
    p.add_argument("--host"), p.add_argument("--port", type=int)
    p.add_argument("--no-dashboard", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("demo", help="mô phỏng thời gian thực + dashboard")
    p.add_argument("--speed", type=float, default=30.0, help="đồng hồ chạy nhanh gấp N lần (mặc định 30: 1 giây = 30 giây)")
    p.add_argument("--start", help="giờ bắt đầu ISO, vd 2026-11-02T06:15 (mặc định: thứ Hai gần nhất 06:15)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cams", type=int, default=4, help="số camera giả để xem thử lưới CCTV (1-36, mặc định 4)")
    p.add_argument("--host"), p.add_argument("--port", type=int)
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("simulate", help="mô phỏng nhanh không đầu + báo cáo")
    p.add_argument("--days", type=float, default=1.0)
    p.add_argument("--start", default="2026-11-02T05:30", help="ISO; chọn thứ Hai để có đủ tuần học")
    p.add_argument("--step", type=float, default=5.0, help="bước vision (giây); spec: 1-5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", help="thư mục ghi bao_cao.txt/json + chuoi_thoi_gian.csv")
    p.add_argument("--storage", help="thư mục lưu DB/ảnh của lần chạy (mặc định: thư mục tạm)")
    p.add_argument("--no-baseline", action="store_true")
    p.set_defaults(fn=cmd_simulate)

    sub.add_parser("check", help="kiểm tra cấu hình").set_defaults(fn=cmd_check)
    p = sub.add_parser("zones", help="vẽ vùng lên ảnh chụp")
    p.add_argument("--snapshot", required=True), p.add_argument("--out")
    p.set_defaults(fn=cmd_zones)
    sub.add_parser("retrain", help="retrain ngay").set_defaults(fn=cmd_retrain)

    a = ap.parse_args(argv)
    try:
        return int(a.fn(a))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
