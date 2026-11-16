"""Dòng lệnh: python -m traffic_safety_ai <lệnh> [tuỳ chọn]

  check          kiểm tra môi trường, model, cấu hình → bảng tình huống BẬT/TẮT kèm lý do
  demo           chạy mô phỏng (không cần camera/model) → hồ sơ mẫu, báo cáo, video
  run            chạy trên video / thư mục video / RTSP / webcam
  calibrate      cửa sổ khai báo vùng, vạch, đèn, hiệu chuẩn → YAML
  preview        vẽ cấu hình lên một khung hình để kiểm tra (không cần màn hình)
  dashboard      trang quản trị: duyệt hồ sơ, tra cứu biển số, thống kê
  report         lập báo cáo tổng hợp HTML + CSV
  lookup         tra cứu hồ sơ theo biển số
  export-models  xuất model sang ONNX / TensorRT
  train          huấn luyện model bổ sung (mũ bảo hiểm, biển số, ký tự...)
  cleanup        xoá hồ sơ quá hạn lưu trữ (privacy.retention_days)
"""

from __future__ import annotations

import argparse
import logging
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from traffic_safety_ai import CONFIG_DIR, PACKAGE_ROOT, __version__

log = logging.getLogger("traffic_safety_ai")


def _paths(args: argparse.Namespace):
    from traffic_safety_ai.core.config import AppPaths

    return AppPaths(getattr(args, "data_dir", None)).ensure()


def _scene_path(p: str | None, channel_hint: str = "cctv") -> Path:
    if p:
        q = Path(p)
        if not q.exists() and (CONFIG_DIR / p).exists():
            q = CONFIG_DIR / p
        return q
    return CONFIG_DIR / ("pov_config.yaml" if channel_hint == "pov" else "cctv_config.yaml")


def _parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise SystemExit(f"Không hiểu thời gian '{s}'. Dạng hợp lệ: 2026-11-04 07:00:00")


# ====================================================================== check
def cmd_check(args: argparse.Namespace) -> int:
    from traffic_safety_ai.core.config import load_models_config, load_site_config
    from traffic_safety_ai.core.detector import ModelHub
    from traffic_safety_ai.rules.catalog import LegalCatalog
    from traffic_safety_ai.rules.registry import build_rules, format_status_table, needed_models

    print(f"traffic_safety_ai {__version__} — Python {platform.python_version()} ({platform.system()})")
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"torch {torch.__version__} — CUDA: {'có' if cuda else 'KHÔNG'}"
              + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""))
    except Exception as e:
        print(f"torch: chưa cài ({e})")
    try:
        import ultralytics

        print(f"ultralytics {ultralytics.__version__}")
    except Exception as e:
        print(f"ultralytics: chưa cài ({e})")
    import cv2

    print(f"opencv {cv2.__version__}; ffmpeg: {'có' if shutil.which('ffmpeg') else 'không (không bắt buộc)'}")
    scene = load_site_config(_scene_path(args.scene))
    mcfg = load_models_config(args.models)
    print(f"\nCấu hình địa điểm: {_scene_path(args.scene)} — camera {scene.camera.id} ({scene.camera.channel})")
    caps: set[str] = set()
    if not args.no_models:
        hub = ModelHub(mcfg)
        hub.load(needed_models(scene, lpr=True))
        print("\nModel:")
        for k, v in sorted(hub.status.items()):
            print(f"  {k:12} {v}")
        print("\nNăng lực:")
        for k, (ok, why) in hub.capabilities().items():
            print(f"  {'✔' if ok else '✘'} {k:18} {why}")
            if ok:
                caps.add(k)
    else:
        caps = {"vehicles", "persons", "traffic_light_det"}
        print("(bỏ qua nạp model — giả định chỉ có model nền COCO)")
    if scene.calibration.enabled:
        from traffic_safety_ai.core.homography import CalibrationError, GroundCalibration

        try:
            cal = GroundCalibration(scene.calibration.image_points, scene.calibration.world_points,
                                    scene.calibration.max_reproj_error_m)
            print("\nHiệu chuẩn: " + cal.summary())
            if cal.ok:
                caps.add("calibration")
        except CalibrationError as e:
            print(f"\nHiệu chuẩn LỖI: {e}")
    cat = LegalCatalog()
    _, report = build_rules(scene, cat, caps)
    print("\n" + format_status_table(report))
    if cat.warning:
        print("\nLưu ý căn cứ pháp lý: " + cat.warning)
    return 0


# ====================================================================== demo
def cmd_demo(args: argparse.Namespace) -> int:
    from traffic_safety_ai.sim.demo import run_demo

    chans = ["cctv", "pov"] if args.channel == "both" else [args.channel]
    for ch in chans:
        print(f"\n=== Mô phỏng {ch.upper()} ===")
        t0 = time.time()
        s = run_demo(ch, data_dir=args.data_dir, write_video=not args.no_video,
                     progress=lambda i, n: print(f"\r  khung {i}/{n}", end="", flush=True))
        print(f"\r  xong trong {time.time() - t0:.0f} s — {len(s['events'])} sự kiện")
        for e in s["events"]:
            print(f"   {e['id']}  {e['rule']:6} {e['name'][:52]:52} t={e['t']:>6}s")
        print(f"  Báo cáo: {s['report']}")
        if s["video"]:
            print(f"  Video chú thích: {s['video']}")
        print(f"  Hồ sơ bằng chứng: {Path(s['db']).parent / 'evidence'}")
        if args.open and s["report"]:
            import webbrowser

            webbrowser.open(Path(s["report"]).as_uri())
    print("\nLưu ý: dữ liệu MÔ PHỎNG chỉ để kiểm tra luồng xử lý, không phản ánh độ chính xác trên video thật.")
    return 0


# ====================================================================== run
def cmd_run(args: argparse.Namespace) -> int:
    import cv2

    from traffic_safety_ai.core.config import load_models_config, load_site_config
    from traffic_safety_ai.core.detector import ModelHub
    from traffic_safety_ai.core.pipeline import AsyncVideoWriter, Pipeline, YoloPerception
    from traffic_safety_ai.core.video_source import VideoSource, guess_start_time
    from traffic_safety_ai.evidence.store import Store
    from traffic_safety_ai.lpr.reader import build_plate_reader
    from traffic_safety_ai.rules.catalog import LegalCatalog
    from traffic_safety_ai.rules.registry import format_status_table, needed_models

    paths = _paths(args)
    scene_file = _scene_path(args.scene)
    scene = load_site_config(scene_file)
    if args.channel:
        scene.camera.channel = args.channel
    mcfg = load_models_config(args.models)
    if args.device:
        mcfg.device = args.device
    src = VideoSource(args.source, stride=args.stride, loop=args.loop, max_frames=args.max_frames)
    W, H = src.size
    ref = scene.camera.frame_size
    scene = scene.scaled_to(W, H)
    if ref and (abs(ref[0] / ref[1] - W / max(H, 1)) > 0.02):
        log.warning("Tỉ lệ khung video (%dx%d) khác cấu hình (%dx%d) — toạ độ vùng/vạch có thể lệch, hãy chạy lại "
                    "'calibrate' cho video này.", W, H, ref[0], ref[1])
    start, how = (_parse_time(args.start_time), "theo --start-time") if args.start_time else \
        guess_start_time(src.sources[0], src.duration_s)
    log.info("Nguồn: %s | %dx%d @ %.1f fps | bắt đầu %s (%s)", args.source, W, H, src.fps, start, how)

    hub = ModelHub(mcfg)
    hub.load(needed_models(scene, lpr=not args.no_lpr))
    for k, v in sorted(hub.status.items()):
        log.info("model %-11s %s", k, v)
    perception = YoloPerception(hub, mcfg, scene)
    plates, lpr_desc = (None, "tắt (--no-lpr)") if args.no_lpr else build_plate_reader(hub, mcfg)
    log.info("Đọc biển số: %s", lpr_desc)
    only = {c.strip().upper() for c in args.rules.split(",")} if args.rules else None
    store = Store(paths.db)
    fps_eff = src.fps / max(1, args.stride)
    models_desc = {k: v for k, v in hub.status.items()}
    models_desc["lpr"] = lpr_desc
    pipe = Pipeline(scene, perception, paths, store, plates, source=str(args.source), fps=fps_eff, start_time=start,
                    catalog=LegalCatalog(), only_rules=only, draw_every=args.draw_every, models_desc=models_desc,
                    run_note=f"cấu hình {scene_file.name}; giờ bắt đầu {how}")
    pipe.viz.labels = "day_du" if args.labels == "full" else "gon"
    print(format_status_table(pipe.rule_report))
    live = None
    if args.dashboard:
        from traffic_safety_ai.dashboard.app import LiveState, serve

        live = LiveState()
        live.pipeline = pipe
        pipe.preview = True
        serve(args.host, args.port, args.token, paths, live, block=False)
    writer = None
    if args.save_video:
        out = pipe.run_dir / "video_chu_thich.mp4"
        writer = AsyncVideoWriter(str(out), fps_eff, (W, H))
        log.info("Ghi video chú thích: %s", out)
    win = "traffic_safety_ai (q: thoat, space: tam dung)"
    paused = False
    n_ev = 0
    t0 = time.time()
    src.start()
    try:
        for pkt in src:
            res = pipe.process(pkt.frame_idx, pkt.t, pkt.frame, render=args.show or args.save_video)
            n_ev += len(res.events)
            if writer is not None and res.annotated is not None:
                writer.write(res.annotated)
            if args.show and res.annotated is not None:
                view = res.annotated
                if view.shape[1] > args.display_width:
                    s = args.display_width / view.shape[1]
                    view = cv2.resize(view, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                cv2.imshow(win, view)
                k = cv2.waitKey(0 if paused else 1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord(" "):
                    paused = not paused
            if pkt.frame_idx % 300 == 0:
                el = time.time() - t0
                log.info("khung %d | %.1f khung/s | %d sự kiện | %s", pkt.frame_idx, pipe.proc_fps, n_ev,
                         f"{pkt.t / max(el, 1e-6):.2f}× thời gian thực")
    except KeyboardInterrupt:
        log.info("Dừng theo yêu cầu (Ctrl+C)")
    finally:
        src.stop()
        pipe.finish()
        if writer is not None:
            writer.close()
        if args.show:
            cv2.destroyAllWindows()
    log.info("Xong: %d khung, %d sự kiện. Thư mục lần chạy: %s", pipe.frames, n_ev, pipe.run_dir)
    if not args.no_report:
        from traffic_safety_ai.analytics.reports import build_report

        rep = build_report(store, paths, camera_id=scene.camera.id, run_id=pipe.run_id,
                           speed_limit=scene.speed.limit_kmh)
        print(f"Báo cáo: {rep}")
    if args.dashboard and live is not None:
        print("Dashboard vẫn mở để duyệt hồ sơ — nhấn Ctrl+C để thoát.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    store.close()
    return 0


# ====================================================================== calibrate / preview
def cmd_calibrate(args: argparse.Namespace) -> int:
    from traffic_safety_ai.tools.roi_calibrator import run_calibrator

    base = str(_scene_path(args.base)) if args.base else str(CONFIG_DIR / "cctv_config.yaml")
    run_calibrator(args.source, args.scene, base=base, start_s=args.at)
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    from traffic_safety_ai.core.config import load_site_config
    from traffic_safety_ai.tools.roi_calibrator import preview_scene

    scene = load_site_config(_scene_path(args.scene))
    out = args.out or str(Path(args.scene).with_suffix(".preview.jpg") if Path(args.scene).exists()
                          else Path("preview.jpg"))
    print(preview_scene(args.source, scene, out, args.at))
    return 0


# ====================================================================== dashboard / report / lookup
def cmd_dashboard(args: argparse.Namespace) -> int:
    from traffic_safety_ai.dashboard.app import serve

    serve(args.host, args.port, args.token, _paths(args), block=True)
    return 0


def _open_app_window(url: str) -> str:
    """Mở Studio trong cửa sổ riêng kiểu ứng dụng (Edge/Chrome --app); không có thì mở tab trình duyệt mặc định."""
    import os
    import shutil
    import subprocess
    import webbrowser

    cands: list[str] = []
    for env in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if base:
            cands += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                      os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    cands += [shutil.which(n) or "" for n in ("msedge", "chrome", "google-chrome", "chromium")]
    for exe in cands:
        if exe and os.path.isfile(exe):
            try:
                subprocess.Popen([exe, f"--app={url}", "--window-size=1500,900"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"cửa sổ ứng dụng ({os.path.basename(exe)})"
            except OSError:
                continue
    webbrowser.open(url)
    return "trình duyệt mặc định"


def _http_json(url: str, timeout: float = 2.0) -> dict | None:
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 — chỉ gọi 127.0.0.1
            return json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _port_busy(port: int) -> bool:
    import socket

    with socket.socket() as sk:
        sk.settimeout(0.5)
        return sk.connect_ex(("127.0.0.1", port)) == 0


def _pid_on_port(port: int) -> int | None:
    """PID của tiến trình đang NGHE cổng (Windows: netstat; Linux/macOS: lsof)."""
    import os
    import subprocess

    try:
        if os.name == "nt":
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3].upper() == "LISTENING":
                    return int(parts[4])
        else:
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True,
                                 timeout=10).stdout.split()
            if out:
                return int(out[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _kill(pid: int) -> None:
    import os
    import signal
    import subprocess

    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=15)
    else:
        os.kill(pid, signal.SIGTERM)


def cmd_studio(args: argparse.Namespace) -> int:
    """Mở như app: bật máy chủ nội bộ + mở cửa sổ Studio (kéo-thả video, tự vẽ, chạy)."""
    import os

    from traffic_safety_ai.dashboard.app import BUILD, serve

    log = logging.getLogger("tsa.studio")
    port = args.port
    if _port_busy(port):
        info = _http_json(f"http://127.0.0.1:{port}/api/info")
        if info is None or "version" not in info:
            # cổng bị CHƯƠNG TRÌNH KHÁC chiếm → không đụng tới nó, tự dời sang cổng trống kế tiếp
            old = port
            port = next((p for p in range(port + 1, port + 20) if not _port_busy(p)), 0)
            if not port:
                print(f"[LỖI] Cổng {old}–{old + 19} đều đang bận. Đóng bớt chương trình rồi mở lại.")
                return 1
            print(f"Cổng {old} đang bị chương trình khác dùng → Studio chạy ở cổng {port}.")
        elif info.get("build") == BUILD:
            url = f"http://127.0.0.1:{port}/studio"
            print(f"Studio đang chạy sẵn (đúng bản mới nhất) — chỉ mở lại cửa sổ: {url}")
            if not args.no_open:
                print("Đã mở bằng", _open_app_window(url))
            return 0
        else:
            job = (_http_json(f"http://127.0.0.1:{port}/api/studio/job") or {}).get("job") or {}
            if job.get("active"):
                print("Máy chủ Studio BẢN CŨ đang phân tích dở một video — để nó chạy xong.")
                print("Xong thì đóng cửa sổ đen cũ rồi mở lại MO_APP.bat để dùng bản mới.")
                if not args.no_open:
                    _open_app_window(f"http://127.0.0.1:{port}/studio")
                return 0
            pid = info.get("pid") or _pid_on_port(port)
            print(f"Phát hiện máy chủ Studio BẢN CŨ còn chạy (PID {pid}) → tắt nó để chạy bản mới...")
            if pid and int(pid) != os.getpid():
                _kill(int(pid))
            for _ in range(50):
                if not _port_busy(port):
                    break
                time.sleep(0.2)
            else:
                print("[LỖI] Không tắt được máy chủ cũ. Hãy đóng mọi cửa sổ đen 'Traffic Studio' / dashboard rồi mở lại.")
                return 1
            log.info("Đã tắt máy chủ bản cũ (PID %s, build %s) để chạy build %s", pid, info.get("build"), BUILD)
    url = f"http://127.0.0.1:{port}/studio"
    server, _tok = serve("127.0.0.1", port, None, _paths(args), block=False)
    for h in logging.getLogger().handlers:                   # lỗi của máy chủ web cũng ghi vào tệp nhật ký
        if isinstance(h, logging.FileHandler):
            for name in ("uvicorn.error", "uvicorn"):
                logging.getLogger(name).addHandler(h)
    for _ in range(150):
        if getattr(server, "started", False):
            break
        time.sleep(0.1)
    else:
        print("[LỖI] Máy chủ nội bộ không khởi động được — xem data\\logs\\studio_*.log")
        return 1
    log.info("Studio sẵn sàng: %s (build %s, PID %s)", url, BUILD, os.getpid())
    print(f"Studio: {url}")
    if not args.no_open:
        print("Đã mở bằng", _open_app_window(url))
    print("GIỮ cửa sổ đen này mở trong lúc dùng (thu nhỏ được). Đóng nó = tắt Studio.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from traffic_safety_ai.analytics.reports import build_report
    from traffic_safety_ai.evidence.store import Store

    paths = _paths(args)
    store = Store(paths.db)
    out = build_report(store, paths, camera_id=args.camera, day=args.day, run_id=args.run,
                       speed_limit=args.limit)
    store.close()
    print(out)
    if args.open:
        import webbrowser

        webbrowser.open(out.as_uri())
    return 0


def cmd_lookup(args: argparse.Namespace) -> int:
    from traffic_safety_ai.evidence.store import Store
    from traffic_safety_ai.lpr.vn_plate import normalize_text, parse_plate

    store = Store(_paths(args).db)
    q = normalize_text(args.plate)
    p = parse_plate(q)
    rows = store.lookup_plate(q)
    print(f"Biển số: {p.display if p else q}{'' if (p and p.valid) else ' (không đúng mẫu biển VN)'} — {len(rows)} hồ sơ")
    for r in rows:
        print(f"  {r['id']} | {r['occurred_at']} | {r['rule_code']} {r['rule_name']} | {r['plate_display']} | "
              f"{r['location']} | {r['status']}")
    store.close()
    return 0


# ====================================================================== export / train / cleanup
def cmd_export(args: argparse.Namespace) -> int:
    from traffic_safety_ai.core.config import load_models_config
    from traffic_safety_ai.core.detector import export_model

    mcfg = load_models_config(args.models)
    names = [n.strip() for n in args.which.split(",")]
    for n in names:
        spec = getattr(mcfg, n, None)
        if spec is None or not spec.weights:
            print(f"  {n}: không có weights trong models.yaml")
            continue
        try:
            out = export_model(spec.weights, args.format, args.imgsz or spec.imgsz, not args.fp32, mcfg.device)
            print(f"  {n}: {out}  → sửa models.yaml: {n}.weights = {out}")
        except Exception as e:
            print(f"  {n}: lỗi xuất — {e}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from traffic_safety_ai.training.train import train

    return train(args.task, args.data, args.epochs, args.imgsz, args.batch, args.model, args.device, args.name)


def cmd_cleanup(args: argparse.Namespace) -> int:
    from traffic_safety_ai.core.config import load_site_config
    from traffic_safety_ai.evidence.store import Store

    paths = _paths(args)
    days = args.days or load_site_config(_scene_path(args.scene)).privacy.retention_days
    store = Store(paths.db)
    old = store.expired(days)
    print(f"{len(old)} hồ sơ quá {days} ngày")
    if not args.yes:
        print("Thêm --yes để xoá thật (thư mục bằng chứng + dòng CSDL).")
        return 0
    from traffic_safety_ai.evidence.store import evidence_folder

    for r in old:
        d = evidence_folder(r, paths.evidence)
        if d is not None and d.resolve().is_relative_to(paths.evidence.resolve()):
            shutil.rmtree(d, ignore_errors=True)
        store.delete_violation(r["id"])
        store.audit("cleanup", "delete", r["id"], f"quá hạn {days} ngày")
    print("Đã xoá.")
    store.close()
    return 0


# ====================================================================== parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m traffic_safety_ai", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true", help="in chi tiết (debug)")
    p.add_argument("--data-dir", help="thư mục dữ liệu (CSDL, bằng chứng, báo cáo); mặc định <gói>/data hoặc TSA_DATA_DIR")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check", help="kiểm tra môi trường + bảng tình huống")
    s.add_argument("--scene", help="file cấu hình địa điểm (mặc định configs/cctv_config.yaml)")
    s.add_argument("--models", help="file models.yaml")
    s.add_argument("--no-models", action="store_true", help="không nạp model (kiểm tra nhanh cấu hình)")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("demo", help="chạy mô phỏng không cần camera/model")
    s.add_argument("--channel", choices=["cctv", "pov", "both"], default="both")
    s.add_argument("--no-video", action="store_true", help="không ghi video chú thích")
    s.add_argument("--open", action="store_true", help="mở báo cáo bằng trình duyệt khi xong")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("run", help="chạy trên video/thư mục/RTSP/webcam")
    s.add_argument("--source", required=True, help="đường dẫn video, thư mục video, rtsp://..., hoặc số webcam")
    s.add_argument("--scene", help="file cấu hình địa điểm")
    s.add_argument("--models", help="file models.yaml")
    s.add_argument("--channel", choices=["cctv", "pov"], help="ghi đè kênh trong cấu hình")
    s.add_argument("--start-time", help="giờ bắt đầu video, vd '2026-11-04 07:00:00' (mặc định đoán theo tên/giờ file)")
    s.add_argument("--rules", help="chỉ chạy các mã này, vd RB-01,RB-05")
    s.add_argument("--stride", type=int, default=1, help="xử lý 1 trên N khung (tăng tốc, giảm độ chính xác)")
    s.add_argument("--max-frames", type=int)
    s.add_argument("--loop", action="store_true", help="lặp lại video (thử nghiệm)")
    s.add_argument("--device", help="cuda:0 | cpu (ghi đè models.yaml)")
    s.add_argument("--no-lpr", action="store_true", help="không đọc biển số")
    s.add_argument("--show", action="store_true", help="hiện cửa sổ xem trực tiếp")
    s.add_argument("--labels", choices=["compact", "full"], default="compact",
                   help="compact: chỉ xe vi phạm mới có chữ (mặc định); full: mọi xe đều ghi mã/loại/tốc độ")
    s.add_argument("--display-width", type=int, default=1280)
    s.add_argument("--save-video", action="store_true", help="ghi video có chú thích")
    s.add_argument("--draw-every", type=int, default=5, help="cập nhật ảnh xem trước mỗi N khung")
    s.add_argument("--no-report", action="store_true")
    s.add_argument("--dashboard", action="store_true", help="mở dashboard trong lúc chạy")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8200)
    s.add_argument("--token", help="mã truy cập dashboard (bắt buộc khi host không phải 127.0.0.1)")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("calibrate", help="khai báo vùng/vạch/đèn/hiệu chuẩn bằng chuột")
    s.add_argument("--source", required=True)
    s.add_argument("--scene", required=True, help="file YAML sẽ ghi (có rồi thì mở ra sửa tiếp)")
    s.add_argument("--base", help="file mẫu để kế thừa tham số (mặc định configs/cctv_config.yaml)")
    s.add_argument("--at", type=float, default=1.0, help="giây lấy khung hình")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("preview", help="vẽ cấu hình lên khung hình → ảnh JPG")
    s.add_argument("--source", required=True)
    s.add_argument("--scene", required=True)
    s.add_argument("--out")
    s.add_argument("--at", type=float, default=1.0)
    s.set_defaults(fn=cmd_preview)

    s = sub.add_parser("dashboard", help="trang quản trị")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8200)
    s.add_argument("--token")
    s.set_defaults(fn=cmd_dashboard)

    s = sub.add_parser("studio", help="mở như app: kéo-thả video, tự vẽ vùng/vạch trên trình duyệt, bấm chạy")
    s.add_argument("--port", type=int, default=8200)
    s.add_argument("--no-open", action="store_true", help="không tự mở cửa sổ")
    s.set_defaults(fn=cmd_studio)

    s = sub.add_parser("report", help="lập báo cáo tổng hợp")
    s.add_argument("--camera")
    s.add_argument("--day", help="YYYY-MM-DD")
    s.add_argument("--run")
    s.add_argument("--limit", type=float, help="giới hạn tốc độ để tính %% vượt")
    s.add_argument("--open", action="store_true")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("lookup", help="tra cứu theo biển số")
    s.add_argument("plate")
    s.set_defaults(fn=cmd_lookup)

    s = sub.add_parser("export-models", help="xuất ONNX/TensorRT")
    s.add_argument("--models")
    s.add_argument("--which", default="base")
    s.add_argument("--format", choices=["onnx", "engine", "openvino"], default="onnx")
    s.add_argument("--imgsz", type=int)
    s.add_argument("--fp32", action="store_true")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("train", help="huấn luyện model bổ sung")
    s.add_argument("--task", required=True, choices=["helmet", "plate", "plate_chars", "cargo", "phone", "lane_seg",
                                                     "orientation"])
    s.add_argument("--data", required=True, help="file dataset YAML (xem training/datasets/)")
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--imgsz", type=int)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--model", help="model khởi tạo (mặc định theo task)")
    s.add_argument("--device", default="0")
    s.add_argument("--name")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("cleanup", help="xoá hồ sơ quá hạn lưu trữ")
    s.add_argument("--days", type=int)
    s.add_argument("--scene")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_cleanup)
    return p


def main(argv: list[str] | None = None) -> int:
    from traffic_safety_ai.utils.logger import setup_logging

    args = build_parser().parse_args(argv)
    log_file = None
    if args.cmd in ("run", "studio"):
        from traffic_safety_ai.core.config import AppPaths

        log_file = AppPaths(args.data_dir).ensure().root / "logs" / f"{args.cmd}_{datetime.now():%Y%m%d}.log"
    setup_logging(args.verbose, log_file)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # thông báo gọn bằng tiếng Việt, chi tiết trong -v
        from traffic_safety_ai.core.config import ConfigError

        if isinstance(e, ConfigError | FileNotFoundError | OSError):
            log.error("%s", e)
            if args.verbose:
                raise
            return 2
        log.exception("Lỗi không lường trước khi chạy lệnh '%s'", args.cmd)      # để lại dấu vết trong tệp nhật ký
        print(f"\n[LỖI] {type(e).__name__}: {e}\n(chi tiết đã ghi trong thư mục data\\logs)")
        return 1


_ = (Any, sys, PACKAGE_ROOT)
