"""Chạy luồng realtime: python -m realtime [--no-collector] [--cam cam01]

Phase 1 nghiệm thu: bbox + ID hiển thị, telemetry chạy, và FPS THẬT >= target_fps.
Nếu chưa đạt thì dừng lại xem bottleneck, đừng đi tiếp sang phase 2.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

from common.config import get_settings
from common.db import init_db
from realtime import dashboard
from realtime.dashboard import StreamBuffer
from realtime.pipeline import RealtimePipeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="realtime")
    ap.add_argument("--cam", help="chỉ chạy 1 camera (mặc định: tất cả trong CAMERAS)")
    ap.add_argument("--video", help="Dùng file video thay cho camera để TEST khi không có camera "
                                    "(vd: --video samples/test.mp4). Chạy đúng 1 nguồn.")
    ap.add_argument("--fps", type=float, default=None,
                    help="Giới hạn tốc độ đọc video (fps) cho giống realtime, vd --fps 25. "
                         "Bỏ trống = đọc nhanh nhất có thể (đo throughput thật).")
    loop_grp = ap.add_mutually_exclusive_group()
    loop_grp.add_argument("--loop", dest="loop", action="store_true", default=False,
                          help="Lặp lại video/danh sách khi hết (mặc định KHÔNG lặp)")
    loop_grp.add_argument("--no-loop", dest="loop", action="store_false",
                          help="(mặc định) hết video thì DỪNG")
    ap.add_argument("--no-collector", action="store_true", help="Phase 1: chỉ realtime, không thu mẫu")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()

    cams = s.camera_map()
    if args.video:
        cam_id = args.cam or next(iter(cams), None) or "cam01"
        cams = {cam_id: args.video}
        logging.getLogger("realtime").info(
            "Nguồn TEST: video '%s' (cam_id=%s, loop=%s, fps_limit=%s)",
            args.video, cam_id, args.loop, args.fps,
        )
    elif args.cam:
        cams = {args.cam: cams[args.cam]}
    if not cams:
        raise SystemExit("Không có camera nào trong CAMERAS")

    # Tạo stream buffers riêng cho dashboard
    stream_buffers = {cam_id: StreamBuffer() for cam_id in cams}

    pipelines: dict[str, RealtimePipeline] = {}
    threads: list[threading.Thread] = []
    for cam_id, src in cams.items():
        p = RealtimePipeline(cam_id, src, s, enable_collector=not args.no_collector,
                             stream_buffer=stream_buffers[cam_id],
                             video_loop=args.loop, video_fps=args.fps)
        pipelines[cam_id] = p
        th = threading.Thread(target=p.run, kwargs={"max_frames": args.max_frames}, name=f"pipe-{cam_id}")
        th.start()
        threads.append(th)

    if not args.no_dashboard:
        dashboard.serve(pipelines, stream_buffers)
        print(f"Dashboard: http://{s.dashboard_host}:{s.dashboard_port}")

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        for p in pipelines.values():
            p.stop()
    for t in threads:
        t.join(timeout=5)

    # Điều kiện nghiệm thu Phase 1
    for cam_id, p in pipelines.items():
        fps = p.telemetry.fps
        drop = p.telemetry.frame_drop_rate * 100
        verdict = "ĐẠT" if fps >= s.target_fps else "CHƯA ĐẠT"
        print(f"[{cam_id}] FPS thật {fps:.1f} (target {s.target_fps}) — {verdict}; "
              f"drop {drop:.1f}%; inference {p.telemetry.inference_ms:.1f} ms")


if __name__ == "__main__":
    main()
