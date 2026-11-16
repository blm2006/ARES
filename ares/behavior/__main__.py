"""python -m ares.behavior <lệnh>

    download   Tải dataset công khai (UR Fall Detection + Surveillance Camera Fight)
    extract    Chạy YOLO-pose trên video đã tải / video của bạn -> chuỗi khung xương (.npz)
    train      Train model ST-GCN-lite (dùng cả sự kiện đã duyệt Đúng/Báo nhầm trên dashboard)
    video      Thử trên file video hoặc thư mục video, xuất video có khung xương + danh sách sự kiện
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv=None) -> int:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="python -m ares.behavior", description="Nhận diện hành vi cho ARES")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="tải dataset công khai")
    d.add_argument("--urfd-max", type=int, default=0, help="chỉ tải N clip ngã + N clip sinh hoạt (0 = tất cả)")
    d.add_argument("--no-urfd", action="store_true")
    d.add_argument("--no-fight", action="store_true")
    e = sub.add_parser("extract", help="tách khung xương")
    e.add_argument("--force", action="store_true", help="tách lại cả clip đã có")
    e.add_argument("--fps", type=float, default=10.0)
    e.add_argument("--only", default="", help="chỉ clip có tên chứa chuỗi này")
    t = sub.add_parser("train", help="train model hành vi")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--device", default="")
    t.add_argument("--no-promote", action="store_true", help="không chép thành current.pt")
    v = sub.add_parser("video", help="thử trên video")
    v.add_argument("path")
    v.add_argument("--fps", type=float, default=10.0)
    v.add_argument("--mode", choices=["rules", "model", "hybrid"])
    v.add_argument("--still-s", type=float, help="ngưỡng bất động (giây), mặc định theo cấu hình (30)")
    v.add_argument("--out", default="")
    v.add_argument("--debug", action="store_true", help="in tư thế/chuyển động từng người mỗi giây")
    v.add_argument("--no-write", action="store_true", help="không ghi video kết quả")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from ares.config import AresSettings

    over = {}
    if a.cmd == "video":
        if a.mode:
            over["behavior_mode"] = a.mode
        if a.still_s:
            over["behavior_still_s"] = a.still_s
    cfg = AresSettings(**over)

    if a.cmd == "download":
        from ares.behavior.datasets import download
        download(cfg, urfd=not a.no_urfd, urfd_max=a.urfd_max, fight=not a.no_fight)
    elif a.cmd == "extract":
        from ares.behavior.datasets import extract
        extract(cfg, force=a.force, fps=a.fps, only=a.only)
    elif a.cmd == "train":
        from ares.behavior.train import train
        train(cfg, epochs=a.epochs, batch=a.batch, lr=a.lr, promote=not a.no_promote, device=a.device)
    elif a.cmd == "video":
        from ares.behavior.run_video import run
        run(cfg, Path(a.path), Path(a.out) if a.out else None, fps=a.fps, debug=a.debug, write=not a.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
