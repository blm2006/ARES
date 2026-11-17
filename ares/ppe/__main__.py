"""python -m ares.ppe setup | check"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Model mặc định: YOLOv8s do nhóm tác giả bộ dữ liệu SH17 công bố (17 lớp: kính, chụp tai, khẩu trang, kính che mặt,
# găng tay, nón, áo phản quang, giày, áo bảo hộ + các bộ phận cơ thể). Bộ dữ liệu SH17 theo giấy phép CC BY-NC-SA 4.0
# (phi thương mại): dùng cho xưởng của trường / nghiên cứu / thử nghiệm. Khi bán thương mại phải train model riêng.
DEFAULT_URL = "https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo8s.pt"
DEFAULT_SHA256 = "dfaf451bdda3f92fe4012327e337225929ad7af22146b00e0d6989c2f4e40a4b"


def setup(cfg, url: str, sha256: str, force: bool) -> int:
    from ares.ppe.load import weights_path

    dest = weights_path(cfg)
    if dest.exists() and not force:
        print(f"Đã có model: {dest}")
    else:
        print(f"Tải model đồ bảo hộ (~22 MB)\n  từ {url}\n  về  {dest}")
        tmp = dest.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                h, n = hashlib.sha256(), 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
                    print(f"\r  {n / 1e6:.1f} MB", end="", flush=True)
            print()
        except Exception as e:  # noqa: BLE001
            print(f"\nTải lỗi: {e}\nKiểm tra Internet rồi chạy lại. Hoặc tự tải file trên, đổi tên thành {dest.name}, đặt vào {dest.parent}")
            return 1
        if sha256 and h.hexdigest() != sha256:
            tmp.unlink(missing_ok=True)
            print("File tải về KHÔNG khớp mã kiểm tra (có thể bị đổi nội dung) — đã xoá, không dùng.")
            return 1
        tmp.replace(dest)
        print("Tải xong, mã kiểm tra khớp.")
    return check(cfg)


def check(cfg) -> int:
    from ares.ppe import ITEMS
    from ares.ppe.load import load_detector

    det, info = load_detector(cfg)
    if det is None:
        print("CHƯA DÙNG ĐƯỢC: " + (info.get("error") or info.get("hint", "")))
        return 1
    print(f"Model {info['weights']} chạy trên {info['device']}. Các lớp của model:")
    for name, item in info["classes"].items():
        print(f"  {name:22s} -> " + (ITEMS[item][0] if item else "(bỏ qua — không phải đồ bảo hộ)"))
    print("Nhận ra được: " + ", ".join(ITEMS[i][0] for i in info["items"]))
    miss = [ITEMS[i][0] for i in ITEMS if i not in info["items"]]
    if miss:
        print("KHÔNG nhận ra (đừng bật các món này): " + ", ".join(miss))
    print("\nXong. Mở lại ARES_MAY_CHU.bat, rồi vào dashboard -> Địa điểm -> Đồ bảo hộ để chọn món bắt buộc.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ares.ppe", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="tải model đồ bảo hộ mặc định và kiểm tra")
    s.add_argument("--url", default=DEFAULT_URL)
    s.add_argument("--sha256", default=None, help="mã kiểm tra của file (mặc định: mã của model mặc định)")
    s.add_argument("--force", action="store_true")
    sub.add_parser("check", help="nạp model, in các lớp và món nhận ra được")
    args = ap.parse_args(argv)
    from ares.config import AresSettings

    cfg = AresSettings()
    if args.cmd == "setup":
        sha = args.sha256 if args.sha256 is not None else (DEFAULT_SHA256 if args.url == DEFAULT_URL else "")
        return setup(cfg, args.url, sha, args.force)
    return check(cfg)


if __name__ == "__main__":
    sys.exit(main())
