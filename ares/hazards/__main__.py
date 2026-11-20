"""python -m ares.hazards setup | check | train [--only fire] [--epochs 30]"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


def download(url: str, dest: Path, sha256: str) -> bool:
    if dest.exists():
        h = hashlib.sha256(dest.read_bytes()).hexdigest()
        if not sha256 or h == sha256:
            print(f"  Đã có {dest.name}")
            return True
        print(f"  {dest.name} khác bản gốc — tải lại")
    print(f"  Tải {dest.name}\n    từ {url}")
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            h, n = hashlib.sha256(), 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                h.update(chunk)
                n += len(chunk)
                print(f"\r    {n / 1e6:.1f} MB", end="", flush=True)
        print()
    except Exception as e:  # noqa: BLE001
        print(f"\n  Tải lỗi: {e}. Tự tải file trên, đổi tên thành {dest.name}, đặt vào {dest.parent}")
        return False
    if sha256 and h.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        print("  File tải về KHÔNG khớp mã kiểm tra — đã xoá, không dùng.")
        return False
    tmp.replace(dest)
    return True


def main(argv=None) -> int:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(prog="python -m ares.hazards")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="tải model cháy/khói + hút thuốc")
    sub.add_parser("check", help="kiểm model đã cài")
    t = sub.add_parser("train", help="train dần từ sự kiện đã đánh giá")
    t.add_argument("--only", default="")
    t.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args(argv)

    from ares.config import REPO_ROOT, AresSettings
    from ares.hazards import HAZARDS

    cfg = AresSettings()
    if a.cmd == "setup":
        ok = True
        for k, h in HAZARDS.items():
            print(f"{h['name']} — {h['source']} — giấy phép {h['license']}")
            ok = download(h["url"], REPO_ROOT / h["weights"], h["sha256"]) and ok
        a.cmd = "check"
        if not ok:
            return 1
    if a.cmd == "check":
        from ares.hazards.detector import load_hazards

        dets, info = load_hazards(cfg)
        for k, i in info.items():
            print(f"  {i.get('name', k)}: " + (f"sẵn sàng ({i['weights']}, {i['device']}, lớp {', '.join(i['classes'])})"
                                                 if i.get("ready") else (i.get("error") or i.get("hint", ""))))
        return 0 if dets else 1
    if a.cmd == "train":
        from ares.hazards.train import train_all

        for r in train_all(cfg, a.epochs, a.only):
            print(r["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
