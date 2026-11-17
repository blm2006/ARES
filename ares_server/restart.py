"""Khởi động lại từ xa (không cần ngồi ở máy chủ).

Tiến trình `run` chạy dưới một "người giám sát" nhỏ: khi file cờ `var/restart.flag` đổi (admin bấm "Khởi động lại" trên
dashboard, hoặc sau khi cập nhật mã nguồn), tiến trình con tự dừng gọn với mã 75 và người giám sát mở lại nó bằng mã MỚI.
Tắt hẳn vẫn như cũ: Ctrl+C hoặc đóng cửa sổ.

File này có 2 bản giống hệt nhau (ares/restart.py và ares_server/restart.py) vì server chạy độc lập được, không import ares.
"""

from __future__ import annotations

import _thread
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)
FLAG = Path("var") / "restart.flag"
CODE = 75  # mã thoát "hãy mở lại tôi"
ENV = "ARES_SUPERVISED"
requested = False


def supervised() -> bool:
    return os.environ.get(ENV) == "1"


def supervise(module: str, argv: list[str]) -> int:
    """Chạy `python -m module argv` và mở lại mỗi khi nó thoát với mã CODE."""
    env = dict(os.environ, **{ENV: "1"})
    quick = 0
    while True:
        t0 = time.time()
        try:
            rc = subprocess.call([sys.executable, "-m", module, *argv], env=env)
        except KeyboardInterrupt:
            return 130
        if rc != CODE:
            return rc
        quick = quick + 1 if time.time() - t0 < 20 else 0
        if quick >= 5:  # mở lại liên tục mà chết ngay: dừng, không quay vòng vô hạn
            print("Khoi dong lai 5 lan lien tiep deu that bai - dung.", flush=True)
            return 1
        print("\n=== ARES: khoi dong lai theo yeu cau ===\n", flush=True)
        time.sleep(1.5)


def watch(flag: Path = FLAG, poll_s: float = 2.0, grace_s: float = 12.0) -> None:
    """Luồng nền: cờ đổi -> dừng gọn (như Ctrl+C); quá grace_s chưa dừng xong thì thoát hẳn. Chỉ bật khi có người giám sát."""
    if not supervised():
        return

    def mtime() -> float:
        try:
            return flag.stat().st_mtime
        except OSError:
            return 0.0

    base = mtime()

    def loop() -> None:
        global requested
        while True:
            time.sleep(poll_s)
            if mtime() > base:
                requested = True
                log.warning("Nhận yêu cầu khởi động lại — đang dừng để mở lại bằng mã mới…")
                t = threading.Timer(grace_s, lambda: os._exit(CODE))
                t.daemon = True
                t.start()
                _thread.interrupt_main()
                return

    threading.Thread(target=loop, name="ares-restart-watch", daemon=True).start()


def request(flag: Path = FLAG) -> None:
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(f"{time.time():.3f}\n", encoding="utf-8")
