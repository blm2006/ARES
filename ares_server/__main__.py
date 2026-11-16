"""python -m ares_server <lệnh>

    run                      chạy server + dashboard (mặc định http://127.0.0.1:8200)
    add-site "Tên" [--address ...]   tạo địa điểm, in KHOÁ THIẾT BỊ để dán vào máy Edge (ARES_UPLINK_KEY)
    sites                    liệt kê địa điểm
    rotate-key <id>          đổi khoá thiết bị của địa điểm (khoá cũ hết hiệu lực)
    link-local ["Tên"] [--live 0.5] [--hub-fps 15]  Edge + server cùng 1 máy: tạo địa điểm, tự ghi URL + khoá (+ hình trực tiếp) vào ares_config/ares.env
    setup-host               chế độ MÁY CHỦ: mở server cho mạng nội bộ (ghi ares_config/server.env)
    lan-info                 in IP máy chủ + địa chỉ để máy khác vào dashboard / tải app ARES Viewer
    reset-password <tên>     đặt lại mật khẩu một tài khoản (khi quên) — chạy trên máy chủ
    share [--pages URL] [--off]  mở máy chủ ra Internet qua Tailscale Funnel (https cố định) + in link dashboard GitHub Pages
    build-web [--server URL] [--out web]   xuất dashboard tĩnh để đặt lên GitHub Pages (gọi về máy chủ qua https)
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv=None) -> int:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="python -m ares_server", description="ARES Server trung tâm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--host")
    r.add_argument("--port", type=int)
    a = sub.add_parser("add-site")
    a.add_argument("name")
    a.add_argument("--address", default="")
    a.add_argument("--note", default="")
    sub.add_parser("sites")
    k = sub.add_parser("rotate-key")
    k.add_argument("site_id", type=int)
    lk = sub.add_parser("link-local")
    lk.add_argument("name", nargs="?", default="")
    lk.add_argument("--live", type=float, default=0.0, help="gửi hình trực tiếp mỗi N giây (0 = không)")
    lk.add_argument("--hub-fps", type=float, default=0.0, help="đặt ARES_HUB_FPS tối thiểu (camera của máy khác)")
    sub.add_parser("setup-host")
    sub.add_parser("lan-info")
    rp = sub.add_parser("reset-password")
    rp.add_argument("username")
    sh = sub.add_parser("share")
    sh.add_argument("--pages", default="", help="địa chỉ dashboard GitHub Pages, vd https://ten.github.io/ares-dashboard/")
    sh.add_argument("--off", action="store_true", help="tắt chia sẻ ra Internet")
    bw = sub.add_parser("build-web")
    bw.add_argument("--server", default="", help="địa chỉ https mặc định của máy chủ (vd https://may-chu.ten.ts.net)")
    bw.add_argument("--out", default="web")
    args = ap.parse_args(argv)
    if args.cmd == "build-web":
        print(build_web(args.out, args.server))
        return 0
    if args.cmd == "share":
        return share(args.pages, args.off)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")

    from ares_server.config import ServerSettings
    from ares_server.db import ServerDB

    over = {}
    if args.cmd == "run":
        if args.host:
            over["host"] = args.host
        if args.port:
            over["port"] = args.port
    if args.cmd == "setup-host":
        return setup_host()
    cfg = ServerSettings(**over)
    cfg.ensure_dirs()
    if args.cmd == "lan-info":
        return lan_info(cfg)

    if args.cmd == "run":
        import uvicorn

        from ares_server.app import check_exposure, create_app

        check_exposure(cfg)
        shown = "localhost" if cfg.host in ("0.0.0.0", "::") else cfg.host
        print(f"ARES Server: http://{shown}:{cfg.port}  (dữ liệu: {cfg.data_dir})")
        try:
            from ares_server.db import ServerDB as _DB
            _d = _DB(cfg.data_dir / "server.db")
            if _d.count_users() == 0 and not cfg.admin_password:
                print(f"CHƯA CÓ TÀI KHOẢN: mở http://127.0.0.1:{cfg.port} trên máy này để tạo tài khoản quản trị.")
            _d.close()
        except Exception:  # noqa: BLE001
            pass
        uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="warning", access_log=False)
        return 0

    db = ServerDB(cfg.data_dir / "server.db")
    try:
        if args.cmd == "add-site":
            sid, key = db.create_site(args.name, args.address, args.note)
            print(f"Đã tạo địa điểm #{sid}: {args.name}")
            print(f"KHOÁ THIẾT BỊ (chỉ hiện một lần): {key}")
            print("Trên máy Edge, thêm vào .env:")
            print(f"  ARES_UPLINK_URL=http://<địa-chỉ-server>:{cfg.port}")
            print(f"  ARES_UPLINK_KEY={key}")
        elif args.cmd == "sites":
            for s in db.sites():
                print(f"#{s['id']:<3} {s['name']:30s} khoá {s['key_hint']}  {'đang bật' if s['active'] else 'ĐÃ TẮT'}")
        elif args.cmd == "rotate-key":
            print(f"Khoá mới cho địa điểm #{args.site_id}: {db.rotate_key(args.site_id)}")
        elif args.cmd == "reset-password":
            import getpass

            u = [x for x in db.users() if x["username"].lower() == args.username.lower()]
            if not u:
                print(f"Không có tài khoản '{args.username}'. Có: {', '.join(x['username'] for x in db.users()) or '(chưa có)'}")
                return 1
            pw = getpass.getpass("Mật khẩu mới (>= 8 ký tự): ")
            if getpass.getpass("Nhập lại: ") != pw:
                print("Hai lần nhập không khớp")
                return 1
            try:
                db.set_password(u[0]["id"], pw)
                db.update_user(u[0]["id"], status="active")
            except ValueError as e:
                print(e)
                return 1
            print(f"Đã đặt lại mật khẩu cho '{u[0]['username']}'.")
        elif args.cmd == "link-local":
            link_local(cfg, db, args.name, args.live, args.hub_fps)
    except KeyError as e:
        print(f"Không có địa điểm {e}")
        return 1
    finally:
        db.close()
    return 0


def tune_env(env, key: str, value: float, lower_is_better: bool, comment: str = "") -> None:
    """Đặt KEY=value trong file .env nếu chưa có, hoặc nếu giá trị đang có kém hơn (giữ nguyên CRLF của file)."""
    import re

    raw = env.read_bytes().decode("utf-8") if env.exists() else ""
    nl = "\r\n" if "\r\n" in raw else "\n"
    m = re.search(rf"^{key}=([^\r\n]*)", raw, re.M)
    if m:
        try:
            cur = float(m.group(1).strip())
        except ValueError:
            cur = None
        if cur is not None and (cur <= value if lower_is_better else cur >= value):
            return  # đã tốt bằng/hơn — giữ cấu hình người dùng
        raw = raw[:m.start()] + f"{key}={value:g}" + raw[m.end():]
        print(f"Đã chỉnh {key}={value:g} (trước: {m.group(1).strip()})")
    else:
        if raw and not raw.endswith(("\n",)):
            raw += nl
        raw += (f"{nl}# {comment}{nl}" if comment else nl) + f"{key}={value:g}{nl}"
        print(f"Đã thêm {key}={value:g}")
    env.write_bytes(raw.encode("utf-8"))


def link_local(cfg, db, name: str, live: float = 0.0, hub_fps: float = 0.0) -> None:
    import re
    import socket

    from ares_server.config import REPO_ROOT

    env = REPO_ROOT / "ares_config" / "ares.env"
    env.parent.mkdir(parents=True, exist_ok=True)
    if live > 0:
        tune_env(env, "ARES_UPLINK_LIVE_S", live, lower_is_better=True,
                 comment="hình trực tiếp cho dashboard (giây/khung — nhỏ hơn = mượt hơn, ít trễ hơn)")
    if hub_fps > 0:
        tune_env(env, "ARES_HUB_FPS", hub_fps, lower_is_better=False,
                 comment="số vòng/giây đọc camera phụ (camera của máy khác gửi về) — cao hơn = ít trễ hơn")
    text = env.read_text(encoding="utf-8") if env.exists() else ""
    if re.search(r"^ARES_UPLINK_KEY=\S+", text, re.M):
        print(f"{env} đã có ARES_UPLINK_KEY — giữ nguyên. Muốn tạo lại thì xoá 2 dòng ARES_UPLINK_* trong file đó.")
        return
    name = name or f"Địa điểm {socket.gethostname()}"
    sid, key = db.create_site(name, note="tạo bởi link-local")
    host = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
    block = (f"\n# ---- gửi lên server trung tâm (tạo bởi: python -m ares_server link-local)\n"
             f"ARES_UPLINK_URL=http://{host}:{cfg.port}\nARES_UPLINK_KEY={key}\n")
    env.parent.mkdir(parents=True, exist_ok=True)
    with open(env, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"Đã tạo địa điểm #{sid} '{name}' và ghi khoá vào {env}.")
    print("Khởi động lại ARES (python -m ares run) để bắt đầu gửi số liệu + cảnh báo lên server.")


def setup_host() -> int:
    """Mở server cho mạng nội bộ (ghi ARES_SERVER_HOST=0.0.0.0 vào ares_config/server.env).
    Tài khoản KHÔNG đặt ở đây: tạo trên dashboard (lần đầu mở http://127.0.0.1:8200 trên chính máy chủ)."""
    import re

    from ares_server.config import REPO_ROOT

    env = REPO_ROOT / "ares_config" / "server.env"
    text = env.read_text(encoding="utf-8") if env.exists() else ""
    if re.search(r"^ARES_SERVER_HOST=0\.0\.0\.0", text, re.M):
        return 0
    keep = [ln for ln in text.splitlines() if not re.match(r"^ARES_SERVER_HOST=", ln)]
    keep += ["# chế độ máy chủ: mở dashboard cho các máy trong mạng nội bộ (python -m ares_server setup-host)",
             "ARES_SERVER_HOST=0.0.0.0"]
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"Đã bật chế độ máy chủ ({env}).")
    print("Lần đầu: mở http://127.0.0.1:8200 TRÊN MÁY NÀY để tạo tài khoản quản trị.")
    return 0


def lan_info(cfg) -> int:
    from ares_server.app import lan_ips
    from ares_server.db import ServerDB

    ips = lan_ips()
    print("=" * 64)
    print(" ĐỊA CHỈ IP MÁY CHỦ (mạng nội bộ): " + (", ".join(ips) if ips else "không tìm thấy — xem bằng lệnh ipconfig"))
    if cfg.host in ("127.0.0.1", "localhost", "::1"):
        print(" Server đang CHỈ mở cho máy này. Bật cho máy khác: python -m ares_server setup-host (hoặc ARES_MAY_CHU.bat)")
    else:
        for ip in ips:
            print(f"   Máy khác vào dashboard : http://{ip}:{cfg.port}")
            print(f"   Máy khác tải app       : http://{ip}:{cfg.port}/tai-app")
        print(" Không vào được: chạy MO_TUONG_LUA.bat, đặt mạng Wi-Fi/LAN là Private")
    db = ServerDB(cfg.data_dir / "server.db")
    try:
        if db.count_users() == 0:
            print(f" CHƯA CÓ TÀI KHOẢN: mở http://127.0.0.1:{cfg.port} trên máy này để tạo tài khoản quản trị.")
        else:
            print(f" Tài khoản: {db.count_users('active')} đang hoạt động, {db.count_users('pending')} chờ duyệt.")
    finally:
        db.close()
    print("=" * 64)
    return 0




def build_web(out: str = "web", server: str = "") -> str:
    """Dashboard tĩnh cho GitHub Pages: cùng giao diện với máy chủ, chỉ khác là gọi API về máy chủ qua https + token."""
    import json
    from pathlib import Path

    src = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    inject = ("<script>window.ARES_REMOTE=true;window.ARES_DEFAULT_SERVER=" + json.dumps(server.strip().rstrip("/"))
              + ";</script>\n<script>")
    assert src.count("<script>") == 1
    html = src.replace("<script>", inject, 1)
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.html").write_text(html, encoding="utf-8")
    (d / ".nojekyll").write_text("", encoding="utf-8")
    (d / "README.md").write_text(
        "# ARES — dashboard\n\nGiao diện dashboard ARES (AI camera: đếm người, phát hiện té ngã / bất động / đánh nhau).\n\n"
        "Trang này chỉ là giao diện: AI, camera, tài khoản và dữ liệu nằm trên **máy chủ ARES**. Mở trang, nhập địa chỉ https "
        "của máy chủ (hoặc dùng link có `?server=https://...`), đăng nhập bằng tài khoản trên máy chủ.\n\n"
        "Tạo lại file này: `python -m ares_server build-web --server https://<máy-chủ>`.\n", encoding="utf-8")
    return f"Đã xuất dashboard tĩnh vào {d.resolve()} (máy chủ mặc định: {server or 'chưa đặt — người dùng tự nhập'})"


def set_env(key: str, value: str, comment: str = "") -> None:
    """Ghi/thay KEY=value trong ares_config/server.env."""
    import re

    from ares_server.config import REPO_ROOT

    env = REPO_ROOT / "ares_config" / "server.env"
    raw = env.read_bytes().decode("utf-8") if env.exists() else ""
    nl = "\r\n" if "\r\n" in raw else "\n"
    line = f"{key}={value}"
    if re.search(rf"^{key}=", raw, re.M):
        raw = re.sub(rf"^{key}=[^\r\n]*", lambda m: line, raw, flags=re.M)
    else:
        if raw and not raw.endswith("\n"):
            raw += nl
        raw += (f"# {comment}{nl}" if comment else "") + line + nl
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_bytes(raw.encode("utf-8"))


def _tailscale() -> str | None:
    import os
    import shutil

    exe = shutil.which("tailscale")
    for p in (exe, os.path.expandvars(r"%ProgramFiles%\Tailscale\tailscale.exe"), "/usr/bin/tailscale",
              "/Applications/Tailscale.app/Contents/MacOS/Tailscale"):
        if p and os.path.exists(p):
            return p
    return None


def share(pages: str = "", off: bool = False) -> int:
    """Mở máy chủ ra Internet bằng Tailscale Funnel: https://<máy>.<tailnet>.ts.net -> 127.0.0.1:<port>. Địa chỉ cố định."""
    import json
    import subprocess

    from ares_server.config import ServerSettings

    cfg = ServerSettings()
    ts = _tailscale()
    if ts is None:
        print("Chưa cài Tailscale. Tải ở https://tailscale.com/download , cài, đăng nhập (Google/Microsoft/GitHub), rồi chạy lại.")
        return 2
    if off:
        subprocess.run([ts, "funnel", "reset"])
        print("Đã tắt chia sẻ ra Internet. Máy khác trong mạng nội bộ vẫn vào được như cũ.")
        return 0
    try:
        st = json.loads(subprocess.run([ts, "status", "--json"], capture_output=True, text=True, timeout=20).stdout or "{}")
    except Exception as e:  # noqa: BLE001
        print(f"Không đọc được trạng thái Tailscale: {e}")
        return 2
    if st.get("BackendState") != "Running":
        print("Tailscale chưa đăng nhập / chưa bật. Mở Tailscale (khay đồng hồ) -> Log in, rồi chạy lại.")
        return 2
    dns = ((st.get("Self") or {}).get("DNSName") or "").rstrip(".")
    if not dns:
        print("Không lấy được tên máy trong Tailscale (bật MagicDNS trong trang quản trị Tailscale).")
        return 2
    print("Đang bật Tailscale Funnel (lần đầu Tailscale có thể in 1 đường link — mở link đó, bấm Enable rồi đợi)...")
    r = subprocess.run([ts, "funnel", "--bg", str(cfg.port)], timeout=600)
    if r.returncode != 0:
        print("Bật Funnel chưa được. Mở https://login.tailscale.com/admin/dns -> bật HTTPS Certificates, rồi bật Funnel "
              "theo link Tailscale in ở trên, sau đó chạy lại file này.")
        return 1
    url = f"https://{dns}"
    set_env("ARES_SERVER_PUBLIC_URL", url, "địa chỉ https công khai (python -m ares_server share)")
    if pages:
        pages = pages if pages.endswith("/") else pages + "/"
        set_env("ARES_SERVER_PAGES_URL", pages, "dashboard trên GitHub Pages")
    pages = pages or cfg.pages_url
    print("=" * 70)
    print(f" Máy chủ đã mở ra Internet (https, cố định): {url}")
    print(f" Mở dashboard ngay trên địa chỉ này cũng được:  {url}")
    if pages:
        print(f" Dashboard GitHub Pages (gửi link này cho người khác): {pages}?server={url}")
    print(" Người khác chỉ cần mở link trên trình duyệt (máy tính/điện thoại, mạng nào cũng được), đăng nhập, bật camera.")
    print(" Tắt chia sẻ: python -m ares_server share --off")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
