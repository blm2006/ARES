"""Server trung tâm ARES: nhận dữ liệu từ các máy Edge + dashboard tổng hợp nhiều địa điểm.

Hai nhóm API:
  /api/v1/*  cho máy Edge — xác thực bằng khoá thiết bị (Authorization: Bearer ares_...), mỗi địa điểm 1 khoá.
  /api/*     cho dashboard — đăng nhập bằng tài khoản quản trị, cookie phiên ký HMAC, mọi POST phải có X-ARES: 1.

Mặc định chỉ nghe 127.0.0.1. Mở ra mạng/Internet bắt buộc đặt ARES_SERVER_ADMIN_PASSWORD và nên đặt sau HTTPS
(Caddy/Nginx) — xem deploy/server/README.md.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import re
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from ares_server import __version__
from ares_server.config import ServerSettings
from ares_server.db import ServerDB

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
EDGE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")
EVID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
from ares_server.ppe import ITEMS as PPE_ITEMS, PRESETS as PPE_PRESETS, clean_rules as clean_ppe  # noqa: E402

KIND_RE = re.compile(r"^[A-Z_]{0,20}$")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".3gp"}
MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB / video
MAX_RESULT_BYTES = 400 * 1024 * 1024
# mạch cảm biến: tên số đo -> dải hợp lệ (ngoài dải = bỏ, coi như cảm biến lỗi). Số đo khác tên vẫn nhận (tối đa 16 số).
SENSOR_RANGES = {"temp": (-40, 125), "temp2": (-40, 125), "hum": (0, 100), "lux": (0, 200000), "light": (0, 100),
                 "motion": (0, 1), "gas": (0, 100), "flame": (0, 1), "sound": (0, 100), "door": (0, 1), "dist": (0, 600),
                 "relay": (0, 1), "led": (0, 100), "buzzer": (0, 1), "co2": (0, 20000), "pm25": (0, 1000), "rssi": (-120, 0)}
SENSOR_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
DEVICE_KINDS = {"esp32": "ESP32 (Wi-Fi)", "esp8266": "ESP8266 — NodeMCU / Wemos (Wi-Fi)",
                "usb": "Arduino Uno / Nano / Mega / R4 — cắm USB", "other": "Mạch khác — gửi HTTP/JSON"}
CONTROL_KEYS = {"relay": (0, 1), "led": (0, 100), "buzzer": (0, 1), **{f"ch{i}": (0, 100) for i in range(1, 9)}}
MAX_CHANNELS = 8


def clean_channels(lst) -> list[dict]:
    """Kênh đèn của 1 mạch: tối đa 8, mỗi kênh = 1 đèn (LED chỉnh độ sáng qua chân PWM, hoặc relay bật/tắt)."""
    out = []
    for i, c in enumerate((lst or [])[:MAX_CHANNELS] if isinstance(lst, list) else []):
        c = c if isinstance(c, dict) else {}
        try:
            pin = int(c.get("pin"))
        except (TypeError, ValueError):
            pin = -1
        out.append({"key": f"ch{i + 1}", "name": str(c.get("name") or f"Đèn {i + 1}").strip()[:40] or f"Đèn {i + 1}",
                    "type": "relay" if c.get("type") == "relay" else "led",
                    "pin": pin if 0 <= pin <= 60 or 100 <= pin <= 115 else -1})  # 100..115 = kênh CH0..CH15 của PCA9685
    return out  # pin -1 = dùng chân mặc định khi tạo code (with_default_pins)


DEFAULT_PINS = {  # chân mặc định cho kênh chưa chọn chân (LED cần chân PWM)
    "usb": {"led": [9, 10, 11, 6, 5, 3], "relay": [7, 8, 4, 12, 2, 13]},
    "esp32": {"led": [25, 26, 27, 32, 33, 14, 13, 4], "relay": [26, 27, 32, 33, 14, 13, 4, 16]},
    "esp8266": {"led": [13, 12, 14, 15, 4, 5], "relay": [12, 13, 14, 15, 4, 5]},  # GPIO: D7 D6 D5 D8 D2 D1
}


def with_default_pins(chs: list[dict], kind: str) -> list[dict]:
    pool = DEFAULT_PINS.get(kind, DEFAULT_PINS["usb"])
    used = {c["pin"] for c in chs if c["pin"] >= 0}
    if any(c["pin"] >= 100 for c in chs):  # mạch dùng PCA9685: kênh chưa chọn -> CH trống tiếp theo
        pool = {"led": list(range(100, 116)), "relay": list(range(100, 116))}
    out = []
    for c in chs:
        c = dict(c)
        if c["pin"] < 0:
            free = [p for p in pool[c["type"]] + pool["led" if c["type"] == "relay" else "relay"] if p not in used]
            c["pin"] = free[0] if free else -1
            used.add(c["pin"])
        out.append(c)
    return out


def channels_code(chs: list[dict], kind: str = "usb") -> str:
    """Khối C khai báo kênh đèn cho code nạp mạch."""
    chs = with_default_pins(chs, kind)
    if not chs:
        return "#define N_CH 0   // mạch này chưa khai báo kênh đèn (thêm ở dashboard: thẻ mạch -> Số đèn / kênh)"
    pins = ", ".join(str(c["pin"]) if c["pin"] >= 0 else "0" for c in chs)
    pwm = ", ".join("1" if c["type"] == "led" else "0" for c in chs)
    rows = []
    for i, c in enumerate(chs):
        nm = re.sub(r"[^\w .\-]", "", c["name"])
        kind_vi = "LED chỉnh độ sáng" if c["type"] == "led" else "relay bật/tắt"
        where = f"PCA9685 CH{c['pin'] - 100}" if c["pin"] >= 100 else f"chân {c['pin']}"
        rows.append(f"//   ch{i + 1}: {nm} — {kind_vi}, {where}")
    names = "\n".join(rows)
    pca = 1 if any(c["pin"] >= 100 for c in chs) else 0
    return (f"#define N_CH {len(chs)}\n{names}\nconst uint8_t CH_PIN[N_CH] = {{{pins}}};   // sửa nếu cắm dây khác (100 = PCA9685 CH0, 101 = CH1...)\n"
            f"const uint8_t CH_PWM[N_CH] = {{{pwm}}};   // 1 = LED chỉnh độ sáng (chân có dấu ~), 0 = relay bật/tắt\n"
            f"#define USE_PCA9685 {pca}   // 1 = đèn nối qua mạch PCA9685 (I2C: SDA-A4, SCL-A5 trên Uno)")


def clean_sensor_data(d) -> dict:
    """JSON từ mạch -> {tên: số}. Bỏ khoá lạ, giá trị không phải số / NaN / ngoài dải."""
    import math
    out: dict = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if len(out) >= 16 or not isinstance(k, str) or not SENSOR_KEY_RE.match(k):
            continue
        if isinstance(v, bool):
            v = int(v)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        lo, hi = SENSOR_RANGES.get(k, (-1e6, 1e6))
        if lo <= v <= hi:
            out[k] = round(float(v), 2)
    return out
NO_STORE = {"Cache-Control": "no-store"}
LOCAL = {"127.0.0.1", "localhost", "::1"}
COOKIE = "ares_srv"


class LoginIn(BaseModel):
    user: str = Field(max_length=64)
    password: str = Field(max_length=256)


class AccountIn(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=128)
    name: str = Field("", max_length=80)
    role: str = Field("viewer", pattern=r"^(admin|viewer)$")


class UserUpd(BaseModel):
    name: str | None = Field(None, max_length=80)
    role: str | None = Field(None, pattern=r"^(admin|viewer)$")
    status: str | None = Field(None, pattern=r"^(active|pending|disabled)$")


class PasswordIn(BaseModel):
    password: str = Field(min_length=1, max_length=128)
    old: str = Field("", max_length=128)


class SiteIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    address: str = Field("", max_length=300)
    note: str = Field("", max_length=500)


class SiteUpd(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    address: str | None = Field(None, max_length=300)
    note: str | None = Field(None, max_length=500)
    active: bool | None = None


class CameraIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    source: str = Field(min_length=1, max_length=300)


SOURCE_RE = re.compile(r"^(\d{1,2}|rtsps?://\S{3,290}|https?://\S{3,290})$")  # USB số | RTSP | HTTP(S) MJPEG/HLS
VIEWER = Path(__file__).parent / "viewer"  # bản build của app xem (ARES Viewer) cho máy khác


class MyCamIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    place_id: int | None = None


class PlaceIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    user_id: int | None = None  # quản trị viên tạo địa điểm cho người khác


class PlaceUpd(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class RetrainIn(BaseModel):
    target: str = Field("behavior", pattern=r"^(behavior|person|hazards|all)$")
    epochs: int = Field(40, ge=5, le=300)


class AssignIn(BaseModel):
    site_id: int
    edge_id: str = Field(max_length=40)
    cam_id: str = Field(max_length=16)
    place_id: int


class DeviceIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    kind: str = Field(pattern=r"^(esp32|esp8266|usb|other)$")
    place_id: int | None = None
    channels: list | None = None  # kênh đèn: [{"name":"Đèn bàn trái","type":"led"|"relay","pin":9}]


class DeviceUpd(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=60)
    place_id: int | None = None
    channels: list | None = None


class MeIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class VerdictIn(BaseModel):
    verdict: str = Field("", pattern=r"^(true|false|)$")


MAC_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>ARES Viewer</string><key>CFBundleExecutable</key><string>ARES Viewer</string>
<key>CFBundleIdentifier</key><string>vn.ueh.iscm.ares.viewer</string><key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.2.0</string><key>LSMinimumSystemVersion</key><string>11.0</string>
</dict></plist>
"""


def _err(code: int, msg: str) -> JSONResponse:
    return JSONResponse({"detail": msg}, status_code=code)


def lan_ips() -> list[str]:
    """Địa chỉ IPv4 trong mạng nội bộ của máy chủ (để hiện link cho máy khác)."""
    import socket

    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # không gửi gói nào, chỉ để hệ điều hành chọn card mạng chính
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return [ip for ip in ips if not ip.startswith("127.")]


def check_exposure(cfg: ServerSettings) -> None:
    """Mọi truy cập dashboard đều cần tài khoản (lưu trong DB server). Tài khoản quản trị ĐẦU TIÊN chỉ tạo được từ
    chính máy chủ (127.0.0.1), nên mở ra mạng khi chưa có tài khoản vẫn an toàn: máy khác không chiếm quyền được."""
    return None


def create_app(cfg: ServerSettings | None = None, db: ServerDB | None = None) -> FastAPI:
    cfg = cfg or ServerSettings()
    cfg.ensure_dirs()
    db = db or ServerDB(cfg.data_dir / "server.db")
    db.fail_stale_uploads()  # video đang tải lên dở lúc server tắt: báo lỗi rõ thay vì treo
    media = cfg.data_dir / "media"
    secret = cfg.session_secret()
    stop = threading.Event()
    from ares_server.zones import ZoneEngine
    zones = ZoneEngine(db)  # đèn theo vùng có người cho camera thật

    def zone_ticker():
        while not stop.wait(1.0):
            try:
                zones.tick()
            except Exception:  # noqa: BLE001
                log.exception("Điều khiển đèn theo vùng lỗi")

    def housekeeping():
        while not stop.wait(3600):
            try:
                n = db.cleanup(media, cfg.media_keep_days, cfg.metrics_keep_days)
                if n:
                    log.info("Đã xoá ảnh/clip của %d sự kiện cũ", n)
            except Exception:  # noqa: BLE001
                log.exception("Dọn dữ liệu cũ lỗi")

    def discovery():
        """Trả lời gói UDP 'ARES_DISCOVER' trong mạng nội bộ để app ARES Viewer tự tìm máy chủ khi đổi IP."""
        import socket

        try:
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sk.bind(("0.0.0.0", cfg.discovery_port))
            sk.settimeout(1.0)
        except OSError as e:
            log.warning("Không mở được cổng dò tìm UDP %d: %s", cfg.discovery_port, e)
            return
        reply = json.dumps({"ares": True, "port": cfg.port, "version": __version__}).encode()
        while not stop.is_set():
            try:
                data, addr = sk.recvfrom(256)
                if data.strip() == b"ARES_DISCOVER":
                    sk.sendto(reply, addr)
            except socket.timeout:
                continue
            except OSError:
                break
        sk.close()

    @asynccontextmanager
    async def lifespan(app):
        threading.Thread(target=zone_ticker, name="ares-server-zones", daemon=True).start()
        th = threading.Thread(target=housekeeping, name="ares-server-housekeep", daemon=True)
        th.start()
        if cfg.host not in LOCAL and cfg.discovery_port:
            threading.Thread(target=discovery, name="ares-server-discovery", daemon=True).start()
        yield
        stop.set()

    app = FastAPI(title="ARES Server", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.db, app.state.cfg, app.state.zones = db, cfg, zones
    live: dict[tuple[int, str, str], tuple[bytes, float]] = {}  # ảnh trực tiếp mới nhất, chỉ giữ trong RAM
    app.state.live = live
    client_frames: dict[str, tuple[bytes, float]] = {}  # ảnh webcam máy khác gửi lên, chờ Edge kéo về chạy AI (RAM)
    app.state.client_frames = client_frames

    # ================================================================== phiên đăng nhập
    # Tài khoản lưu trong DB (bảng users, mật khẩu băm PBKDF2). Bản cũ cấu hình mật khẩu trong server.env: tự chuyển
    # thành tài khoản quản trị ở lần chạy đầu, sau đó quản lý trên dashboard (Thiết lập > Tài khoản).
    if db.count_users() == 0 and cfg.admin_password:
        try:
            db.create_user(cfg.admin_user, cfg.admin_password, "Quản trị viên", "admin", note="từ server.env")
        except ValueError as e:
            log.warning("Không tạo được tài khoản từ ARES_SERVER_ADMIN_PASSWORD: %s", e)

    PUBLIC = ("/api/login", "/api/me", "/api/app-info", "/api/setup", "/api/register")

    def sign(uid: int, exp: float) -> str:
        body = base64.urlsafe_b64encode(json.dumps({"i": uid, "e": int(exp)}).encode()).decode()
        mac = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()
        return f"{body}.{mac}"

    def session_token(request: Request) -> str:
        """Cookie (dashboard mở từ chính máy chủ) hoặc token (dashboard trên GitHub Pages: header Authorization,
        hoặc ?t= cho <img>/<video> vì thẻ ảnh không gửi được header)."""
        tok = request.cookies.get(COOKIE, "")
        if not tok and not request.url.path.startswith("/api/v1/"):
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                tok = auth[7:].strip()
            elif request.method in ("GET", "HEAD"):
                tok = request.query_params.get("t", "")
        return tok

    def current_user(request: Request) -> dict | None:
        tok = session_token(request)
        body, _, mac = tok.partition(".")
        if not body or not hmac.compare_digest(mac, hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()):
            return None
        try:
            d = json.loads(base64.urlsafe_b64decode(body.encode()))
        except Exception:  # noqa: BLE001
            return None
        if d.get("e", 0) <= time.time():
            return None
        u = db.user(int(d.get("i", 0)))  # đọc lại mỗi lần: khoá/xoá/đổi quyền có hiệu lực ngay
        return u if u and u["status"] == "active" else None

    def session_user(request: Request) -> str | None:
        u = getattr(request.state, "user", None) or current_user(request)
        return u["username"] if u else None

    def proxied(request: Request) -> bool:
        """Đi qua đường hầm (Tailscale Funnel / Cloudflare): kết nối tới từ 127.0.0.1 nhưng KHÔNG phải người ngồi ở máy chủ."""
        h = request.headers
        return bool(h.get("x-forwarded-for") or h.get("tailscale-funnel-request") or h.get("cf-connecting-ip")
                    or h.get("forwarded"))

    def from_server_machine(request: Request) -> bool:
        return bool(request.client) and request.client.host in ("127.0.0.1", "::1") and not proxied(request)

    def client_ip(request: Request) -> str:
        ip = request.client.host if request.client else "?"
        if ip in ("127.0.0.1", "::1") and proxied(request):  # sau đường hầm: lấy IP thật để chặn đoán mật khẩu theo từng người
            ip = (request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0]).strip() or ip
        return ip

    def admin_only(path: str, method: str) -> bool:
        if path.startswith(("/api/users", "/api/admin/")) or path == "/api/sites" or path.startswith("/api/sites/"):
            return True  # người dùng app chỉ thấy camera của mình, không thấy danh sách địa điểm/người dùng
        return False

    @app.middleware("http")
    async def guard(request: Request, call_next):
        p = request.url.path
        if p == "/traffic" or p.startswith("/traffic/"):  # gói Giao thông gắn vào server: chỉ quản trị viên
            u = current_user(request)
            if u is None and request.method == "GET" and request.query_params.get("t"):
                return _err(401, "Phiên hết hạn — đăng nhập lại")
            if u is None or u["role"] != "admin":
                if p.startswith("/traffic/api/"):
                    return _err(401 if u is None else 403, "Cần đăng nhập quản trị viên")
                from fastapi.responses import HTMLResponse
                return HTMLResponse("<meta charset='utf-8'><p style='font:15px system-ui;padding:20px'>Phần <b>Giao thông</b> "
                                    "chỉ dành cho quản trị viên. <a href='/'>Đăng nhập</a></p>", status_code=401)
            if request.query_params.get("t") and not request.cookies.get(COOKIE):  # mở từ dashboard GitHub Pages: đặt cookie
                from fastapi.responses import RedirectResponse
                r = RedirectResponse(p, status_code=303)
                r.set_cookie(COOKIE, request.query_params["t"], httponly=True, samesite="lax", max_age=12 * 3600,
                             secure=request.url.scheme == "https")
                return r
            request.state.user = u
            return await call_next(request)
        public = p in PUBLIC and not (p == "/api/me" and request.method != "GET")  # GET /api/me công khai, POST thì không
        if p.startswith("/api/") and not p.startswith("/api/v1/") and not public:
            u = current_user(request)
            if u is None:
                return _err(401, "Cần đăng nhập")
            if admin_only(p, request.method) and u["role"] != "admin":
                return _err(403, "Chỉ quản trị viên được làm việc này")
            if request.method not in ("GET", "HEAD") and request.headers.get("x-ares") != "1":
                return _err(403, "Thiếu header X-ARES: 1")
            request.state.user = u
        elif p in ("/api/setup", "/api/register", "/api/login") and request.headers.get("x-ares") != "1":
            return _err(403, "Thiếu header X-ARES: 1")
        return await call_next(request)

    from fastapi.middleware.cors import CORSMiddleware
    # thêm SAU guard => CORS bọc ngoài cùng: cả câu trả lời 401/403 của guard cũng có header CORS, preflight OPTIONS không bị chặn
    app.add_middleware(  # dashboard trên GitHub Pages gọi về máy chủ; không gửi cookie (allow_credentials=False) -> chỉ token
        CORSMiddleware, allow_origins=[o.strip().rstrip("/") for o in cfg.cors_origins.split(",") if o.strip()],
        allow_origin_regex=r"https://[A-Za-z0-9\-]+\.github\.io", allow_credentials=False,
        allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type", "X-ARES"], max_age=3600)

    def edge_auth(request: Request) -> tuple[dict, str] | JSONResponse:
        auth = request.headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        site = db.site_by_key(key) if key else None
        if site is None:
            return _err(401, "Khoá thiết bị không hợp lệ hoặc địa điểm đã bị tắt")
        edge_id = request.headers.get("x-edge-id", "edge")
        if not EDGE_RE.match(edge_id):
            return _err(400, "X-Edge-Id không hợp lệ")
        return site, edge_id

    # ================================================================== API cho Edge
    @app.get("/api/v1/ping")
    def v1_ping(request: Request):
        a = edge_auth(request)
        if isinstance(a, JSONResponse):
            return a
        return {"ok": True, "site": a[0]["name"], "server_version": __version__, "time": time.time()}

    @app.post("/api/v1/heartbeat")
    async def v1_heartbeat(request: Request):
        a = edge_auth(request)
        if isinstance(a, JSONResponse):
            return a
        raw = await request.body()
        if len(raw) > 1_000_000:
            return _err(413, "Heartbeat quá lớn")
        try:
            payload = json.loads(raw)
            assert isinstance(payload, dict)
        except Exception:  # noqa: BLE001
            return _err(400, "JSON không hợp lệ")
        db.heartbeat(a[0]["id"], a[1], payload, request.client.host if request.client else "")
        if isinstance(payload.get("command_results"), list):
            db.save_results(a[0]["id"], a[1], payload["command_results"])
        return {"ok": True, "time": time.time(), "updates": db.pop_updates(a[0]["id"], a[1]),
                "commands": db.pop_commands(a[0]["id"], a[1]), "ppe_rules": ppe_rules_for(a[0]["id"], a[1]),
                "zone_layouts": zones.zones_for(a[0]["id"], a[1])}

    @app.post("/api/v1/zones")
    async def v1_zones(request: Request):
        """Máy AI gửi số người trong từng vùng của từng camera (~1 lần/giây) -> server bật/tắt đèn theo vùng."""
        a = edge_auth(request)
        if isinstance(a, JSONResponse):
            return a
        try:
            d = json.loads(await request.body() or b"{}")
        except ValueError:
            return _err(400, "JSON không hợp lệ")
        now = time.time()
        for cam, v in list((d.get("cams") or {}).items())[:64] if isinstance(d.get("cams"), dict) else []:
            if isinstance(v, dict) and isinstance(v.get("counts"), dict):
                zones.feed((a[0]["id"], a[1], str(cam)[:20]), v["counts"], int(v.get("n") or 0), now)
        return {"ok": True, "layouts": zones.zones_for(a[0]["id"], a[1])}

    # ------------------------------------------------------------------ thử video: máy AI nhận job, tải video, gửi kết quả
    vdir = media / "videos"

    def _job_dir(jid: int) -> Path:
        d = vdir / str(int(jid))
        d.mkdir(parents=True, exist_ok=True)
        return d

    # Video "Xong" mà 0 khung (lỗi cũ: máy AI nhận file khi chưa tải lên xong): file gốc giờ đã đủ -> tự đưa vào hàng chờ lại.
    for _j in db.video_jobs(None, 500):
        _src = vdir / str(_j["id"]) / f"source{_j['ext']}"
        if _j["status"] == "done" and not _j["n_frames"] and _src.exists() and _src.stat().st_size >= 1024:
            db.update_video_job(_j["id"], status="queued", progress=0.0, result="", has_mp4=0, size=_src.stat().st_size,
                                started=None, finished=None, message="Đang chờ máy AI (tự phân tích lại sau khi sửa lỗi)")

    def _edge_job(request: Request, jid: int):
        a = edge_auth(request)
        if isinstance(a, JSONResponse):
            return a, None
        j = db.video_job(jid)
        if j is None or j["site_id"] != a[0]["id"] or j["edge_id"] != a[1] or j["status"] != "running":
            return _err(404, "Không có job này (có thể đã bị huỷ)"), None
        return None, j

    @app.post("/api/v1/jobs/next")
    def v1_job_next(request: Request):
        a = edge_auth(request)
        if isinstance(a, JSONResponse):
            return a
        j = db.claim_video_job(a[0]["id"], a[1])
        if j is None:
            return Response(status_code=204)
        try:
            rules = clean_ppe(json.loads(j.get("ppe") or "{}"))
        except ValueError:
            rules = clean_ppe(None)
        return {"id": j["id"], "name": j["name"], "ext": j["ext"], "size": j["size"], "ppe": rules,
                "kind": j.get("kind") or "behavior", "opts": j.get("opts") or {}}

    @app.get("/api/v1/jobs/{jid}/video")
    def v1_job_video(jid: int, request: Request):
        e, j = _edge_job(request, jid)
        if e is not None:
            return e
        p = _job_dir(jid) / f"source{j['ext']}"
        return FileResponse(p) if p.exists() else _err(404, "Mất file video")

    @app.post("/api/v1/jobs/{jid}/progress")
    async def v1_job_progress(jid: int, request: Request):
        e, j = _edge_job(request, jid)
        if e is not None:
            return e
        try:
            d = json.loads(await request.body() or b"{}")
        except ValueError:
            return _err(400, "JSON không hợp lệ")
        if d.get("error"):
            db.update_video_job(jid, status="error", message=str(d["error"])[:300], finished=time.time())
            return {"ok": True}
        if d.get("release"):  # máy AI đang tắt / khởi động lại giữa chừng: trả job về hàng chờ để chạy lại ngay sau đó
            db.update_video_job(jid, status="queued", progress=0.0, message="Máy AI khởi động lại — đang chờ phân tích lại")
            return {"ok": True}
        db.update_video_job(jid, progress=max(0.0, min(1.0, float(d.get("progress") or 0))),
                            message=str(d.get("message") or "Đang xử lý")[:200])
        return {"ok": True, "cancel": False}

    @app.post("/api/v1/jobs/{jid}/result")
    async def v1_job_result(jid: int, request: Request):
        e, j = _edge_job(request, jid)
        if e is not None:
            return e
        form = await request.form()
        try:
            meta = json.loads(form.get("meta") or "{}")
        except ValueError:
            return _err(400, "meta không hợp lệ")
        d = _job_dir(jid)
        frz, mp4 = form.get("frames"), form.get("video")
        n_frames, has_mp4 = 0, 0
        if frz is not None and hasattr(frz, "read"):
            data = await frz.read()
            if len(data) > MAX_RESULT_BYTES or not data.startswith(b"PK"):
                return _err(415, "frames phải là file zip")
            (d / "frames.zip").write_bytes(data)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                n_frames = len([n for n in zf.namelist() if n.endswith(".jpg")])
        if mp4 is not None and hasattr(mp4, "read"):
            data = await mp4.read()
            if 0 < len(data) <= MAX_RESULT_BYTES:
                (d / "result.mp4").write_bytes(data)
                has_mp4 = 1
        ft, tk = meta.get("frame_t"), meta.get("tracks")
        if isinstance(ft, list) and isinstance(tk, list) and 0 < len(ft) == len(tk) <= 40000:
            try:
                clean_t = [round(float(x), 3) for x in ft]
                clean_k = [[[int(b[0]), *[round(float(v), 4) for v in b[1:5]], int(b[5])] for b in fr[:200]
                            if isinstance(b, list) and len(b) == 6] for fr in tk]
                (d / "tracks.json").write_text(json.dumps({"t": clean_t, "b": clean_k}, separators=(",", ":")), encoding="utf-8")
            except (TypeError, ValueError):
                pass
        keep = {k: meta.get(k) for k in ("events", "fps", "duration", "people_max", "people_series", "processing_s", "frames",
                                         "device", "mode", "posture_s", "ppe", "ppe_note", "kind", "layout", "light_runs",
                                         "lights_stat", "zones_stat", "energy_saved_pct", "track_stats")}
        keep = {k: v for k, v in keep.items() if v is not None}
        for k in ("light_runs",):  # chặn kết quả quá lớn
            if isinstance(keep.get(k), dict):
                keep[k] = {str(a)[:12]: [[int(x[0]), int(x[1])] for x in (b or [])[:20000] if isinstance(x, list) and len(x) == 2]
                           for a, b in list(keep[k].items())[:24]}
        keep["status_runs"] = [[int(r[0]), int(r[1]), str(r[2])[:200]] for r in (meta.get("status_runs") or [])[:5000]
                               if isinstance(r, list) and len(r) == 3]
        keep["events"] = (keep.get("events") or [])[:500]
        keep.setdefault("people_series", [])
        keep["people_series"] = (keep.get("people_series") or [])[:7200]
        if not n_frames:  # máy AI (kể cả bản cũ) không đọc được khung nào: đây là LỖI, không phải "Xong, không có sự kiện"
            db.update_video_job(jid, status="error", finished=time.time(), n_frames=0, has_mp4=0,
                                message="Máy AI không đọc được khung hình nào từ video (file hỏng hoặc định dạng lạ). "
                                        "Bấm Phân tích lại; vẫn lỗi thì đổi video sang MP4 (H.264) rồi tải lại.")
            return {"ok": True}
        db.update_video_job(jid, status="done", progress=1.0, finished=time.time(), result=keep, n_frames=n_frames,
                            has_mp4=has_mp4, message=f"Xong — {len(keep.get('events') or [])} sự kiện")
        return {"ok": True}

    # ------------------------------------------------------------------ mạch cảm biến gửi số đo (ESP32/ESP8266 qua Wi-Fi)
    def _ingest(dev: dict, data, via: str, request: Request) -> JSONResponse:
        now = time.time()
        if dev["last_seen"] and now - dev["last_seen"] < 0.5:  # gửi dồn dập: bỏ bớt, vẫn trả lệnh
            return JSONResponse({"ok": True, "cmd": dev["desired"], "skipped": True})
        clean = clean_sensor_data(data)
        db.device_data(dev["id"], clean, via, client_ip(request))
        return JSONResponse({"ok": True, "cmd": dev["desired"], "interval": 5, "saved": len(clean)})

    @app.post("/api/v1/sensor")
    async def v1_sensor(request: Request):
        try:
            body = json.loads(await request.body() or b"{}")
        except ValueError:
            return _err(400, "JSON không hợp lệ")
        tok = request.headers.get("x-device-token") or request.query_params.get("token") or \
            (body.pop("token", "") if isinstance(body, dict) else "")
        dev = db.device_by_token(str(tok))
        if dev is None:
            return _err(401, "Sai khoá thiết bị — tải lại code từ dashboard")
        return _ingest(dev, body, "wifi", request)

    @app.post("/api/v1/live")
    async def v1_live(request: Request):
        a = edge_auth(request)
        if isinstance(a, JSONResponse):
            return a
        form = await request.form()
        now, n = time.time(), 0
        for key, up in list(form.items())[:20]:
            if not key.startswith("cam_") or not hasattr(up, "read"):
                continue
            cam = key[4:]
            if not re.fullmatch(r"[A-Za-z0-9]{1,16}", cam):
                continue
            data = await up.read(700_001)
            if len(data) > 700_000 or not data.startswith(b"\xff\xd8\xff"):
                continue
            live[(a[0]["id"], a[1], cam)] = (data, now)
            n += 1
        return {"ok": True, "n": n}

    async def _read(up: UploadFile | None, magic: tuple[bytes, ...], limit: int) -> bytes:
        if up is None:
            return b""
        data = await up.read(limit + 1)
        if len(data) > limit:
            raise ValueError("File quá lớn")
        if not any(data[off:off + len(m)] == m for m in magic for off in (0, 4)):
            raise ValueError("Định dạng file không đúng")
        return data

    @app.post("/api/v1/events")
    async def v1_event(request: Request, meta: str = Form(...), snapshot: UploadFile | None = File(None),
                       clip: UploadFile | None = File(None), frames: UploadFile | None = File(None)):
        a = edge_auth(request)
        if isinstance(a, JSONResponse):
            return a
        site, edge_id = a
        try:
            ev = json.loads(meta)
            assert isinstance(ev, dict) and EVID_RE.match(str(ev.get("id", ""))) and float(ev.get("t", 0)) > 0
        except Exception:  # noqa: BLE001
            return _err(400, "meta không hợp lệ (cần id, t)")
        limit = int(cfg.max_upload_mb * 1024 * 1024)
        try:  # kiểm tra hết file TRƯỚC khi ghi DB (gói hỏng không để lại sự kiện dở dang)
            snap = await _read(snapshot, (b"\xff\xd8\xff",), limit)
            vid = await _read(clip, (b"ftyp",), limit)
            frz = await _read(frames, (b"PK\x03\x04",), limit)
            nfr = 0
            if frz:
                with zipfile.ZipFile(io.BytesIO(frz)) as zf:
                    nfr = sum(1 for n in zf.namelist() if re.fullmatch(r"\d{4}\.jpg", n))
        except (ValueError, zipfile.BadZipFile) as e:
            return _err(413 if "lớn" in str(e) else 415, str(e))
        eid, new = db.add_event(site["id"], edge_id, ev)
        if new:
            month = datetime.fromtimestamp(float(ev["t"])).strftime("%Y%m")
            folder = media / str(site["id"]) / month
            folder.mkdir(parents=True, exist_ok=True)
            rel = {}
            for name, data, ext in (("snapshot", snap, "snap.jpg"), ("clip", vid, "clip.mp4"), ("frames", frz, "frames.zip")):
                if data:
                    (folder / f"{eid}_{ext}").write_bytes(data)
                    rel[name] = (folder / f"{eid}_{ext}").relative_to(media).as_posix()
            db.set_event_media(eid, rel.get("snapshot", ""), rel.get("clip", ""), rel.get("frames", ""), nfr)
        return {"ok": True, "id": eid, "new": new}

    # ================================================================== đăng nhập dashboard
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html; charset=utf-8", headers=NO_STORE)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__}

    @app.get("/api/me")
    def me(request: Request):
        u = current_user(request)
        need_setup = db.count_users() == 0
        return {"user": None if u is None else {k: u[k] for k in ("id", "username", "name", "role")},
                "need_setup": need_setup, "can_setup": need_setup and from_server_machine(request),
                "pending": db.count_users("pending") if u and u["role"] == "admin" else 0, "version": __version__}

    _fails: dict[str, list[float]] = {}

    def _throttle(key: str, limit: int, window: float) -> bool:
        now = time.time()
        recent = [t for t in _fails.get(key, []) if now - t < window]
        _fails[key] = recent
        return len(recent) >= limit

    def _login_response(u: dict, request: Request) -> JSONResponse:
        exp = time.time() + cfg.session_hours * 3600
        tok = sign(u["id"], exp)
        # token trả kèm cho dashboard đặt ở trang khác (GitHub Pages) — trang đó tự lưu, gửi lại qua header Authorization
        resp = JSONResponse({"ok": True, "user": {k: u[k] for k in ("id", "username", "name", "role")}, "token": tok})
        resp.set_cookie(COOKIE, tok, httponly=True, samesite="strict",
                        max_age=int(cfg.session_hours * 3600), secure=request.url.scheme == "https")
        return resp

    @app.post("/api/login")
    def login(body: LoginIn, request: Request):
        ip = client_ip(request)
        if _throttle("login:" + ip, 8, 300):
            return _err(429, "Sai quá nhiều lần — thử lại sau 5 phút")
        u, why = db.authenticate(body.user, body.password)
        if u is None:
            _fails.setdefault("login:" + ip, []).append(time.time())
            return _err(401, why)
        _fails.pop("login:" + ip, None)
        return _login_response(u, request)

    @app.post("/api/setup")
    def setup(body: AccountIn, request: Request):
        """Tạo tài khoản quản trị đầu tiên. Chỉ khi DB chưa có tài khoản nào VÀ yêu cầu đến từ chính máy chủ."""
        if db.count_users() > 0:
            return _err(409, "Đã có tài khoản — hãy đăng nhập")
        if not from_server_machine(request):
            return _err(403, "Tài khoản quản trị đầu tiên chỉ tạo được trên chính máy chủ: mở http://127.0.0.1:"
                             f"{cfg.port} trên máy chủ.")
        try:
            db.create_user(body.username, body.password, body.name or "Quản trị viên", "admin")
        except ValueError as e:
            return _err(400, str(e))
        u, _ = db.authenticate(body.username, body.password)
        return _login_response(u, request)

    @app.post("/api/register")
    def register(body: AccountIn, request: Request):
        """Tự đăng ký: tạo tài khoản 'chờ duyệt' (vai trò Xem). Quản trị viên duyệt trong Thiết lập > Tài khoản."""
        if db.count_users() == 0:
            return _err(409, "Máy chủ chưa có quản trị viên — tạo tài khoản quản trị trên chính máy chủ trước")
        ip = client_ip(request)
        if _throttle("reg:" + ip, 5, 3600):
            return _err(429, "Đăng ký quá nhiều lần — thử lại sau")
        try:
            db.create_user(body.username, body.password, body.name, "viewer", "pending", note=f"tự đăng ký từ {ip}")
        except ValueError as e:
            return _err(400, str(e))
        _fails.setdefault("reg:" + ip, []).append(time.time())
        return {"ok": True, "message": "Đã gửi yêu cầu. Quản trị viên duyệt xong là đăng nhập được."}

    # ------------------------------------------------------------------ quản lý tài khoản (quản trị viên)
    @app.get("/api/users")
    def users_list():
        items = db.users()
        pv = places_view()
        for u in items:
            mine = [p for p in pv if p["user_id"] == u["id"]]
            u["places"] = [{"id": p["id"], "name": p["name"], "cams_online": p["cams_online"], "people": p["people"]}
                           for p in mine]
            u["n_places"] = len(mine)
            u["cams_online"] = sum(p["cams_online"] for p in mine)
            u["people"] = sum(p["people"] for p in mine)
            u["new_alerts"] = sum(p["new_alerts"] for p in mine)
            fr = [client_frames[c["token"]][1] for c in db.client_cams(u["id"]) if c["token"] in client_frames]
            u["last_frame"] = max(fr, default=0) or None
        return JSONResponse({"items": items}, headers=NO_STORE)

    @app.post("/api/users")
    def users_create(body: AccountIn):
        try:
            return {"ok": True, "id": db.create_user(body.username, body.password, body.name, body.role)}
        except ValueError as e:
            return _err(400, str(e))

    @app.post("/api/users/{uid}")
    def users_update(uid: int, body: UserUpd, request: Request):
        me_ = request.state.user
        if uid == me_["id"] and (body.status not in (None, "active") or body.role not in (None, "admin")):
            return _err(400, "Không tự hạ quyền hoặc tự khoá chính mình")
        try:
            db.update_user(uid, **body.model_dump())
        except KeyError:
            return _err(404, "Không có tài khoản này")
        except ValueError as e:
            return _err(400, str(e))
        return {"ok": True}

    @app.post("/api/users/{uid}/password")
    def users_password(uid: int, body: PasswordIn):
        try:
            db.set_password(uid, body.password)
        except KeyError:
            return _err(404, "Không có tài khoản này")
        except ValueError as e:
            return _err(400, str(e))
        return {"ok": True}

    @app.post("/api/users/{uid}/delete")
    def users_delete(uid: int, request: Request):
        if uid == request.state.user["id"]:
            return _err(400, "Không tự xoá tài khoản đang đăng nhập")
        try:
            db.delete_user(uid)
            import shutil
            for j in db.video_jobs(uid, 1000):
                db.delete_video_job(j["id"])
                shutil.rmtree(vdir / str(j["id"]), ignore_errors=True)
            for c in db.client_cams(uid):
                db.add_command(c["site_id"], c["edge_id"], {"type": "remove_camera", "cam_id": c["cam_id"]},
                               session_user(request) or "?")
                db.delete_client_cam(c["id"])
        except KeyError:
            return _err(404, "Không có tài khoản này")
        except ValueError as e:
            return _err(400, str(e))
        return {"ok": True}

    @app.post("/api/me/password")
    def my_password(body: PasswordIn, request: Request):
        u = request.state.user
        ok, _ = db.authenticate(u["username"], body.old)
        if not ok:
            return _err(400, "Mật khẩu hiện tại không đúng")
        try:
            db.set_password(u["id"], body.password)
        except ValueError as e:
            return _err(400, str(e))
        return {"ok": True}

    @app.post("/api/logout")
    def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE)
        return resp

    # ================================================================== tổng quan
    def _behavior_state(edges: list[dict]) -> str:
        """running nếu có máy chạy được; error nếu có máy lỗi; loading; off nếu không máy nào bật AI hành vi."""
        sts = [((e["info"].get("ai") or {}).get("behavior") or None) for e in edges]
        sts = [b for b in sts if b]
        if any(b.get("running") for b in sts):
            return "running"
        if any(b.get("error") for b in sts):
            return "error"
        if any(b.get("loading") for b in sts):
            return "loading"
        return "off"

    def _site_cards() -> list[dict]:
        now = time.time()
        edges, cams = db.edges(), db.cameras()
        new_by_site: dict[int, int] = {}
        for e in db.events(limit=1000, status="new"):
            new_by_site[e["site_id"]] = new_by_site.get(e["site_id"], 0) + 1
        out = []
        for s in db.sites():
            se = [e for e in edges if e["site_id"] == s["id"]]
            online_edges = [e for e in se if now - (e["last_seen"] or 0) <= cfg.offline_after_s]
            on_ids = {e["edge_id"] for e in online_edges}
            sc = [c for c in cams if c["site_id"] == s["id"]]
            live = [c for c in sc if c["edge_id"] in on_ids and c["online"]]
            integ = {}
            for e in online_edges:
                for k, v in (e["info"].get("integrations") or {}).items():
                    if isinstance(v, dict) and v.get("enabled"):
                        integ[k] = v
            out.append({**s, "edges_total": len(se), "edges_online": len(online_edges),
                        "cams_total": len(sc), "cams_online": len(live), "people": sum(c["n_people"] for c in live),
                        "new_alerts": new_by_site.get(s["id"], 0),
                        "last_seen": max((e["last_seen"] or 0 for e in se), default=None),
                        "behavior_state": _behavior_state(online_edges),
                        "integrations": integ})
        return out

    # ================================================================== phạm vi xem theo người dùng
    def place_of_cams() -> dict[tuple, int]:
        """(site, edge, cam) -> id địa điểm. Camera chưa gán thuộc địa điểm đầu tiên của quản trị viên đầu tiên."""
        m: dict[tuple, int] = {}
        for c in db.client_cams():
            m[(c["site_id"], c["edge_id"], c["cam_id"])] = c["place_id"] or db.default_place(c["user_id"])
        for r in db.cam_places():
            m.setdefault((r["site_id"], r["edge_id"], r["cam_id"]), r["place_id"])
        admins = [u for u in db.users() if u["role"] == "admin" and u["status"] == "active"]
        fallback = db.default_place(min(a["id"] for a in admins)) if admins else None
        for c in db.cameras():
            k = (c["site_id"], c["edge_id"], c["cam_id"])
            if k not in m and fallback is not None:
                m[k] = fallback
        return m

    def place_ppe(p: dict) -> dict:
        try:
            return clean_ppe(json.loads(p.get("ppe") or "{}"))
        except ValueError:
            return clean_ppe(None)

    def ppe_rules_for(site_id: int, edge_id: str) -> dict:
        """Luật đồ bảo hộ cho từng camera của 1 máy AI = luật của địa điểm chứa camera đó. Gửi về trong heartbeat."""
        pinfo = {p["id"]: place_ppe(p) for p in db.places()}
        out = {}
        for (sid, eid, cam), pid in place_of_cams().items():
            if sid == site_id and eid == edge_id:
                out[cam] = pinfo.get(pid) or clean_ppe(None)
        return out

    def user_scope(uid: int) -> list[tuple]:
        mine = {p["id"] for p in db.places(uid)}
        return [k for k, pid in place_of_cams().items() if pid in mine]

    def scope_of(request: Request):
        """None = quản trị viên (xem hết). Người dùng: camera nằm trong các địa điểm của chính họ."""
        u = request.state.user
        if u["role"] == "admin":
            return None
        return user_scope(u["id"])

    def places_view(uid: int | None = None) -> list[dict]:
        """Danh sách địa điểm kèm camera + số liệu riêng từng địa điểm (uid=None: tất cả)."""
        now = time.time()
        edges = {(e["site_id"], e["edge_id"]): e for e in db.edges()}
        def edge_on(sid, eid):
            e = edges.get((sid, eid))
            return bool(e and now - (e["last_seen"] or 0) <= cfg.offline_after_s)
        hw = {(c["site_id"], c["edge_id"], c["cam_id"]): c for c in db.cameras()}
        pmap = place_of_cams()
        clients = {(c["site_id"], c["edge_id"], c["cam_id"]): c for c in db.client_cams()}
        users = {u["id"]: u for u in db.users()}
        new_by_cam: dict[tuple, int] = {}
        for e in db.events(limit=1000, status="new"):
            k = (e["site_id"], e["edge_id"], e["cam_id"])
            new_by_cam[k] = new_by_cam.get(k, 0) + 1
        keys = set(hw) | set(clients)
        devs_by_place: dict = {}
        for dv in db.devices():
            devs_by_place.setdefault(dv["place_id"], []).append(
                {"id": dv["id"], "name": dv["name"], "online": bool(dv["last_seen"] and now - dv["last_seen"] <= DEVICE_ONLINE_S),
                 "data": dv["last_data"]})
        out = []
        for p in (db.places() if uid is None else db.places(uid)):
            cams = []
            for k in sorted(x for x in keys if pmap.get(x) == p["id"]):
                h, cc = hw.get(k, {}), clients.get(k)
                on = edge_on(k[0], k[1]) and bool(h.get("online"))
                fr = client_frames.get(cc["token"]) if cc else None
                lv = live.get(k)
                cams.append({"site_id": k[0], "edge_id": k[1], "cam_id": k[2],
                             "name": (cc["name"] if cc else None) or h.get("name") or k[2],
                             "client_id": cc["id"] if cc else None, "online": on,
                             "n_people": h.get("n_people", 0) if on else 0,
                             "sending": bool(fr and now - fr[1] <= 3) if cc else None,
                             "live": bool(lv and now - lv[1] <= 10), "new_alerts": new_by_cam.get(k, 0)})
            owner = users.get(p["user_id"], {})
            out.append({**p, "owner": owner.get("name") or owner.get("username") or "?",
                        "owner_username": owner.get("username", "?"), "cams": cams,
                        "cams_online": sum(1 for c in cams if c["online"]), "cams_total": len(cams),
                        "people": sum(c["n_people"] for c in cams), "new_alerts": sum(c["new_alerts"] for c in cams),
                        "sensors": devs_by_place.get(p["id"], []), "ppe": place_ppe(p)})
        return out

    DEVICE_ONLINE_S = 60.0

    def ai_status() -> dict:
        """Trạng thái máy chủ AI (máy Edge chạy cùng server): GPU/CPU, AI hành vi."""
        e = ai_edge()
        if e is None:
            return {"online": False}
        ai = e["info"].get("ai") or {}
        beh = ai.get("behavior") or {}
        return {"online": True, "name": e["name"], "device": ai.get("device") or beh.get("device") or "",
                "fallback": beh.get("fallback") or ai.get("fallback") or "",
                "behavior": _behavior_state([e]), "version": e["version"], "ppe": beh.get("ppe")}

    def in_scope(request: Request, site_id: int, edge_id: str, cam_id: str) -> bool:
        sc = scope_of(request)
        return sc is None or (site_id, edge_id, cam_id) in sc

    def ai_edge() -> dict | None:
        """Máy Edge sẽ chạy AI cho camera của máy khác: ưu tiên Edge chạy cùng máy với server, đang trực tuyến."""
        now = time.time()
        on = [e for e in db.edges() if now - (e["last_seen"] or 0) <= cfg.offline_after_s
              and ((e["info"].get("uplink") or {}).get("remote_cameras", True))]
        if not on:
            return None
        on.sort(key=lambda e: (e["ip"] not in ("127.0.0.1", "::1"), -(e["last_seen"] or 0)))
        return on[0]

    def _my_cams(u: dict) -> list[dict]:
        now = time.time()
        cams = {(c["site_id"], c["edge_id"], c["cam_id"]): c for c in db.cameras()}
        edges = {(e["site_id"], e["edge_id"]): e for e in db.edges()}
        out = []
        for c in db.client_cams(u["id"]):
            key = (c["site_id"], c["edge_id"], c["cam_id"])
            hw = cams.get(key, {})
            e = edges.get((c["site_id"], c["edge_id"]))
            edge_on = bool(e and now - (e["last_seen"] or 0) <= cfg.offline_after_s)
            fr = client_frames.get(c["token"])
            lv = live.get(key)
            out.append({"id": c["id"], "name": c["name"], "cam_id": c["cam_id"], "site_id": c["site_id"],
                        "place_id": c["place_id"] or db.default_place(u["id"]),
                        "edge_id": c["edge_id"], "created": c["created"],
                        "sending": bool(fr and now - fr[1] <= 3), "last_frame": fr[1] if fr else None,
                        "ai_online": edge_on and bool(hw.get("online")), "server_ai": edge_on,
                        "n_people": hw.get("n_people", 0) if hw.get("online") else 0,
                        "live": bool(lv and now - lv[1] <= 10)})
        return out

    @app.get("/api/overview")
    def overview(request: Request):
        now = time.time()
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        u = request.state.user
        sc = scope_of(request)
        pl = places_view(None if sc is None else u["id"])
        return JSONResponse({
            "time": now, "places": pl, "ai": ai_status(),
            "totals": {"places": len(pl), "people": sum(p["people"] for p in pl),
                       "cams_online": sum(p["cams_online"] for p in pl),
                       "users": len({p["user_id"] for p in pl}),
                       "new_alerts": len(db.events(500, "new", scope=sc))},
            "today": db.counts_since(today, sc),
            "series": db.series(None, now - 24 * 3600, 300) if sc is None else db.cam_series(sc, now - 24 * 3600),
            # đèn / điều hoà / cảm biến của các máy AI (ARES điều khiển qua MQTT) — chỉ quản trị viên
            "integrations": [] if sc is not None else [
                {"name": e["name"] or e["edge_id"], "online": now - (e["last_seen"] or 0) <= cfg.offline_after_s,
                 **(e["info"].get("integrations") or {})} for e in db.edges()],
        }, headers=NO_STORE)

    # ================================================================== train lại AI từ đánh giá cảnh báo
    def _verdict_counts() -> dict:
        with db._lock:
            rows = db._db.execute("SELECT kind, verdict, COUNT(*) FROM events WHERE verdict IN ('true','false') "
                                  "GROUP BY kind, verdict").fetchall()
        out: dict = {"true": 0, "false": 0, "by_kind": {}}
        for kind, v, n in rows:
            out[v] += n
            out["by_kind"].setdefault(kind, {"true": 0, "false": 0})[v] = n
        return out

    @app.get("/api/admin/retrain")
    def retrain_status():
        e = ai_edge()
        st = (e["info"].get("retrain") if e else None) or {}
        cmds = []
        if e:
            cmds = [c for c in db.commands(e["site_id"], 30) if (c["cmd"] or {}).get("type") == "retrain"][:5]
        clf = ((e["info"].get("ai") or {}).get("behavior") or {}).get("classifier") if e else None
        beh = ((e["info"].get("ai") or {}).get("behavior") or {}) if e else {}
        return JSONResponse({"ai_online": e is not None, "edge": e["name"] if e else "", "state": st,
                             "verdicts": _verdict_counts(), "classifier": clf,
                             "behavior": {"ppe": beh.get("ppe") or {}, "hazards": beh.get("hazards") or {}},
                             "classifier_info": beh.get("classifier_info"),
                             "history": [{"id": c["id"], "created": c["created"], "by": c.get("created_by"),
                                          "target": c["cmd"].get("target"), "done": c.get("done"), "ok": c.get("ok"),
                                          "message": c.get("message")} for c in cmds]}, headers=NO_STORE)

    @app.post("/api/admin/retrain")
    def retrain_start(body: RetrainIn, request: Request):
        e = ai_edge()
        if e is None:
            return _err(409, "Máy chủ AI chưa chạy — mở ARES_MAY_CHU.bat rồi thử lại")
        st = e["info"].get("retrain") or {}
        if st.get("running"):
            return _err(409, "Máy AI đang train — đợi xong")
        pend = [c for c in db.commands(e["site_id"], 10) if (c["cmd"] or {}).get("type") == "retrain" and not c.get("done")]
        if pend:
            return _err(409, "Lệnh train trước chưa được máy AI nhận — đợi khoảng 30 giây")
        cid = db.add_command(e["site_id"], e["edge_id"], {"type": "retrain", "target": body.target, "epochs": body.epochs},
                             request.state.user["username"])
        return {"ok": True, "id": cid, "message": "Đã gửi lệnh — máy AI nhận ở nhịp liên lạc tiếp theo (≤ 30 giây)."}

    # ================================================================== địa điểm
    def _place_for(request: Request, pid: int) -> dict | None:
        p = db.place(pid)
        u = request.state.user
        return p if p and (u["role"] == "admin" or p["user_id"] == u["id"]) else None

    @app.get("/api/places")
    def places_list(request: Request):
        u = request.state.user
        return JSONResponse({"items": places_view(None if u["role"] == "admin" else u["id"])}, headers=NO_STORE)

    @app.post("/api/places")
    def place_create(body: PlaceIn, request: Request):
        u = request.state.user
        owner = body.user_id if (u["role"] == "admin" and body.user_id) else u["id"]
        if u["role"] != "admin" and len(db.places(owner)) >= 10:
            return _err(409, "Mỗi tài khoản tối đa 10 địa điểm")
        try:
            return {"ok": True, "id": db.create_place(owner, body.name)}
        except KeyError:
            return _err(404, "Không có tài khoản này")
        except ValueError as e:
            return _err(400, str(e))

    @app.post("/api/places/assign")
    def place_assign(body: AssignIn, request: Request):
        """Chuyển camera sang địa điểm khác. Người dùng: chỉ camera của mình, sang địa điểm của mình."""
        u = request.state.user
        if _place_for(request, body.place_id) is None:
            return _err(404, "Không có địa điểm này")
        k = (body.site_id, body.edge_id, body.cam_id)
        if u["role"] != "admin" and k not in user_scope(u["id"]):
            return _err(404, "Không có camera này")
        if k not in place_of_cams():
            return _err(404, "Không có camera này")
        db.assign_cam(*k, body.place_id)
        return {"ok": True}

    @app.post("/api/places/{pid}")
    def place_rename(pid: int, body: PlaceUpd, request: Request):
        if _place_for(request, pid) is None:
            return _err(404, "Không có địa điểm này")
        try:
            db.rename_place(pid, body.name)
        except ValueError as e:
            return _err(400, str(e))
        return {"ok": True}

    @app.get("/api/ppe")
    def ppe_catalog():
        """Danh mục đồ bảo hộ + mẫu + tình trạng model trên máy chủ AI (đã cài chưa, nhận ra món nào)."""
        e = ai_edge()
        st = ((((e or {}).get("info") or {}).get("ai") or {}).get("behavior") or {}).get("ppe") if e else None
        return JSONResponse({"items": {k: {"name": v[0], "short": v[1]} for k, v in PPE_ITEMS.items()},
                             "presets": PPE_PRESETS, "model": st or {"ready": False, "hint": "Máy chủ AI chưa chạy hoặc chưa cập nhật bản mới."}},
                            headers=NO_STORE)

    @app.post("/api/places/{pid}/ppe")
    async def place_set_ppe(pid: int, request: Request):
        if _place_for(request, pid) is None:
            return _err(404, "Không có địa điểm này")
        try:
            body = json.loads(await request.body() or b"{}")
        except ValueError:
            return _err(400, "JSON không hợp lệ")
        rules = clean_ppe(body.get("preset") if isinstance(body, dict) and body.get("preset") else body)
        db.set_place_ppe(pid, rules)
        return {"ok": True, "ppe": rules}

    @app.post("/api/places/{pid}/delete")
    def place_delete(pid: int, request: Request):
        if _place_for(request, pid) is None:
            return _err(404, "Không có địa điểm này")
        try:
            db.delete_place(pid)
        except ValueError as e:
            return _err(400, str(e))
        return {"ok": True}

    # ================================================================== mạch cảm biến
    def _dev_for(request: Request, did: int) -> dict | None:
        d = db.device(did)
        u = request.state.user
        return d if d and (u["role"] == "admin" or d["user_id"] == u["id"]) else None

    def _dev_view(d: dict, places: dict, users: dict, now: float) -> dict:
        p = places.get(d["place_id"], {})
        return {"id": d["id"], "name": d["name"], "kind": d["kind"], "kind_label": DEVICE_KINDS.get(d["kind"], d["kind"]),
                "place_id": d["place_id"], "place_name": p.get("name", "?"), "user_id": d["user_id"],
                "owner": users.get(d["user_id"], "?"), "online": bool(d["last_seen"] and now - d["last_seen"] <= DEVICE_ONLINE_S),
                "last_seen": d["last_seen"], "data": d["last_data"], "desired": d["desired"], "via": d["via"],
                "created": d["created"],
                # chân thật sẽ dùng (kênh chưa chọn chân -> chân mặc định, giống code tải về)
                "channels": with_default_pins(d.get("channels") or [], "usb" if d["kind"] in ("usb", "other") else d["kind"])}

    @app.get("/api/devices")
    def devices_list(request: Request):
        u = request.state.user
        now = time.time()
        places = {p["id"]: p for p in db.places()}
        users = {x["id"]: (x["name"] or x["username"]) for x in db.users()}
        items = [_dev_view(d, places, users, now) for d in db.devices(None if u["role"] == "admin" else u["id"])]
        mqtt = None
        if u["role"] == "admin":  # cảm biến MQTT của phòng máy chủ (ESP32 nối với ARES điều khiển đèn/điều hoà)
            e = ai_edge()
            s = (((e or {}).get("info") or {}).get("integrations") or {}).get("sensors") if e else None
            if s:
                mqtt = {"data": {k: v for k, v in {"temp": s.get("temp_in"), "hum": s.get("humidity"),
                                                    "lux": s.get("lux_window")}.items()
                                 if v is not None}, "name": e["name"]}
        return JSONResponse({"items": items, "kinds": DEVICE_KINDS, "mqtt": mqtt,
                             "places": db.places(None if u["role"] == "admin" else u["id"])}, headers=NO_STORE)

    @app.post("/api/devices")
    def device_create(body: DeviceIn, request: Request):
        u = request.state.user
        pid = body.place_id or db.default_place(u["id"])
        p = _place_for(request, pid)
        if p is None:
            return _err(404, "Không có địa điểm này")
        if u["role"] != "admin" and len(db.devices(u["id"])) >= 20:
            return _err(409, "Mỗi tài khoản tối đa 20 mạch")
        d = db.add_device(p["user_id"], pid, body.name, body.kind, clean_channels(body.channels))
        return {"ok": True, "id": d["id"]}

    @app.post("/api/devices/{did}")
    def device_update(did: int, body: DeviceUpd, request: Request):
        d = _dev_for(request, did)
        if d is None:
            return _err(404, "Không có mạch này")
        kw = {}
        if body.name:
            kw["name"] = body.name.strip()[:60]
        if body.place_id:
            p = _place_for(request, body.place_id)
            if p is None:
                return _err(404, "Không có địa điểm này")
            kw.update(place_id=p["id"], user_id=p["user_id"])
        if body.channels is not None:
            kw["channels"] = clean_channels(body.channels)
        if kw:
            db.set_device(did, **kw)
        return {"ok": True, "channels": kw.get("channels", d["channels"])}

    @app.post("/api/devices/{did}/delete")
    def device_delete(did: int, request: Request):
        if _dev_for(request, did) is None:
            return _err(404, "Không có mạch này")
        db.delete_device(did)
        return {"ok": True}

    @app.post("/api/devices/{did}/cmd")
    async def device_cmd(did: int, request: Request):
        """Bật/tắt relay, còi, chỉnh độ sáng LED — mạch Wi-Fi nhận ở lần gửi số đo tiếp theo, mạch USB nhận ngay qua trình duyệt."""
        d = _dev_for(request, did)
        if d is None:
            return _err(404, "Không có mạch này")
        try:
            body = json.loads(await request.body() or b"{}")
        except ValueError:
            return _err(400, "JSON không hợp lệ")
        des = dict(d["desired"])
        for k, v in (body.items() if isinstance(body, dict) else []):
            if k in CONTROL_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
                lo, hi = CONTROL_KEYS[k]
                des[k] = int(max(lo, min(hi, v)))
        db.set_device(did, desired=des)
        return {"ok": True, "desired": des}

    @app.post("/api/devices/{did}/data")
    async def device_data_usb(did: int, request: Request):
        """Số đo từ mạch cắm USB: trình duyệt (Web Serial) đọc cổng USB rồi gửi lên đây bằng phiên đăng nhập."""
        d = _dev_for(request, did)
        if d is None:
            return _err(404, "Không có mạch này")
        try:
            body = json.loads(await request.body() or b"{}")
        except ValueError:
            return _err(400, "JSON không hợp lệ")
        return _ingest(d, body, "usb", request)

    @app.get("/api/devices/{did}/cmd")
    def device_cmd_get(did: int, request: Request):
        """Lệnh mong muốn hiện tại — dashboard đang nối mạch USB hỏi mỗi giây để đèn phản hồi nhanh."""
        d = _dev_for(request, did)
        return JSONResponse({"desired": d["desired"]}, headers=NO_STORE) if d else _err(404, "Không có mạch này")

    # ------------------------------------------------------------------ vùng & đèn của camera thật
    def _zone_cams(request: Request) -> list[dict]:
        sc = scope_of(request)
        now = time.time()
        pmap = place_of_cams()
        pinfo = {p["id"]: p for p in db.places()}
        unames = {u["id"]: (u["name"] or u["username"]) for u in db.users()}
        cnames = {(c["site_id"], c["edge_id"], c["cam_id"]): c["name"] for c in db.client_cams()}
        hw = {(c["site_id"], c["edge_id"], c["cam_id"]): c for c in db.cameras()}
        lay = zones.layouts
        out = []
        for k in set(pmap) | set(hw):
            if sc is not None and k not in sc:
                continue
            c, pl = hw.get(k, {}), pinfo.get(pmap.get(k), {})
            lv = live.get(k)
            L = lay.get(k) or {}
            v = zones.view(k)
            out.append({"site_id": k[0], "edge_id": k[1], "cam_id": k[2], "name": cnames.get(k) or c.get("name") or k[2],
                        "place_id": pl.get("id"), "place_name": pl.get("name", "Chưa gán"), "owner": unames.get(pl.get("user_id"), ""),
                        "owner_id": pl.get("user_id"), "online": bool(c.get("online")) and now - (c.get("last_seen") or 0) <= cfg.offline_after_s,
                        "n_people": c.get("n_people", 0), "live": bool(lv and now - lv[1] <= 10),
                        "n_zones": len(L.get("zones", [])), "n_lights": len(L.get("lights", [])),
                        "lights_on": sum(1 for x in (v.get("lights") or {}).values() if x["on"])})
        out.sort(key=lambda x: (x["owner"], x["place_name"], x["name"]))
        return out

    @app.get("/api/zones")
    def zones_list(request: Request):
        return JSONResponse({"cams": _zone_cams(request)}, headers=NO_STORE)

    def _zone_key(request: Request, site_id: int, edge_id: str, cam_id: str):
        k = (int(site_id), edge_id, cam_id)
        if not any((c["site_id"], c["edge_id"], c["cam_id"]) == k for c in _zone_cams(request)):
            return None
        return k

    @app.get("/api/zones/{site_id}/{edge_id}/{cam_id}")
    def zones_get(site_id: int, edge_id: str, cam_id: str, request: Request):
        k = _zone_key(request, site_id, edge_id, cam_id)
        if k is None:
            return _err(404, "Không có camera này")
        L = {kk: v for kk, v in (zones.layouts.get(k) or {}).items() if not kk.startswith("_")}
        return JSONResponse({"layout": L or None, "state": zones.view(k)}, headers=NO_STORE)

    @app.post("/api/zones/{site_id}/{edge_id}/{cam_id}")
    async def zones_set(site_id: int, edge_id: str, cam_id: str, request: Request):
        from ares_server.layout import clean_layout
        k = _zone_key(request, site_id, edge_id, cam_id)
        if k is None:
            return _err(404, "Không có camera này")
        try:
            raw = (await request.body()).decode("utf-8", "replace")
            if len(raw) > 30000:
                raise ValueError("bố trí quá lớn")
            L = clean_layout(json.loads(raw or "{}"))
        except ValueError as e:
            return _err(400, f"Bố trí vùng/đèn không hợp lệ: {e}")
        for li in L["lights"]:
            if li["device_id"] and _dev_for(request, li["device_id"]) is None:
                return _err(400, f"{li['name']}: mạch #{li['device_id']} không phải của bạn")
        db.set_cam_layout(*k, L, request.state.user["username"])
        zones.reload()
        return {"ok": True, "layout": L, "state": zones.view(k)}

    @app.get("/api/zones-state")
    def zones_state(request: Request):
        """Trạng thái trực tiếp (số người từng vùng, đèn bật/tắt + lý do) của mọi camera có bố trí mà người này xem được."""
        sc = scope_of(request)
        out = {f"{k[0]}/{k[1]}/{k[2]}": zones.view(k) for k in list(zones.layouts) if sc is None or k in sc}
        return JSONResponse({"cams": out, "t": time.time()}, headers=NO_STORE)

    @app.get("/api/devices/{did}/series")
    def device_series(did: int, request: Request, hours: float = 24):
        if _dev_for(request, did) is None:
            return _err(404, "Không có mạch này")
        return JSONResponse({"items": db.device_series(did, time.time() - max(0.5, min(hours, 24 * 7)) * 3600)},
                            headers=NO_STORE)

    @app.get("/api/devices/{did}/firmware.ino")
    def device_firmware(did: int, request: Request, server: str = ""):
        """Code nạp cho mạch, đã ghi sẵn khoá thiết bị + địa chỉ máy chủ. Wi-Fi do trình duyệt điền (không gửi lên máy chủ)."""
        d = _dev_for(request, did)
        if d is None:
            return _err(404, "Không có mạch này")
        fw = Path(__file__).parent / "firmware"
        tpl = (fw / ("ares_usb.ino" if d["kind"] == "usb" else "ares_wifi.ino")).read_text(encoding="utf-8")
        srv = server.strip().rstrip("/") or cfg.public_url.rstrip("/") or _base_url(request).rstrip("/")
        if not re.match(r"^https?://[A-Za-z0-9.\-:\[\]]+(/[A-Za-z0-9._~/\-]*)?$", srv):
            return _err(400, "Địa chỉ máy chủ không hợp lệ")
        name = re.sub(r"[^\w .\-]", "", d["name"])[:60]
        sens = (fw / "common_sensors.inc").read_text(encoding="utf-8").strip()
        # bật sẵn đúng cảm biến mạch đang có (theo số đo mạch đã gửi) — mặc định tắt hết để nạp được ngay, không đòi thư viện
        keys = set((d.get("last_data") or d.get("data") or {}).keys())
        on = {"DHT": bool(keys & {"temp", "hum"}), "LDR": bool(keys & {"lux", "raw", "lux_window", "light"}),
              "PIR": "motion" in keys, "GAS": "gas" in keys, "SOUND": "sound" in keys, "SR04": "dist" in keys}
        for k, v in on.items():
            sens = re.sub(rf"(#define USE_{k}\s+)[01]", lambda m, v=v: m.group(1) + ("1" if v else "0"), sens, count=1)
        lz = (d.get("last_data") or {}).get("lux_zones")
        n_ldr = max([len(lz) if isinstance(lz, list) else 0] + [int(k[5:]) for k in keys if re.fullmatch(r"light[1-6]", k)] + [1])
        if n_ldr > 1:
            sens = re.sub(r"(#define USE_LDR\s+)[01]", r"\g<1>1", sens, count=1)
        code = (tpl.replace("{{LDR_COUNT}}", str(min(6, n_ldr))).replace("{{SENSORS}}", sens)
                .replace("{{CHANNELS}}", channels_code(d.get("channels") or [], "usb" if d["kind"] in ("usb", "other") else d["kind"]))
                .replace("{{SERVER}}", srv).replace("{{TOKEN}}", d["token"]).replace("{{NAME}}", name))
        fn = ("ares_usb" if d["kind"] == "usb" else "ares_wifi") + f"_{d['id']}.ino"
        return Response(code, media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{fn}"', "Cache-Control": "no-store"})

    # ================================================================== thử video (tải video lên -> AI máy chủ xử lý)
    def _own_job(request: Request, jid: int) -> dict | None:
        j = db.video_job(jid)
        u = request.state.user
        return j if j and (j["user_id"] == u["id"] or u["role"] == "admin") else None

    def _parse_opts(raw: str) -> dict | None:
        from ares_server.layout import clean_layout
        if not raw:
            return clean_layout({})
        if len(raw) > 20000:
            raise ValueError("Bố trí vùng/đèn quá lớn")
        return clean_layout(json.loads(raw))

    @app.post("/api/videos")
    async def video_upload(request: Request, name: str = "", ppe: str = "", kind: str = "behavior", opts: str = ""):
        u = request.state.user
        if kind not in ("behavior", "people"):
            return _err(400, "Loại thử video không hợp lệ")
        try:
            layout = _parse_opts(opts) if kind == "people" else None
        except ValueError as e:
            return _err(400, f"Bố trí vùng/đèn không hợp lệ: {e}")
        name = (name or "video.mp4").replace("\\", "/").split("/")[-1].strip()[:120] or "video.mp4"
        ext = Path(name).suffix.lower()
        if ext not in VIDEO_EXT:
            return _err(415, "Chỉ nhận video: " + ", ".join(sorted(VIDEO_EXT)))
        db.fail_stale_uploads(3600)  # tải lên quá 1 giờ chưa xong = đã đứt
        waiting = [j for j in db.video_jobs(u["id"]) if j["status"] in ("uploading", "queued", "running")]
        if len(waiting) >= 3:
            return _err(409, "Đang có 3 video chờ xử lý — đợi xong rồi tải tiếp")
        size = int(request.headers.get("content-length") or 0)
        if size > MAX_VIDEO_BYTES:
            return _err(413, "Video tối đa 500 MB — cắt ngắn bớt rồi tải lại")
        if ppe and ppe not in PPE_PRESETS:
            return _err(400, "Mẫu đồ bảo hộ không hợp lệ")
        # 'uploading': máy AI chỉ được nhận job khi file đã về ĐỦ. (Trước đây job vào hàng chờ ngay lúc bắt đầu tải, tải qua
        # Internet mất cả phút nên máy AI nhận file dở -> đọc được 0 khung mà vẫn báo "Xong".)
        jid = db.add_video_job(u["id"], name, ext, size, clean_ppe(ppe) if ppe and kind == "behavior" else None,
                               status="uploading", kind=kind, opts=layout)
        p = _job_dir(jid) / f"source{ext}"
        n = 0
        try:
            with open(p, "wb") as f:
                async for chunk in request.stream():
                    n += len(chunk)
                    if n > MAX_VIDEO_BYTES:
                        raise ValueError("Video tối đa 500 MB")
                    f.write(chunk)
            if n < 1024:
                raise ValueError("File rỗng hoặc quá nhỏ")
            if size and n != size:
                raise ValueError(f"mới nhận {n}/{size} byte — mạng đứt giữa chừng, tải lại")
        except Exception as e:  # noqa: BLE001 — tải dở/quá lớn/đứt mạng: xoá luôn job
            import shutil
            db.delete_video_job(jid)
            shutil.rmtree(_job_dir(jid), ignore_errors=True)
            return _err(400, f"Tải video lỗi: {e}")
        db.update_video_job(jid, status="queued", size=n,
                            message="Đang chờ máy AI" if ai_edge() else "Đang chờ — máy chủ AI chưa chạy")
        return {"ok": True, "id": jid}

    @app.post("/api/videos/{jid}/retry")
    def video_retry(jid: int, request: Request):
        """Phân tích lại video đã tải lên (không phải tải lại): dùng khi lần trước lỗi / ra 0 khung / vừa cập nhật AI."""
        j = _own_job(request, jid)
        if j is None:
            return _err(404, "Không có video này")
        if j["status"] not in ("done", "error"):
            return _err(409, "Video này đang chờ hoặc đang phân tích")
        src = _job_dir(jid) / f"source{j['ext']}"
        if not src.exists() or src.stat().st_size < 1024:
            return _err(410, "Máy chủ không còn file gốc — tải lại video")
        for fn in ("frames.zip", "result.mp4", "tracks.json"):
            (_job_dir(jid) / fn).unlink(missing_ok=True)
        db.update_video_job(jid, status="queued", progress=0.0, result="", n_frames=0, has_mp4=0, size=src.stat().st_size,
                            started=None, finished=None,
                            message="Đang chờ máy AI (phân tích lại)" if ai_edge() else "Đang chờ — máy chủ AI chưa chạy")
        return {"ok": True}

    @app.post("/api/videos/{jid}/opts")
    async def video_opts(jid: int, request: Request):
        """Đổi bố trí vùng/đèn của video phần Người & môi trường (rồi bấm Phân tích lại — không phải tải lại video)."""
        j = _own_job(request, jid)
        if j is None:
            return _err(404, "Không có video này")
        if j["kind"] != "people":
            return _err(400, "Chỉ video thử phần Người & môi trường mới có vùng/đèn")
        if j["status"] in ("queued", "running", "uploading"):
            return _err(409, "Video đang chờ hoặc đang phân tích — đợi xong rồi sửa")
        try:
            layout = _parse_opts((await request.body()).decode("utf-8", "replace"))
        except ValueError as e:
            return _err(400, f"Bố trí vùng/đèn không hợp lệ: {e}")
        db.update_video_job(jid, opts=json.dumps(layout, ensure_ascii=False))
        return {"ok": True, "opts": layout}

    @app.get("/api/videos/{jid}/thumb.jpg")
    def video_thumb(jid: int, request: Request):
        """Khung hình gốc đầu video (để vẽ lại vùng). Không có OpenCV trên server thì dùng khung kết quả đầu tiên."""
        j = _own_job(request, jid)
        if j is None:
            return _err(404, "Không có video này")
        d = _job_dir(jid)
        th = d / "thumb.jpg"
        if not th.exists():
            src = d / f"source{j['ext']}"
            try:
                import cv2
                cap = cv2.VideoCapture(str(src))
                cap.set(cv2.CAP_PROP_POS_MSEC, 500)
                ok, img = cap.read()
                cap.release()
                if ok:
                    if img.shape[1] > 960:
                        img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
                    cv2.imwrite(str(th), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            except Exception:  # noqa: BLE001
                pass
        if th.exists():
            return FileResponse(th, media_type="image/jpeg")
        return video_frame(jid, 1, request)

    @app.get("/api/videos")
    def video_list(request: Request, kind: str = ""):
        u = request.state.user
        items = db.video_jobs(None if u["role"] == "admin" else u["id"], kind=kind or None)
        names = {x["id"]: (x["name"] or x["username"]) for x in db.users()}
        q = [j["id"] for j in sorted(db.video_jobs(None, 500), key=lambda j: j["id"]) if j["status"] == "queued"]
        for j in items:
            j["owner"] = names.get(j["user_id"], "?")
            j["queue_pos"] = q.index(j["id"]) + 1 if j["id"] in q else 0
        return JSONResponse({"items": items, "ai": ai_status()}, headers=NO_STORE)

    @app.get("/api/videos/{jid}")
    def video_get(jid: int, request: Request):
        j = _own_job(request, jid)
        return JSONResponse(j, headers=NO_STORE) if j else _err(404, "Không có video này")

    @app.get("/api/videos/{jid}/frames/{n}.jpg")
    def video_frame(jid: int, n: int, request: Request):
        j = _own_job(request, jid)
        p = _job_dir(jid) / "frames.zip" if j else None
        if p is None or not p.exists():
            return _err(404, "Chưa có kết quả")
        try:
            with zipfile.ZipFile(p) as zf:
                data = zf.read(f"{max(1, min(n, 99999)):04d}.jpg")
        except KeyError:
            return _err(404, "Không có khung này")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/videos/{jid}/source")
    def video_source(jid: int, request: Request):
        """Video gốc đã tải lên — web phát mượt bằng thẻ <video> (hỗ trợ tua: Range) và tự vẽ vùng/người/đèn đè lên."""
        j = _own_job(request, jid)
        p = _job_dir(jid) / f"source{j['ext']}" if j else None
        if p is None or not p.exists():
            return _err(404, "Không còn video gốc")
        mt = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
              ".mkv": "video/x-matroska", ".avi": "video/x-msvideo"}.get(j["ext"], "application/octet-stream")
        return FileResponse(p, media_type=mt, headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/videos/{jid}/tracks.json")
    def video_tracks(jid: int, request: Request):
        j = _own_job(request, jid)
        p = _job_dir(jid) / "tracks.json" if j else None
        if p is None or not p.exists():
            return _err(404, "Kết quả này chưa có vị trí người từng khung — bấm Phân tích lại")
        return FileResponse(p, media_type="application/json", headers={"Cache-Control": "private, max-age=600"})

    @app.get("/api/videos/{jid}/result.mp4")
    def video_mp4(jid: int, request: Request):
        j = _own_job(request, jid)
        p = _job_dir(jid) / "result.mp4" if j else None
        if p is None or not p.exists():
            return _err(404, "Không có video kết quả")
        return FileResponse(p, media_type="video/mp4", filename=f"{Path(j['name']).stem}_ket_qua_AI.mp4")

    @app.post("/api/videos/{jid}/delete")
    def video_delete(jid: int, request: Request):
        import shutil
        if _own_job(request, jid) is None:
            return _err(404, "Không có video này")
        db.delete_video_job(jid)
        shutil.rmtree(vdir / str(int(jid)), ignore_errors=True)
        return {"ok": True}

    @app.post("/api/me")
    def me_update(body: MeIn, request: Request):
        db.set_user_name(request.state.user["id"], body.name)
        return {"ok": True}

    @app.get("/api/sites")
    def sites():
        return {"items": _site_cards()}

    @app.post("/api/sites")
    def create_site(body: SiteIn):
        sid, key = db.create_site(body.name, body.address, body.note)
        return {"id": sid, "key": key, "note": "Khoá chỉ hiện một lần — dán vào ARES_UPLINK_KEY trên máy Edge."}

    @app.get("/api/sites/{site_id}")
    def site_detail(site_id: int):
        try:
            s = db.site(site_id)
        except KeyError:
            return _err(404, "Không có địa điểm này")
        now = time.time()
        edges = db.edges(site_id)
        for e in edges:
            e["online"] = now - (e["last_seen"] or 0) <= cfg.offline_after_s
        on = {e["edge_id"] for e in edges if e["online"]}
        cams = db.cameras(site_id)
        for c in cams:
            c["online"] = bool(c["online"]) and c["edge_id"] in on
        return JSONResponse({"site": s, "edges": edges, "cameras": cams, "series": db.series(site_id, now - 24 * 3600, 300),
                             "events": db.events(limit=30, site_id=site_id), "commands": db.commands(site_id, 10),
                             "live": sorted(c for (sid, e, c) in live if sid == site_id)}, headers=NO_STORE)

    @app.post("/api/sites/{site_id}")
    def update_site(site_id: int, body: SiteUpd):
        db.update_site(site_id, **{k: (int(v) if isinstance(v, bool) else v) for k, v in body.model_dump().items() if v is not None})
        return {"ok": True}

    @app.post("/api/sites/{site_id}/rotate-key")
    def rotate(site_id: int):
        try:
            return {"key": db.rotate_key(site_id), "note": "Khoá cũ hết hiệu lực ngay. Cập nhật ARES_UPLINK_KEY trên máy Edge."}
        except KeyError:
            return _err(404, "Không có địa điểm này")

    # ================================================================== sự kiện
    @app.get("/api/events")
    def events(request: Request, status: str = "", site: int = 0, kind: str = "", limit: int = 100, place: int = 0):
        if status not in ("", "new", "ack") or not KIND_RE.match(kind):
            return _err(400, "Tham số không hợp lệ")
        sc = scope_of(request)
        if place:
            pm = place_of_cams()
            keys = [k for k, pid in pm.items() if pid == place]
            sc = keys if sc is None else [k for k in sc if k in keys]
        return JSONResponse({"items": _with_place(db.events(max(1, min(limit, 500)), status, site or None, kind, scope=sc))},
                            headers=NO_STORE)

    def _with_place(items: list[dict]) -> list[dict]:
        """Gắn tên địa điểm (theo người dùng) cho sự kiện — thay cho tên 'địa điểm kỹ thuật' của máy Edge."""
        pm, pinfo = place_of_cams(), {p["id"]: p for p in db.places()}
        cn = {(c["site_id"], c["edge_id"], c["cam_id"]): c["name"] for c in db.client_cams()}
        for e in items:
            k = (e["site_id"], e["edge_id"], e["cam_id"])
            pl = pinfo.get(pm.get(k), {})
            e["place_id"], e["place_name"] = pl.get("id"), pl.get("name", "")
            if k in cn:
                e["cam_name"] = cn[k]
        return items

    def _event_for(request: Request, eid: int):
        try:
            e = db.event(eid)
        except KeyError:
            return None
        return e if in_scope(request, e["site_id"], e["edge_id"], e["cam_id"]) else None

    @app.get("/api/events/{eid}")
    def ev_get(eid: int, request: Request):
        e = _event_for(request, eid)
        return JSONResponse(_with_place([e])[0], headers=NO_STORE) if e else _err(404, "Không có sự kiện này")

    def _media(request: Request, eid: int, col: str, mime: str):
        e = _event_for(request, eid)
        if e is None:
            return _err(404, "Không có sự kiện này")
        if not e[col]:
            return _err(404, "Sự kiện không có file này (Edge không gửi hoặc đã quá hạn lưu)")
        p = (media / e[col]).resolve()
        if media.resolve() not in p.parents or not p.exists():
            return _err(404, "File không còn")
        return FileResponse(p, media_type=mime)

    @app.get("/api/events/{eid}/snapshot.jpg")
    def ev_snapshot(eid: int, request: Request):
        return _media(request, eid, "snapshot", "image/jpeg")

    @app.get("/api/events/{eid}/frames/{n}.jpg")
    def ev_frame(eid: int, n: int, request: Request):
        e = _event_for(request, eid)
        if e is None:
            return _err(404, "Không có sự kiện này")
        p = (media / e["frames"]).resolve() if e["frames"] else None
        if p is None or media.resolve() not in p.parents or not p.exists():
            return _err(404, "Sự kiện không có clip")
        try:
            with zipfile.ZipFile(p) as zf:
                data = zf.read(f"{max(1, min(n, 9999)):04d}.jpg")
        except KeyError:
            return _err(404, "Không có khung này")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/events/{eid}/clip.mp4")
    def ev_clip(eid: int, request: Request):
        return _media(request, eid, "clip", "video/mp4")

    @app.post("/api/events/{eid}/ack")
    def ev_ack(eid: int, request: Request):
        if _event_for(request, eid) is None:
            return _err(404, "Không có sự kiện này")
        try:
            db.handle_event(eid, session_user(request) or "?")
        except KeyError:
            return _err(404, "Không có sự kiện này")
        return {"ok": True}

    @app.post("/api/events/{eid}/verdict")
    def ev_verdict(eid: int, body: VerdictIn, request: Request):
        if _event_for(request, eid) is None:
            return _err(404, "Không có sự kiện này")
        try:
            db.handle_event(eid, session_user(request) or "?", body.verdict)
        except KeyError:
            return _err(404, "Không có sự kiện này")
        return {"ok": True}

    @app.post("/api/events/ack-all")
    def ev_ack_all(request: Request):
        return {"acked": db.ack_all(session_user(request) or "?", scope_of(request))}

    # ================================================================== hình trực tiếp
    @app.get("/api/live")
    def live_list(request: Request):
        now = time.time()
        sc = scope_of(request)
        pmap = place_of_cams()
        pinfo = {p["id"]: p for p in db.places()}
        unames = {u["id"]: (u["name"] or u["username"]) for u in db.users()}
        cnames = {(c["site_id"], c["edge_id"], c["cam_id"]): c["name"] for c in db.client_cams()}
        cams = {(c["site_id"], c["edge_id"], c["cam_id"]): c for c in db.cameras()}
        items = []
        for (sid, eid, cam), (_, t) in list(live.items()):
            if now - t > 300:
                live.pop((sid, eid, cam), None)
                continue
            if sc is not None and (sid, eid, cam) not in sc:
                continue
            c = cams.get((sid, eid, cam), {})
            pl = pinfo.get(pmap.get((sid, eid, cam)), {})
            items.append({"site_id": sid, "edge_id": eid, "cam_id": cam,
                          "place_id": pl.get("id"), "place_name": pl.get("name", "Chưa gán"),
                          "owner": unames.get(pl.get("user_id"), ""),
                          "name": cnames.get((sid, eid, cam)) or c.get("name") or cam,
                          "n_people": c.get("n_people", 0), "age_s": round(now - t, 1)})
        items.sort(key=lambda x: (x["owner"], x["place_name"], x["name"]))
        return JSONResponse({"items": items}, headers=NO_STORE)

    @app.get("/api/live/{site_id}/{edge_id}/{cam_id}.jpg")
    def live_frame(site_id: int, edge_id: str, cam_id: str, request: Request):
        if not in_scope(request, site_id, edge_id, cam_id):
            return _err(404, "Không có camera này")
        v = live.get((site_id, edge_id, cam_id))
        if v is None:
            return _err(404, "Camera này chưa gửi hình trực tiếp (máy Edge cần ARES_UPLINK_LIVE_S > 0)")
        return Response(v[0], media_type="image/jpeg", headers=NO_STORE)

    # ================================================================== camera của máy khác (người dùng app)
    @app.get("/api/my/cameras")
    def my_cams(request: Request):
        return JSONResponse({"items": _my_cams(request.state.user), "server_ai": ai_edge() is not None,
                             "places": db.places(request.state.user["id"])},
                            headers=NO_STORE)

    @app.post("/api/my/cameras")
    def my_cam_add(body: MyCamIn, request: Request):
        u = request.state.user
        if len(db.client_cams(u["id"])) >= 4:
            return _err(409, "Mỗi tài khoản tối đa 4 camera")
        e = ai_edge()
        if e is None:
            return _err(409, "Máy chủ AI chưa chạy (chưa có máy Edge nào trực tuyến). Báo quản trị viên mở ARES_MAY_CHU.bat.")
        if body.place_id and (_place_for(request, body.place_id) or {}).get("user_id") != u["id"]:
            return _err(404, "Không có địa điểm này")
        same = [c for c in db.client_cams(u["id"]) if c["name"].strip().lower() == body.name.strip().lower()
                and c["site_id"] == e["site_id"] and c["edge_id"] == e["edge_id"]]
        if same:  # mở lại app trên cùng máy: dùng lại camera cũ, không tạo bản trùng
            return {"ok": True, "id": same[0]["id"], "cam_id": same[0]["cam_id"], "reused": True}
        c = db.add_client_cam(u["id"], e["site_id"], e["edge_id"], body.name, body.place_id)
        label = f"{body.name.strip()} · {u['name'] or u['username']}"[:60]
        db.add_command(e["site_id"], e["edge_id"], {"type": "add_camera", "name": label, "source": f"ares://{c['token']}",
                                                    "cam_id": c["cam_id"]}, u["username"])
        return {"ok": True, "id": c["id"], "cam_id": c["cam_id"]}

    def _own_cam(request: Request, cid: int) -> dict | None:
        c = db.client_cam(cid)
        u = request.state.user
        return c if c and (c["user_id"] == u["id"] or u["role"] == "admin") else None

    @app.post("/api/my/cameras/{cid}/frame")
    async def my_cam_frame(cid: int, request: Request):
        c = _own_cam(request, cid)
        if c is None:
            return _err(404, "Không có camera này")
        data = await request.body()
        if len(data) > 800_000 or not data.startswith(b"\xff\xd8\xff"):
            return _err(415, "Ảnh phải là JPEG ≤ 800 KB")
        client_frames[c["token"]] = (data, time.time())
        return {"ok": True}

    @app.post("/api/my/cameras/{cid}/delete")
    def my_cam_delete(cid: int, request: Request):
        c = _own_cam(request, cid)
        if c is None:
            return _err(404, "Không có camera này")
        db.add_command(c["site_id"], c["edge_id"], {"type": "remove_camera", "cam_id": c["cam_id"]},
                       session_user(request) or "?")
        client_frames.pop(c["token"], None)
        live.pop((c["site_id"], c["edge_id"], c["cam_id"]), None)
        db.delete_client_cam(cid)
        return {"ok": True}

    @app.get("/api/v1/frame/{token}")
    def v1_frame(token: str, request: Request):
        a = edge_auth(request)
        if isinstance(a, JSONResponse):
            return a
        c = db.client_cam_by_token(token)
        if c is None or c["site_id"] != a[0]["id"] or c["edge_id"] != a[1]:
            return _err(404, "Không có camera này")
        fr = client_frames.get(token)
        if fr is None or time.time() - fr[1] > 5:
            return Response(status_code=204)
        return Response(fr[0], media_type="image/jpeg", headers={"X-Frame-T": f"{fr[1]:.3f}", **NO_STORE})

    # ================================================================== quản lý camera từ xa
    def _edge_of(site_id: int, edge_id: str):
        if not EDGE_RE.match(edge_id) or not any(e["edge_id"] == edge_id for e in db.edges(site_id)):
            return _err(404, "Không có máy Edge này ở địa điểm này")
        return None

    @app.post("/api/sites/{site_id}/edges/{edge_id}/cameras")
    def cam_add(site_id: int, edge_id: str, body: CameraIn, request: Request):
        if (e := _edge_of(site_id, edge_id)) is not None:
            return e
        src = body.source.strip()
        if not SOURCE_RE.match(src):
            return _err(400, "Địa chỉ camera không hợp lệ. Dùng rtsp://user:mk@IP:554/..., http(s)://... hoặc số camera USB (0, 1, 2).")
        cid = db.add_command(site_id, edge_id, {"type": "add_camera", "name": body.name.strip(), "source": src},
                             session_user(request) or "?")
        return {"ok": True, "command_id": cid, "note": "Lệnh sẽ được máy Edge nhận trong vòng 30 giây."}

    @app.post("/api/sites/{site_id}/edges/{edge_id}/cameras/{cam_id}/remove")
    def cam_remove(site_id: int, edge_id: str, cam_id: str, request: Request):
        if (e := _edge_of(site_id, edge_id)) is not None:
            return e
        if not re.fullmatch(r"[A-Za-z0-9]{1,16}", cam_id) or cam_id == "main":
            return _err(400, "Không gỡ được camera này")
        cid = db.add_command(site_id, edge_id, {"type": "remove_camera", "cam_id": cam_id}, session_user(request) or "?")
        return {"ok": True, "command_id": cid}

    # ================================================================== app xem cho máy khác (không cần đăng nhập để TẢI)
    def _base_url(request: Request) -> str:
        host = request.headers.get("host", f"127.0.0.1:{cfg.port}")
        if not re.fullmatch(r"[A-Za-z0-9.\-:\[\]]{1,100}", host):
            host = f"127.0.0.1:{cfg.port}"
        if host.split(":")[0] in ("127.0.0.1", "localhost", "[") and cfg.host not in LOCAL:
            ips = lan_ips()  # mở trang tải ngay trên máy chủ: ghi IP mạng nội bộ để máy khác dùng được
            if ips:
                host = f"{ips[0]}:{cfg.port}"
        return f"{request.url.scheme}://{host}"

    @app.get("/tai-app")
    def download_page():
        return FileResponse(STATIC / "tai-app.html", media_type="text/html; charset=utf-8", headers=NO_STORE)

    @app.post("/api/admin/restart")
    def admin_restart():
        """Mở lại server + máy AI bằng mã mới (admin). Chỉ chạy được khi máy chủ mở bằng ARES_MAY_CHU.bat bản có tự mở lại."""
        from ares_server import restart

        if not restart.supervised():
            return _err(409, "Máy chủ đang chạy bản cũ chưa tự mở lại được — cần mở lại ARES_MAY_CHU.bat trên máy chủ một lần")
        restart.request()
        return {"ok": True}

    @app.get("/api/app-info")
    def app_info(request: Request):
        files = {p: (VIEWER / f).exists() for p, f in (("windows", "ARES-Viewer.exe"), ("macos", "viewer-darwin-arm64"),
                                                      ("linux", "viewer-linux-amd64"))}
        lan = [f"http://{ip}:{cfg.port}" for ip in lan_ips()] if cfg.host not in LOCAL else []
        pub, pages = cfg.public_url.rstrip("/"), cfg.pages_url
        return {"server": _base_url(request), "available": files, "version": __version__, "lan_urls": lan,
                "lan_ips": lan_ips(), "exposed": cfg.host not in LOCAL, "public_url": pub, "pages_url": pages,
                "share_link": f"{pages}?server={pub}" if pub and pages else ""}

    @app.get("/tai-app/ARES-Viewer-{platform}.zip")
    def download_zip(platform: str, request: Request):
        base = _base_url(request)
        conf = json.dumps({"server": base, "created": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=2)
        readme = (f"ARES Viewer — app xem dashboard ARES\r\n\r\nMáy chủ: {base}\r\n"
                  "Mở ARES Viewer là thấy dashboard. AI và camera chạy trên máy chủ, máy này chỉ để xem.\r\n"
                  "Đổi máy chủ: sửa file ares_viewer.json (hoặc xoá đi, app sẽ tự dò trong mạng nội bộ).\r\n")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            if platform == "Windows":
                exe = VIEWER / "ARES-Viewer.exe"
                if not exe.exists():
                    return _err(404, "Máy chủ chưa có bản build app Windows (ares_server/viewer/ARES-Viewer.exe)")
                z.write(exe, "ARES Viewer/ARES Viewer.exe")
                z.writestr("ARES Viewer/ares_viewer.json", conf)
                z.writestr("ARES Viewer/HUONG_DAN.txt", readme)
            elif platform == "macOS":
                arm, x64 = VIEWER / "viewer-darwin-arm64", VIEWER / "viewer-darwin-amd64"
                if not arm.exists() or not x64.exists():
                    return _err(404, "Máy chủ chưa có bản build app macOS")
                root = "ARES Viewer.app/Contents/"
                z.writestr(root + "Info.plist", MAC_PLIST)
                for src, name in ((arm, "viewer-arm64"), (x64, "viewer-x86_64")):
                    zi = zipfile.ZipInfo(root + "MacOS/" + name)
                    zi.external_attr = 0o755 << 16
                    z.writestr(zi, src.read_bytes())
                zi = zipfile.ZipInfo(root + "MacOS/ARES Viewer")
                zi.external_attr = 0o755 << 16
                z.writestr(zi, '#!/bin/sh\nD="$(cd "$(dirname "$0")" && pwd)"\nexec "$D/viewer-$(uname -m)" "$@"\n')
                z.writestr(root + "Resources/ares_viewer.json", conf)
                z.writestr("HUONG_DAN.txt", readme + "macOS: lần đầu bấm chuột phải vào app > Open (app chưa ký số).\r\n")
            elif platform == "Linux":
                b = VIEWER / "viewer-linux-amd64"
                if not b.exists():
                    return _err(404, "Máy chủ chưa có bản build app Linux")
                zi = zipfile.ZipInfo("ARES Viewer/ares-viewer")
                zi.external_attr = 0o755 << 16
                z.writestr(zi, b.read_bytes())
                z.writestr("ARES Viewer/ares_viewer.json", conf)
                z.writestr("ARES Viewer/HUONG_DAN.txt", readme)
            else:
                return _err(404, "Chỉ có Windows, macOS, Linux")
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="ARES-Viewer-{platform}.zip"', **NO_STORE})

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    _mount_traffic(app, cfg)
    return app


TRAFFIC_PAGES = {"": "index.html", "studio": "studio.html", "vi-pham": "vi_pham.html"}
_TRAFFIC_URL = re.compile(r"""(["'`])/(api/|studio|vi-pham)""")


def traffic_html(name: str) -> str:
    """Trang của gói Giao thông, đổi đường dẫn tuyệt đối /api/... -> /traffic/api/... để chạy dưới /traffic."""
    from traffic_safety_ai.dashboard.app import STATIC as TS

    html = (TS / name).read_text(encoding="utf-8")
    html = _TRAFFIC_URL.sub(lambda m: f"{m.group(1)}/traffic/{m.group(2)}", html)
    html = html.replace('href="/"', 'href="/traffic/"')
    # đường dẫn báo cáo do API trả về (/reports/...) cũng phải nằm dưới /traffic
    return html.replace('href="${esc(j.report_url)}"', 'href="/traffic${esc(j.report_url)}"').replace(
        'href="${esc(r.url)}"', 'href="/traffic${esc(r.url)}"')


def _mount_traffic(app: FastAPI, cfg) -> None:
    """Gắn dashboard + Studio (thử video) + trang xử lý vi phạm của gói traffic_safety_ai vào /traffic.

    Bảo vệ bằng phiên đăng nhập ARES (chỉ quản trị viên, xem guard). Gói chưa cài đủ thư viện -> /traffic báo rõ thiếu gì."""
    from fastapi.responses import HTMLResponse

    try:
        from traffic_safety_ai.dashboard.app import create_app as traffic_app
        sub = traffic_app(token=None)
        app.state.traffic_error = ""
    except Exception as e:  # noqa: BLE001 — thiếu yaml/scipy/... : server vẫn chạy, chỉ phần giao thông báo lỗi
        log.warning("Không gắn được phần Giao thông: %s", e)
        sub, app.state.traffic_error = None, f"{type(e).__name__}: {e}"

    def page(name: str):
        def h():
            if sub is None:
                return HTMLResponse("<meta charset='utf-8'><div style='font:15px system-ui;padding:20px'><b>Chưa chạy được phần "
                                    "Giao thông</b><p>" + str(app.state.traffic_error).replace("<", "&lt;") + "</p><p>Cài thư viện: "
                                    "<code>.venv\\Scripts\\python -m pip install -r traffic_safety_ai\\requirements.txt</code> "
                                    "rồi khởi động lại máy chủ.</p></div>", status_code=503)
            return HTMLResponse(traffic_html(name), headers={"Cache-Control": "no-store"})
        return h

    for path, name in TRAFFIC_PAGES.items():
        app.add_api_route(f"/traffic/{path}", page(name), methods=["GET"], include_in_schema=False)
    app.add_api_route("/traffic", lambda: Response(status_code=307, headers={"Location": "/traffic/"}), methods=["GET"],
                      include_in_schema=False)

    @app.get("/api/traffic/info")
    def traffic_info():
        return {"ok": sub is not None, "error": app.state.traffic_error}

    if sub is not None:
        app.mount("/traffic", sub)
