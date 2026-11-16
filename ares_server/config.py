"""Cấu hình server trung tâm ARES. Biến môi trường tiền tố ARES_SERVER_ hoặc file `server.env` / `ares_config/server.env`."""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARES_SERVER_",
        env_file=("server.env", str(REPO_ROOT / "ares_config" / "server.env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8200
    discovery_port: int = 8201  # UDP: app ARES Viewer gửi "ARES_DISCOVER" để tự tìm máy chủ trong mạng nội bộ (0 = tắt)
    data_dir: Path = REPO_ROOT / "var" / "server"
    timezone: str = "Asia/Ho_Chi_Minh"

    # đăng nhập dashboard (1 tài khoản quản trị cho bản MVP)
    admin_user: str = "admin"
    admin_password: str = ""  # BẮT BUỘC đặt khi mở ra mạng; để trống chỉ chạy được ở 127.0.0.1
    secret: str = ""  # khoá ký cookie phiên; trống = tự sinh và lưu vào data_dir/secret.key
    session_hours: float = Field(12.0, ge=0.5, le=24 * 30)
    # dashboard đặt ở trang khác (vd GitHub Pages) gọi về máy chủ: cho phép các trang này (cách nhau bằng dấu phẩy).
    # Trang *.github.io luôn được phép. Các trang này đăng nhập bằng token (không dùng cookie) nên không bị CSRF.
    cors_origins: str = ""
    public_url: str = ""  # địa chỉ https công khai của máy chủ (python -m ares_server share ghi vào) — hiện trong Thiết lập
    pages_url: str = ""  # địa chỉ dashboard trên GitHub Pages, vd https://ten.github.io/ares-dashboard/

    # nhận dữ liệu từ máy Edge
    max_upload_mb: float = Field(25.0, ge=1.0, le=500.0)
    offline_after_s: float = Field(120.0, ge=30.0, le=3600.0)  # quá lâu không nhận heartbeat => coi là mất kết nối
    media_keep_days: int = Field(30, ge=0)  # ảnh/clip sự kiện chưa đánh giá: tự xoá sau N ngày (0 = giữ mãi)
    metrics_keep_days: int = Field(90, ge=1)

    def ensure_dirs(self) -> None:
        (self.data_dir / "media").mkdir(parents=True, exist_ok=True)

    def session_secret(self) -> bytes:
        if self.secret:
            return self.secret.encode()
        self.ensure_dirs()
        p = self.data_dir / "secret.key"
        if not p.exists():
            p.write_text(secrets.token_hex(32), encoding="utf-8")
        return p.read_text(encoding="utf-8").strip().encode()


@lru_cache
def get_settings() -> ServerSettings:
    return ServerSettings()
