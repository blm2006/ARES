# Triển khai server trung tâm ARES

Server **không cần GPU**: nó chỉ nhận số liệu và sự kiện từ các máy Edge, lưu lại và hiển thị dashboard.
Chọn một trong ba cách dưới đây.

## Cách 1 — Local (Edge và server trên cùng một máy, không cần Internet)

```bat
ARES_TRON_BO.bat
```

Script tự tạo địa điểm, ghi `ARES_UPLINK_URL` và `ARES_UPLINK_KEY` vào `ares_config\ares.env`, mở server ở một cửa sổ riêng rồi chạy AI.

- Dashboard tổng hợp: http://127.0.0.1:8200
- Dashboard kỹ thuật của Edge: http://127.0.0.1:8100

## Cách 2 — Máy chủ trong mạng nội bộ (Windows, không dùng Docker)

```bat
.venv\Scripts\python -m pip install -r requirements-server.txt
copy ares_config\server.env.example ares_config\server.env
:: sửa server.env: ARES_SERVER_HOST=0.0.0.0, ARES_SERVER_ADMIN_PASSWORD=<mật khẩu dài>
ARES_SERVER.bat
```

Mở cổng 8200 trên tường lửa Windows.

Trên mỗi máy Edge, đặt `ARES_UPLINK_URL=http://<IP-máy-chủ>:8200` và dán khoá của địa điểm vào `ARES_UPLINK_KEY`.

## Cách 3 — VPS trên Internet (Ubuntu + Docker, có HTTPS tự động)

1. Trỏ tên miền về IP của VPS (bản ghi A).
2. Sao chép file cấu hình rồi sửa `ARES_DOMAIN` và `ARES_SERVER_ADMIN_PASSWORD`:
   ```bash
   cp deploy/server/.env.example deploy/server/.env
   ```
3. Khởi động:
   ```bash
   docker compose -f deploy/server/docker-compose.yml up -d --build
   ```
4. Tạo địa điểm và lấy khoá thiết bị:
   ```bash
   docker compose -f deploy/server/docker-compose.yml exec server python -m ares_server add-site "Trường A"
   ```
5. Trên máy Edge, đặt:
   ```
   ARES_UPLINK_URL=https://<tên-miền>
   ARES_UPLINK_KEY=ares_...
   ```

Caddy tự xin chứng chỉ Let's Encrypt. Dữ liệu nằm trong volume `ares_data` (SQLite, ảnh, clip), cần sao lưu volume này định kỳ.

## Bảo mật

- Tài khoản lưu trong DB server (mật khẩu băm PBKDF2). Tài khoản quản trị **đầu tiên** chỉ tạo được từ chính máy chủ (127.0.0.1); trên VPS/Docker dùng `ARES_SERVER_ADMIN_PASSWORD` để server tự tạo quản trị viên ở lần chạy đầu. Sau đó quản lý tài khoản ở Thiết lập → Tài khoản.
- Mỗi địa điểm có một khoá thiết bị riêng. Server chỉ lưu bản băm SHA-256 của khoá. Đổi khoá ở **Thiết lập → Đổi khoá**: khoá cũ hết hiệu lực ngay.
- Edge **không gửi luồng hình**. Mức dữ liệu gửi kèm sự kiện chỉnh bằng `ARES_UPLINK_MEDIA`:
  - `none`: chỉ thông tin
  - `snapshot`: thêm ảnh chụp
  - `clip`: thêm ảnh chụp và clip
- Ảnh/clip của sự kiện **chưa được đánh giá** sẽ tự xoá sau `ARES_SERVER_MEDIA_KEEP_DAYS` ngày (mặc định 30).
