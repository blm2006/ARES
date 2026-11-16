# ARES — AI camera chạy tại chỗ (YOLO)

Kho mã này gồm ba phần dùng chung một môi trường Python:

| Phần | Làm gì | Thư mục | Tài liệu chi tiết |
|---|---|---|---|
| **yolo-al** | Nhận diện + theo dõi người, vòng học chủ động (thu mẫu → duyệt → train → triển khai) | `realtime/` `collector/` `worker/` `review/` `dataset/` `training/` `deploy/` `common/` | [YOLO_AL.md](YOLO_AL.md), [SETUP.md](SETUP.md) |
| **ARES** | Đếm người, nhận diện hành vi nguy hiểm (té ngã, nằm bất động, ngồi gục, đánh nhau), cảnh báo kèm clip, dashboard; tuỳ chọn điều khiển đèn/điều hòa | `ares/` `ares_server/` `viewer/` `web/` `arduino/` | [SYSTEM.md](SYSTEM.md), [BEHAVIOR_AI.md](BEHAVIOR_AI.md), [ARES_UPGRADE.md](ARES_UPGRADE.md), `WORKFLOW.html` |
| **traffic_safety_ai** | Giám sát an toàn giao thông khu vực trường học: 16 tình huống, đọc biển số, phiếu ghi nhận, báo cáo | `traffic_safety_ai/` | [traffic_safety_ai/README.md](traffic_safety_ai/README.md) |

Toàn bộ giao diện và báo cáo bằng tiếng Việt. Các file `.bat` / `.ps1` viết không dấu để console Windows không lỗi font.

---

## 1. Yêu cầu

- Windows 10/11, Python **3.11 hoặc 3.12**.
- Nên có GPU NVIDIA. Không có GPU vẫn chạy được nhưng chậm (xem mục "Không có GPU" trong [SETUP.md](SETUP.md)).
- Git (để tải mã). Docker chỉ cần khi dùng Redis hoặc triển khai server riêng.

## 2. Cài đặt (một lần)

```powershell
git clone https://github.com/blm2006/ARES.git yolo-al
cd yolo-al
copy .env.example .env                                   # cấu hình yolo-al: sửa CAMERAS, DEVICE nếu cần
copy ares_config\ares.env.example ares_config\ares.env   # cấu hình ARES (tuỳ chọn)
Set-ExecutionPolicy -Scope Process Bypass -Force
.\install.ps1
```

`install.ps1` tự tìm Python, tạo `.venv`, dò GPU để cài PyTorch đúng bản, cài thư viện, tạo CSDL và chạy test.
Tuỳ chọn: `-Cpu` (ép bản CPU), `-NoVenv`, `-SkipTest`.

Cài thêm cho từng phần:

```powershell
.venv\Scripts\python -m pip install -r requirements-server.txt               # ARES Server (dashboard :8200)
.venv\Scripts\python -m pip install -r traffic_safety_ai\requirements.txt    # gói giao thông
```

**Model YOLO (`*.pt`) không nằm trong kho.** Ultralytics tự tải `yolov8n.pt`, `yolo11s.pt`, `yolo11n-pose.pt` ở lần chạy
đầu (cần Internet), hoặc chép các file này từ máy đã có vào thư mục gốc.

Card đời mới (ví dụ RTX 50xx) báo `CUDA error: no kernel image is available` → chạy `NANG_CAP_GPU.bat`.

## 3. Sử dụng nhanh

### 3.1 ARES — camera AI + cảnh báo hành vi

| Muốn | Bấm đúp / lệnh | Mở ở đâu |
|---|---|---|
| Một máy làm tất cả, không cần Internet | `ARES_TRON_BO.bat` | dashboard `http://127.0.0.1:8200` |
| Máy này làm **máy chủ** cho các máy khác trong mạng | `MO_TUONG_LUA.bat` (1 lần) rồi `ARES_MAY_CHU.bat` | máy khác tải app tại `http://<IP-máy-chủ>:8200/tai-app` |
| Xem IP của máy chủ | `XEM_IP.bat` | — |
| Tự chạy máy chủ khi đăng nhập Windows | `TU_KHOI_DONG.bat` | — |
| Chia sẻ ra Internet (Tailscale Funnel, có HTTPS) | `CHIA_SE_INTERNET.bat` (tắt: `CHIA_SE_INTERNET.bat tat`) | link in ra màn hình |
| Chạy thử không cần camera / phần cứng | `.\start_ares.ps1 -Demo` hoặc `ARES.bat` | `http://127.0.0.1:8100` |
| Kiểm tra cấu hình | `.\start_ares.ps1 -Check` | — |
| Thử AI hành vi trên một video | `python -m ares.behavior video duong_dan\video.mp4` | — |

- Dashboard **:8200** dành cho người quản lý / người dùng (Tổng quan, Trực tiếp, Cảnh báo, Địa điểm, Thử video).
  Dashboard **:8100** dành cho kỹ thuật viên (camera, duyệt ảnh, model, thiết bị, trang `/behavior`).
- Người trực bấm **Đúng** hoặc **Báo nhầm** trên cảnh báo; dữ liệu này dùng để train lại.
- **Không** mở cổng 8200 thẳng ra Internet bằng port forwarding (không có HTTPS) — dùng `CHIA_SE_INTERNET.bat`.
- Hệ thống chỉ phát hiện *dấu hiệu nhìn thấy được*, không chẩn đoán y tế. Chi tiết và giới hạn: [BEHAVIOR_AI.md](BEHAVIOR_AI.md).

**Đèn qua Arduino cắm USB** (thay cho ESP32 + MQTT): nạp `arduino/ares_led/ares_led.ino`, rồi thêm vào `ares_config\ares.env`:

```
ARES_BUS=serial
ARES_SERIAL_PORT=COM3      # bỏ trống để tự dò cổng Arduino
```

### 3.2 Giao thông khu vực trường học

| Muốn | Bấm đúp / lệnh |
|---|---|
| Mở **Studio**: thả video, tự vẽ vùng/vạch, bấm chạy | `MO_APP.bat` → `http://127.0.0.1:8200/studio` |
| Menu đầy đủ | `TRAFFIC.bat` |
| Chạy thử bằng mô phỏng (không cần camera, không cần model) | `.venv\Scripts\python -m traffic_safety_ai demo --open` |
| Kiểm tra thư viện, GPU, tình huống nào đang BẬT/TẮT và vì sao | `.venv\Scripts\python -m traffic_safety_ai check` |

AI **không có thẩm quyền xử phạt**: mọi hồ sơ ở trạng thái chờ xác minh, mức phạt in trên phiếu chỉ để tham khảo.
Studio và ARES Server cùng mặc định cổng **8200** — đừng mở hai thứ cùng lúc trên một máy.

### 3.3 yolo-al — vòng học chủ động

```powershell
.\start_all.ps1            # mở 3 cửa sổ: thu mẫu · worker · bảng theo dõi
.\start_all.ps1 -Review    # thêm giao diện duyệt ảnh
```

Làm theo từng phase (Phase 1 phải đạt FPS thật ≥ 25 mới đi tiếp) — xem [YOLO_AL.md](YOLO_AL.md).

## 4. Chạy test

```powershell
.venv\Scripts\python -m pytest tests
.venv\Scripts\python -m pytest traffic_safety_ai\tests
```

## 5. Những thứ không có trong kho

Các mục dưới đây nằm trong `.gitignore` vì là bí mật, dữ liệu cá nhân hoặc file nặng:

- `.env`, `ares_config/ares.env`, `ares_config/server.env` — cấu hình và khoá riêng từng máy. Dùng các file `*.example` làm mẫu;
  khoá kết nối (`ARES_UPLINK_KEY`...) xin trực tiếp người quản trị máy chủ, **không** gửi qua GitHub.
- `var/`, `traffic_safety_ai/data/`, `samples/`, `ket_qua_thu_video_mau/` — ảnh, video, clip bằng chứng, CSDL (có hình người và biển số xe).
- `*.pt`, `.venv/`, các gói `*.zip`.

## 6. Dữ liệu cá nhân

Hình ảnh người và biển số xe là dữ liệu cá nhân (Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15, Nghị định 356/2025/NĐ-CP).
Luồng hình camera không rời địa điểm lắp đặt; đừng đưa video, ảnh hoặc clip bằng chứng lên kho công khai này.
