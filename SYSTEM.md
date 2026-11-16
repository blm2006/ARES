# ARES — Hệ thống AI camera (tài liệu tổng thể)

> Sơ đồ workflow có hình minh hoạ: `WORKFLOW.html` (mở bằng trình duyệt).
> Chi tiết phần nhận diện hành vi: `BEHAVIOR_AI.md`. Cách triển khai server: `deploy/server/README.md`.
> Cập nhật: 15/11/2026.

## 1. Sản phẩm

ARES là **AI chạy tại chỗ trên camera sẵn có** của khách hàng. Hệ thống làm ba việc:

1. **Đếm người** theo camera và theo vùng.
2. **Phát hiện dấu hiệu nguy hiểm**: té ngã, nằm xuống đột ngột / nằm dưới sàn (2–5 giây, kể cả còn cử động), nằm bất động ≥ 15 giây, ngồi gục bất động, đánh nhau/xô đẩy.
3. **Báo ngay cho người trực**, kèm clip bằng chứng.

Người trực xác nhận **Đúng** hoặc **Báo nhầm**. Dữ liệu xác nhận được dùng để train lại, nên AI chính xác dần theo từng nơi lắp.

| | Gồm |
|---|---|
| **Gói cơ bản (AI)** | YOLO nhận diện người · YOLO-pose nhận diện hành vi · cảnh báo + clip · dashboard tại chỗ · server trung tâm nhiều địa điểm · vòng học |
| **Tích hợp (tuỳ chọn)** | Đèn (độ sáng theo người + ánh sáng tự nhiên) · Điều hòa (làm mát trước, setpoint theo mật độ) · sau này: giao thông khu trường học, Telegram/Zalo |

## 2. Thành phần

| Thành phần | Chạy ở đâu | Việc chính | Mã nguồn | Cổng |
|---|---|---|---|---|
| **ARES Edge** | Máy có GPU tại địa điểm | Đọc camera, YOLO người, YOLO-pose hành vi, vòng học, điều khiển thiết bị (tuỳ chọn), dashboard kỹ thuật | `ares/`, `ares/behavior/` | 8100 |
| **Uplink** | Trong Edge | Gửi heartbeat 30 s và sự kiện lên server; xếp hàng khi mất mạng; nhận ngược đánh giá | `ares/uplink.py` | — |
| **ARES Server** | VPS, máy chủ nội bộ, hoặc chính máy Edge | Nhận dữ liệu nhiều địa điểm, dashboard dễ hiểu cho người quản lý | `ares_server/` | 8200 |
| **Thiết bị** (tuỳ chọn) | ESP32/Arduino + MQTT | Đèn PWM, điều hòa IR, cảm biến | `ares/control_engine.py` | 1883 |

**Hai cách triển khai**, dùng chung một mã nguồn:

- **Local**: 1 máy chạy cả Edge và Server, không cần Internet. Khởi động bằng `ARES_TRON_BO.bat`.
- **Server**: mỗi địa điểm có 1 máy Edge, tất cả gửi lên 1 server trung tâm qua HTTPS. Server chạy bằng Docker theo `deploy/server/`.

## 3. Dashboard

| Dashboard | Dành cho | Nội dung |
|---|---|---|
| **Server** `:8200` | Người quản lý, người dùng | Quản trị viên: **Tổng quan** · **Trực tiếp** · **Cảnh báo** · **Địa điểm** · **Thử video** · **Tài khoản** · **Thiết lập**. Người dùng: **Tổng quan** · **Camera của tôi** · **Cảnh báo** · **Địa điểm** · **Thử video**. Góc phải: nút tài khoản (tên, đổi tên hiển thị, đổi mật khẩu, đăng xuất). |
| **Edge** `:8100` | Kỹ thuật viên | Giám sát, camera, duyệt ảnh, model, thiết bị, và trang `/behavior` (cảnh báo hành vi + khung xương trực tiếp) |

## 4. Dữ liệu đi đâu

```
camera ──(luồng hình, KHÔNG rời địa điểm)──> Edge
Edge ──heartbeat 30 s: số người, trạng thái AI/thiết bị──────────────> Server
Edge ──sự kiện: thông tin [+ ảnh] [+ clip] theo ARES_UPLINK_MEDIA─────> Server
Server ──Đã xử lý / Đúng / Báo nhầm (trả về trong heartbeat)──────────> Edge (dữ liệu train)
```

Edge vẫn chạy đầy đủ khi mất mạng: nhận diện, cảnh báo tại chỗ và điều khiển thiết bị đều không dừng. Sự kiện chưa gửi được nằm trong DB của Edge và được gửi bù khi có mạng lại, không tạo bản trùng.

**Pháp lý:** áp dụng Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15 (hiệu lực 01/01/2026) và Nghị định 356/2025/NĐ-CP.

## 5. Chế độ MÁY CHỦ + app xem (cách đang dùng)

Máy của bạn (có GPU) là **máy chủ**: chạy YOLO người + YOLO hành vi cho MỌI camera, chạy server dashboard mở cho mạng nội bộ.
Máy khác **không chạy AI**: cài app **ARES Viewer** (≈6 MB), bật **camera của máy đó** → hình gửi về máy chủ → AI chạy trên máy chủ →
kết quả (khung xương, số người, cảnh báo + clip) trả về **dashboard của chính người đó**.

```
Webcam máy khác ──(app ARES Viewer, 640px, tới ~12 khung/giây)──> SERVER :8200 ──> AI trên máy chủ (YOLO người + hành vi)
       ▲                                                                                   │
       └──────── dashboard của người đó: hình kết quả AI ~0,5 giây/khung, số người, cảnh báo ◄┘
Camera IP (RTSP) / USB cắm vào máy chủ ──> AI trên máy chủ (quản trị viên gán camera vào địa điểm của ai thì người đó thấy)
```

| Ai | Thấy gì |
|---|---|
| **Quản trị viên** (máy chủ) | Tất cả: Tổng quan (thẻ từng địa điểm: số người, camera hoạt động, cảnh báo), Trực tiếp (nhóm theo người dùng → địa điểm), Cảnh báo, **Địa điểm** (nhóm theo tài khoản: thêm/đổi tên/xoá địa điểm, **Chuyển tới** để gán camera), **Tài khoản** (duyệt, khoá, vai trò, địa điểm + số người của từng tài khoản), **Thiết lập** (máy chủ AI chạy GPU hay CPU, IP + link tải app, phần Nâng cao: máy Edge, khoá thiết bị, thêm camera IP) |
| **Người dùng** (máy khác) | Chỉ của mình: Tổng quan (địa điểm của tôi kèm hình AI), **Camera của tôi** (bật webcam, chọn địa điểm, hình gốc cạnh hình AI), Cảnh báo của mình, Địa điểm của mình. Không thấy người khác. |

**Địa điểm chia theo người dùng.** Mỗi tài khoản có sẵn 1 địa điểm mang tên người đó và tạo thêm được (vd: Nhà, Cửa hàng; tối đa 10).
Camera thuộc địa điểm nào thì chủ địa điểm đó thấy. Camera của máy chủ chưa gán nằm ở địa điểm của quản trị viên đầu tiên.
Số người, camera hoạt động, cảnh báo được tính **riêng từng địa điểm**. "Máy Edge / khoá thiết bị" là chi tiết kỹ thuật, chỉ còn trong Thiết lập → Nâng cao.

| Bước | Trên máy chủ (1 lần) |
|---|---|
| 1 | Chạy `MO_TUONG_LUA.bat` (tự xin quyền quản trị) — mở TCP 8200 + UDP 8201. Đặt mạng Wi-Fi/LAN là **Private**. |
| 2 | Chạy `ARES_MAY_CHU.bat` — tự nối AI với server, bật hình trực tiếp, in **IP máy chủ** và địa chỉ cho máy khác, rồi mở dashboard. |
| 3 | Lần đầu, dashboard trên máy chủ hiện **Tạo tài khoản quản trị**: đặt tên đăng nhập + mật khẩu. Tài khoản lưu trong DB server (`var/server/server.db`), mật khẩu băm PBKDF2. Chỉ máy chủ (127.0.0.1) mới tạo được tài khoản đầu tiên. |
| 4 | (Tuỳ chọn) `TU_KHOI_DONG.bat` → máy chủ tự chạy khi đăng nhập Windows. Xem lại IP bất cứ lúc nào: `XEM_IP.bat`. |

**Tài khoản:** mục **Tài khoản**: tạo tài khoản cho người khác (Quản trị viên = toàn quyền; Người dùng = camera, địa điểm, cảnh báo của riêng họ),
duyệt người tự **Đăng ký** ở màn hình đăng nhập, khoá/mở, đặt lại mật khẩu, xoá (xoá luôn địa điểm + camera của người đó).
Mỗi người bấm vào **tên mình ở góc phải** để đổi tên hiển thị, đổi mật khẩu, đăng xuất.
Quên mật khẩu quản trị: trên máy chủ chạy `python -m ares_server reset-password <tên>`.

| Bước | Trên máy khác |
|---|---|
| 1 | Mở trình duyệt `http://<IP-máy-chủ>:8200/tai-app` → tải **Windows / macOS / Linux**. File tải về đã ghi sẵn địa chỉ máy chủ. |
| 2 | Giải nén, mở **ARES Viewer** → cửa sổ dashboard riêng → đăng nhập (hoặc bấm Đăng ký rồi chờ quản trị viên duyệt). Máy chủ đổi IP: app tự dò lại (UDP 8201), không thấy thì hỏi địa chỉ. |
| — | Điện thoại: mở thẳng `http://<IP-máy-chủ>:8200`. |

Trong dashboard:
- **Trực tiếp**: xem mọi camera theo người dùng → địa điểm, ảnh AI đã vẽ khung người/khung xương (không vẽ vùng đèn), tải nối tiếp ~0,15–0,5 giây/khung.
- **Thiết lập → Nâng cao → Thêm camera**: dán địa chỉ RTSP của camera IP, máy chủ tự mở camera và chạy AI; sau đó vào **Địa điểm** gán camera cho người dùng.

**Độ trễ.** Nguyên nhân trễ lớn nhất là AI chạy CPU thay vì GPU: PyTorch 2.6 + CUDA 12.4 không có mã cho card đời mới
(báo `CUDA error: no kernel image is available`), ARES tự chuyển sang CPU để không dừng, Thiết lập → Máy chủ AI hiện cảnh báo vàng.
Sửa: tắt ARES → chạy **`NANG_CAP_GPU.bat`** (cài PyTorch bản CUDA 12.8, kiểm tra GPU) → mở lại `ARES_MAY_CHU.bat`.
Các tối ưu khác đã bật sẵn: webcam gửi 640px chất lượng 0,7 không dồn hàng đợi; máy chủ đọc camera của máy khác 15 vòng/giây (`ARES_HUB_FPS=15`);
hình trực tiếp gửi luồng riêng mỗi 0,5 giây (`ARES_UPLINK_LIVE_S=0.5`), bỏ khung không đổi, không nén lại; dashboard tải ảnh nối tiếp, chỉ ảnh đang hiện.

**Thử video** (mục riêng, không liên quan camera): ở bất kỳ máy nào, vào **Thử video** → **Chọn video…** → **Tải lên và phân tích**.
Video lên máy chủ, máy AI nhận việc (luồng riêng, không làm chậm camera trực tiếp), chạy YOLO người + YOLO hành vi 10 khung/giây, trả về:
video đã vẽ khung xương (phát ngay trong dashboard, có tua/tốc độ, bấm sự kiện để nhảy tới đúng lúc), danh sách sự kiện kèm mốc thời gian,
số người tối đa, thời gian xử lý, nút tải video kết quả. Tối đa 500 MB/video, mỗi người 3 video chờ; xoá video là máy AI dừng ngay.
Máy AI không giữ lại video sau khi xử lý. Tắt tính năng: `ARES_UPLINK_VIDEO_JOBS=false`. Thử bằng lệnh trên chính máy chủ vẫn được: `python -m ares.behavior video x.mp4`.

**Cảm biến (mục Cảm biến trong dashboard).** Mỗi mạch thuộc 1 địa điểm, giống camera; số đo hiện trên thẻ địa điểm ở Tổng quan.

| Mạch | Cách gửi | Máy cắm mạch cần cài |
|---|---|---|
| ESP32, ESP8266 (NodeMCU, Wemos) | Wi-Fi → `POST /api/v1/sensor` với khoá riêng của mạch, mỗi 5 giây; nhận lệnh relay/LED/còi trong câu trả lời | Không — nạp code 1 lần bằng Arduino IDE rồi cắm cục sạc bất kỳ |
| Arduino Uno / Nano / Mega / R4 (và mọi mạch cắm USB) | USB → dashboard đọc cổng USB (Web Serial, Chrome/Edge/ARES Viewer) → gửi lên máy chủ bằng phiên đăng nhập | Không — chỉ mở dashboard, bấm "Kết nối mạch qua USB" |
| ESP32 nối MQTT phòng máy chủ (đèn/điều hoà) | như cũ, ARES đọc qua MQTT | — (quản trị viên thấy trong mục Cảm biến) |

- Dashboard tự tạo code `.ino` cho từng mạch (đã ghi khoá + địa chỉ máy chủ; tên/mật khẩu Wi-Fi do trình duyệt điền, không gửi lên máy chủ).
  Code có sẵn: DHT11/22, DS18B20, BH1750, quang trở, PIR, MQ-2/MQ-135, lửa, âm thanh, cửa từ, HC-SR04; điều khiển relay, LED (0–100 %), còi — đổi `0`→`1` cho cảm biến đã gắn.
- Lịch sử số đo lưu tối đa 1 bản/30 giây/mạch (biểu đồ 24 giờ), xoá theo `metrics_keep_days`. Xoá mạch là khoá cũ hết hiệu lực.
- Code `.ino` chưa được biên dịch thử trên mạch thật.

**Dùng từ mạng khác (Internet) + dashboard trên GitHub Pages.**

```
Người ở mạng khác ──https──> blm2006.github.io/ARES (giao diện tĩnh)
                    └─https + token──> https://<máy-chủ>.<tailnet>.ts.net (Tailscale Funnel) ──> máy chủ ARES :8200 (AI, camera, tài khoản)
```

| Bước | Trên máy chủ (1 lần) |
|---|---|
| 1 | Cài **Tailscale** (tailscale.com/download), đăng nhập. |
| 2 | Chạy `CHIA_SE_INTERNET.bat`. Lần đầu Tailscale in 1 link để bật **Funnel** + HTTPS — mở link, bấm Enable, chạy lại. File in ra địa chỉ https cố định và **link dashboard** dạng `https://blm2006.github.io/ARES/?server=https://...ts.net`. |
| 3 | Gửi link đó cho người dùng. Họ mở bằng trình duyệt bất kỳ (mạng nào cũng được), đăng nhập, bật camera / thử video. Không cần ARES Viewer vì là https. |

- Trang GitHub Pages chỉ là giao diện; tắt máy chủ thì trang báo không kết nối được. Đăng nhập trên trang đó dùng **token** (không cookie khác trang); máy chủ chỉ cho trang `*.github.io` gọi vào (CORS), các trang khác bị chặn.
- Qua đường hầm, máy chủ không coi người ngoài là "người ngồi ở máy chủ" (không tạo được tài khoản quản trị đầu tiên từ xa), chặn đoán mật khẩu theo IP thật.
- Cập nhật giao diện trên GitHub Pages: `python -m ares_server build-web --out web` rồi đẩy thư mục `web/` lên repo.
- Tắt chia sẻ: `CHIA_SE_INTERNET.bat tat`. **Không** mở cổng 8200 thẳng ra Internet (port forwarding) — không có HTTPS.

## 6. Chạy thử nhanh

```bat
:: 1 máy làm tất cả
ARES_TRON_BO.bat

:: Hoặc từng phần
ARES_SERVER.bat                               :: server + tạo địa điểm
python -m ares_server link-local              :: nối ARES trên máy này với server cùng máy
python -m ares run                            :: AI (Edge)
python -m ares.behavior video samples\x.mp4   :: thử AI hành vi trên video
```

## 7. Kiểm thử đã chạy

- **269 test tự động pass.** Gồm luật hành vi trên khung xương tổng hợp, server (khoá thiết bị, chống trùng, chống CSRF, đăng nhập, địa điểm theo người dùng: chỉ thấy của mình, gán camera, đổi tên/xoá), AI tự chuyển CPU khi GPU lỗi, thử video tải lên (nhận job, huỷ, nhận lại khi máy AI tắt), dashboard từ xa (CORS, token, an toàn sau đường hầm), mạch cảm biến (Wi-Fi bằng khoá, USB bằng phiên đăng nhập, lệnh relay/LED, chặn người khác), uplink qua HTTP thật (mất mạng, gửi bù, khoá sai, chế độ không gửi ảnh, đồng bộ đánh giá ngược).
- **Chạy thật trên máy dev** với 2 Edge cùng lúc:
  - 1 Edge nhận video giám sát thật, chạy YOLO-pose + model đánh nhau.
  - 1 Edge mô phỏng đèn/điều hòa.
  - Cả hai gửi lên 1 server: dashboard hiện đủ cảnh báo, clip, số người, trạng thái đèn/điều hòa.

- App ARES Viewer: bản Linux chạy thật trên máy dev, tự dò được máy chủ qua UDP và mở cửa sổ dashboard. Bản Windows/macOS build chéo, **chưa chạy thử trên máy thật**.
- Camera máy khác: chạy thật trọn vòng trên máy dev — trình duyệt (vai người dùng, địa chỉ mạng nội bộ) bật webcam giả bằng video giám sát → máy chủ chạy YOLO-pose + model → dashboard người đó hiện hình AI, 5 người, cảnh báo đánh nhau kèm clip; người dùng khác không thấy; quản trị viên thấy trong mục Người dùng.

## 8. Trước khi bán: còn thiếu

1. Đo độ chính xác té ngã trên video thật (URFD + quay tại UEH), báo cáo tỉ lệ đúng/nhầm.
2. Pilot 2–4 tuần tại 1 địa điểm thật.
3. Chia sẻ 1 địa điểm cho nhiều tài khoản (hiện mỗi địa điểm thuộc đúng 1 tài khoản; quản trị viên thấy tất cả).
4. Cập nhật model từ xa (OTA), Telegram/Zalo, sao lưu và giám sát server.
5. Luật sư rà soát dữ liệu cá nhân; mẫu thông báo cho người bị ghi hình; hợp đồng xử lý dữ liệu với khách.
