# ARES — Hệ thống AI camera (tài liệu tổng thể)

> Sơ đồ workflow có hình minh hoạ: `WORKFLOW.html` (mở bằng trình duyệt).
> Chi tiết phần nhận diện hành vi: `BEHAVIOR_AI.md`. AI bảo hộ lao động: `PPE_AI.md`. Cách triển khai server: `deploy/server/README.md`.
> Cập nhật: 17/11/2026 (thêm 3 phần trên dashboard: Hành vi · Người & môi trường · Giao thông).

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

## 3b. Dashboard chia 3 phần (+ Hệ thống)

Thanh trên cùng của dashboard máy chủ `:8200` chọn **phần**, thanh thứ hai là các mục của phần đó:

| Phần | Mục | Ghi chú |
|---|---|---|
| **Hành vi** | Tổng quan · Trực tiếp / Camera của tôi · Cảnh báo · Thử video · **Train lại AI** (quản trị) | Té ngã, nằm/gục bất động, đánh nhau, đồ bảo hộ |
| **Người & môi trường** | Tổng quan (số người, nhiệt độ/độ ẩm, ánh sáng, đèn/điều hoà) · Trực tiếp / Camera của tôi · Cảm biến · **Thử video** (chia vùng + đèn) | YOLO người + cảm biến |
| **Giao thông** (quản trị) | Tổng quan · Thử video (Studio) · Xử lý vi phạm | Gói `traffic_safety_ai` chạy ngay trong server ở `/traffic` — không cần mở `MO_APP.bat` riêng (2 cái cùng dùng cổng 8200) |
| **Hệ thống** | Địa điểm · Tài khoản · Thiết lập | |

**Thử video — Người & môi trường.** Chọn video → khung hình đầu hiện ra → **+ Vẽ vùng** (bấm các góc, bấm lại điểm đầu để đóng, kéo chấm trắng để sửa)
→ **+ Thêm đèn**: tên, vùng của đèn, **mạch** (mạch ESP32/Arduino ở mục Cảm biến hoặc "chỉ mô phỏng"), cổng (relay bật/tắt hoặc LED độ sáng %)
→ đặt "tắt đèn khi vùng vắng N giây" → **Tải lên và phân tích**. Kết quả: video có vùng (vàng = đèn đang bật), khung người mảnh,
mốc bật/tắt từng đèn, thời gian bật, % tiết kiệm so với bật cả buổi. Tick **Điều khiển mạch thật** khi xem kết quả thì đèn thật bật/tắt đúng nhịp video
(đóng cửa sổ là tắt). **Sửa vùng/đèn** ở bảng video rồi **Lưu & phân tích lại** — không phải tải lại video. Bố trí được nhớ trên máy (Lưu bố trí).
Vùng đánh dấu **Cửa ra vào** = chỗ người được phép "xuất hiện" giữa khung hình (xem bám vết bên dưới). Mã: `ares/people_video.py`, `ares_server/layout.py`.

**Bám vết người (mọi phần thử video + camera hành vi).** Người mới chỉ được nhận nếu: có sẵn trong **2 giây đầu**, hoặc đi vào từ **rìa khung hình**
(cách mép < 6 %), hoặc xuất hiện trong vùng **Cửa ra vào**. Khung người mọc ra giữa khung hình:
trùng phần lớn với người đang có (1 người vỡ thành 2 khung) → **bỏ**; gần chỗ 1 người vừa mất dấu ≤ 6 giây → **nối lại đúng mã cũ**;
còn lại chỉ là ứng viên, phải thấy liên tục **4 giây** mới được nhận (người ngồi khuất sau bàn rồi đứng dậy).
Chỉnh trong `ares_config/ares.env`: `ARES_BEHAVIOR_TRACK_GATE` (true/false), `_WARMUP_S`, `_EDGE`, `_CONFIRM_S` (0 = không bao giờ nhận người mọc giữa khung), `_RECOVER_S`.
Không giới hạn số người / khung (`ARES_BEHAVIOR_MAX_DET=1000`). Khung người, khung xương và chữ trên hình kết quả đã vẽ mảnh hơn.

**Train lại AI từ đánh giá cảnh báo** (Hành vi → Train lại AI, chỉ quản trị viên). Bấm **Đúng / Báo nhầm** ở Cảnh báo → đánh giá về máy AI.
Bấm **Train lại ngay** → server gửi lệnh trong heartbeat → máy AI train ở luồng nền: model hành vi dùng dataset công khai (nếu đã tải) + mọi sự kiện
đã đánh giá (Đúng = mẫu của loại đó, Báo nhầm = mẫu bình thường), xong **nạp model mới ngay**; "YOLO người" chạy vòng học chủ động sẵn có (có cổng kiểm định).
Dashboard hiện số đánh giá theo loại, nhật ký train đang chạy, kết quả (macro-F1) và lịch sử. Cần ≥ 2 clip/sự kiện có khung xương. Mã: `ares/remote_train.py`.

**Model trong phần Hành vi (18/11/2026).** YOLO-pose (khung xương) · ST-GCN hành vi (train từ đánh giá) · đồ bảo hộ = SH17 + Construction-PPE
chạy cùng lúc (nón, kính, áo phản quang, chụp tai, khẩu trang, găng, giày) · **cháy / khói** · **hút thuốc**. Cài/kiểm: `CAI_AI_BAO_HO.bat`.
Danh sách model đang chạy xem ở Hành vi → Train lại AI. Chi tiết, giấy phép, cách train dần: `BEHAVIOR_AI.md`, `PPE_AI.md`.

**Vùng & đèn cho camera thật** (Người & môi trường → Vùng & đèn, 18/11/2026). Chọn camera (camera IP của máy chủ hoặc webcam
"Camera của tôi" ở máy khác) → hình trực tiếp hiện ra → vẽ vùng → **+ Thêm đèn**: vùng của đèn, **mạch** + cổng (Relay bật/tắt hoặc LED %),
chế độ Tự động / Luôn bật / Luôn tắt → **Lưu & áp dụng**.

```
máy AI: YOLO người + bám vết -> điểm giữa chân -> đếm người từng vùng -> POST /api/v1/zones mỗi 1 giây (ARES_UPLINK_ZONE_S)
server: ares_server/zones.py — vùng có người: BẬT; vắng đủ N giây: TẮT; mất số liệu > 15 giây: Tự động BẬT để an toàn
mạch:   ESP32/ESP8266 nhận lệnh khi gửi số đo (2 giây/lần) · Arduino USB ở MÁY KHÁC: dashboard mở trên máy đó hỏi lệnh mỗi giây, ghi xuống USB
```

- Bố trí lưu trên server (bảng `cam_layouts`), máy AI nhận trong ≤ 5 giây. Người dùng chỉ sửa được camera và mạch của mình.
- Nhiều đèn chung 1 cổng mạch: bật nếu 1 đèn bật (LED lấy độ sáng lớn nhất). Bấm lệnh tay ở mục Cảm biến không bị ghi đè cho tới khi trạng thái đèn đổi.
- Vùng **Cửa ra vào** cũng được dùng cho bám vết camera thật (người mới chỉ "xuất hiện" ở rìa khung hoặc qua cửa).
- Mạch USB: máy cắm mạch phải mở dashboard (app ARES Viewer, hoặc Chrome/Edge qua https) và bấm **Kết nối USB trên máy này**; đóng dashboard là mạch thôi nhận lệnh.
- **Nạp lại code cho ESP32** đã nạp trước 18/11 để có chu kỳ 2 giây (bản cũ 5 giây vẫn chạy, chỉ chậm hơn).

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

**Nhãn tư thế trên hình.** Mọi hình kết quả AI (camera trực tiếp, clip cảnh báo, Thử video) đều vẽ: khung người theo mức
(xanh an toàn · cam chú ý · đỏ báo động), khung xương màu theo bộ phận, nhãn `ID 1 · NGỒI 92%` (Đứng / Đang đi / Ngồi / Nằm / Ngồi gục /
Thấy nửa người), và thanh trạng thái phía trên: `TRẠNG THÁI: AN TOÀN · 2 người: 1 đứng, 1 ngồi` → `CHÚ Ý: Mất thăng bằng? (người #1)` →
`BÁO ĐỘNG: TÉ NGÃ! (người #1)` kèm viền đỏ. % là độ chắc của tư thế (độ đồng thuận 1 giây gần nhất × độ tin cậy các khớp), không phải xác suất té ngã.
Tắt thanh trạng thái: `ARES_BEHAVIOR_OVERLAY_STATUS=false`.

**AI bảo hộ (đồ bảo hộ lao động).** Mỗi địa điểm chọn món **bắt buộc** / **cấm** (Địa điểm → Chọn đồ bảo hộ; có mẫu Xưởng gỗ,
Công trường, Phòng thí nghiệm). Mẫu Xưởng gỗ: bắt buộc kính + chụp tai + khẩu trang, cấm găng tay. Người thiếu liên tục ~5 giây → cảnh báo
"Vi phạm đồ bảo hộ" kèm clip; trên hình ghi `THIẾU KÍNH, CHỤP TAI` / `ĐỦ BẢO HỘ`. Chỉ kết luận khi thấy rõ người. Cài model 1 lần trên máy chủ:
`CAI_AI_BAO_HO.bat`. Model mặc định (SH17) chỉ dùng phi thương mại. Thử video cũng chọn được "Kiểm đồ bảo hộ". Chi tiết: `PPE_AI.md`.

**Thử video** (mục riêng, không liên quan camera): ở bất kỳ máy nào, vào **Thử video** → **Chọn video…** → **Tải lên và phân tích**.
Video lên máy chủ, máy AI nhận việc (luồng riêng, không làm chậm camera trực tiếp), chạy YOLO người + YOLO hành vi 10 khung/giây, trả về:
video đã vẽ khung xương (phát ngay trong dashboard, có tua/tốc độ, bấm sự kiện để nhảy tới đúng lúc), danh sách sự kiện kèm mốc thời gian,
số người tối đa, thời gian xử lý, nút tải video kết quả; thêm thẻ trạng thái đổi theo từng khung (An toàn / Chú ý / Báo động), dải thời gian tô màu theo mức (bấm để tua) và tổng thời gian từng tư thế. Tối đa 500 MB/video, mỗi người 3 video chờ; xoá video là máy AI dừng ngay.
Máy AI chỉ nhận video khi đã tải lên **đủ** (trạng thái *Đang tải lên* → *Đang chờ*); video không đọc được khung nào thì báo **Lỗi** kèm lý do, không báo "Xong" rỗng. Nút **Phân tích lại** cho AI chạy lại trên video đã tải lên (không phải tải lại).
Máy AI không giữ lại video sau khi xử lý. Tắt tính năng: `ARES_UPLINK_VIDEO_JOBS=false`. Thử bằng lệnh trên chính máy chủ vẫn được: `python -m ares.behavior video x.mp4`.

**Khởi động lại từ xa.** `ARES_MAY_CHU.bat` chạy server và máy AI dưới một "người giám sát": quản trị viên bấm **Thiết lập → Khởi động lại máy chủ** (hoặc file `var/restart.flag` đổi) thì cả hai tự dừng gọn rồi mở lại bằng mã mới nhất, khoảng 1 phút, không cần ngồi ở máy chủ. Tắt hẳn vẫn là Ctrl+C / đóng cửa sổ.

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

- **296 test tự động pass** (thêm: cổng sinh người của tracker, thử video vùng + đèn trọn vòng server ↔ máy AI, train lại theo lệnh, Giao thông gắn ở /traffic chỉ cho quản trị viên). Trước đó: Gồm luật hành vi trên khung xương tổng hợp, server (khoá thiết bị, chống trùng, chống CSRF, đăng nhập, địa điểm theo người dùng: chỉ thấy của mình, gán camera, đổi tên/xoá), AI tự chuyển CPU khi GPU lỗi, thử video tải lên (nhận job, huỷ, nhận lại khi máy AI tắt, không nhận khi đang tải dở, 0 khung = lỗi, phân tích lại), khởi động lại từ xa, dashboard từ xa (CORS, token, an toàn sau đường hầm), đồ bảo hộ (ánh xạ lớp model, ghép đồ với người, thiếu/cấm, luật theo địa điểm tới máy AI), mạch cảm biến (Wi-Fi bằng khoá, USB bằng phiên đăng nhập, lệnh relay/LED, chặn người khác), uplink qua HTTP thật (mất mạng, gửi bù, khoá sai, chế độ không gửi ảnh, đồng bộ đánh giá ngược).
- **Chạy thật trên máy dev** với 2 Edge cùng lúc:
  - 1 Edge nhận video giám sát thật, chạy YOLO-pose + model đánh nhau.
  - 1 Edge mô phỏng đèn/điều hòa.
  - Cả hai gửi lên 1 server: dashboard hiện đủ cảnh báo, clip, số người, trạng thái đèn/điều hòa.

- App ARES Viewer: bản Linux chạy thật trên máy dev, tự dò được máy chủ qua UDP và mở cửa sổ dashboard. Bản Windows/macOS build chéo, **chưa chạy thử trên máy thật**.
- Camera máy khác: chạy thật trọn vòng trên máy dev — trình duyệt (vai người dùng, địa chỉ mạng nội bộ) bật webcam giả bằng video giám sát → máy chủ chạy YOLO-pose + model → dashboard người đó hiện hình AI, 5 người, cảnh báo đánh nhau kèm clip; người dùng khác không thấy; quản trị viên thấy trong mục Người dùng.

## 8. Trước khi bán: còn thiếu

1. Đo độ chính xác té ngã trên video thật (URFD + quay tại UEH), báo cáo tỉ lệ đúng/nhầm. Đo AI bảo hộ tại xưởng thật; train model đồ bảo hộ riêng (model SH17 mặc định chỉ dùng phi thương mại).
2. Pilot 2–4 tuần tại 1 địa điểm thật.
3. Chia sẻ 1 địa điểm cho nhiều tài khoản (hiện mỗi địa điểm thuộc đúng 1 tài khoản; quản trị viên thấy tất cả).
4. Cập nhật model từ xa (OTA), Telegram/Zalo, sao lưu và giám sát server.
5. Luật sư rà soát dữ liệu cá nhân; mẫu thông báo cho người bị ghi hình; hợp đồng xử lý dữ liệu với khách.
