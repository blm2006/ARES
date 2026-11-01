# ARES — nâng cấp "ai detech" thành hệ thống BEMS theo hiện diện

Tài liệu này mô tả phần **ARES** thêm vào dự án `yolo-al`: camera + YOLO đếm người theo vùng → điều khiển
đèn (PWM từng vùng) và điều hòa (IR) → học chủ động có người duyệt → retrain ban đêm → dashboard.
Đặc tả gốc: `ARES.md`. Mã nguồn: thư mục `ares/`. Phần `collector/ worker/ review/ dataset/ training/` cũ
**không bị sửa** và vẫn chạy như trước (xem mục 3).

---

## 1. Trạng thái: cái gì đã kiểm chứng, cái gì chưa

| Thành phần | Kiểm chứng bằng | Ghi chú |
|---|---|---|
| Công thức (PWM, t_lead, setpoint, deadband, EMA R_cool), bộ đệm, vùng, lịch, thói quen | 244 test tự động (`pytest tests/ares`) | Chạy offline, không cần camera/ESP32 |
| Bộ điều khiển đèn + điều hòa + pre-cooling + fail-safe | Test đơn vị + mô phỏng không đầu nhiều ngày | Vật lý phòng trong mô phỏng là **mô hình giả định** |
| YOLO thật (`YoloDetector`) | Chạy `yolov8n.pt` trên ảnh có người: phát hiện 3 người | Chưa thử với camera/RTSP thật |
| Retrain thật (tiến trình con ultralytics) | Chạy thật trên CPU, 1 epoch, nhãn giả: huấn luyện → đo → **cổng chất lượng từ chối** model kém → sổ model ghi `rejected`, `base` giữ nguyên | Chưa thử trên GPU và dữ liệu thật |
| Dashboard (giám sát, duyệt ảnh, model, thói quen) | Test API + khởi động HTTP thật | Giao diện chưa được ai dùng thử trên trình duyệt thật ngoài test |
| MQTT / ESP32 | Test với broker/thiết bị **giả** | **Chưa** chạy với broker + ESP32 thật; firmware ESP32 nằm ngoài phạm vi (chỉ có hợp đồng ở mục 5) |
| Số tiết kiệm điện | Chỉ từ mô phỏng | **Không phải số đo thực tế** — xem mục 9 |

Test cũ của `yolo-al` (73 test) vẫn xanh (cần Python 3.11). Tổng cộng 317 test.

---

## 2. Chạy nhanh

```powershell
pip install -r requirements.txt          # thêm paho-mqtt, tzdata (Windows), httpx (dev)
copy ares_config\ares.env.example ares_config\ares.env     # rồi sửa
.\start_ares.ps1 -Check                  # kiểm tra thư viện, CUDA, zones.json, lịch, weights
.\start_ares.ps1 -Demo                   # mô phỏng thời gian thực x30 + dashboard http://127.0.0.1:8100
.\start_ares.ps1 -Simulate -Days 4 -Out bao_cao_mo_phong   # báo cáo so với baseline
.\start_ares.ps1                         # chạy THẬT: camera + YOLO + MQTT + dashboard
.\start_ares.ps1 -Retrain                # retrain ngay
```

Tương đương không qua PowerShell: `python -m ares check|demo|simulate|run|zones|retrain`.
`python -m ares zones --snapshot anh.jpg` vẽ các vùng lên một ảnh chụp từ camera để chỉnh đa giác.

**SimulationMode** (`--sim` hoặc `ARES_SIMULATION_MODE=true`): cảm biến, camera, đèn và điều hòa đều là giả;
không cần ESP32/broker. Dữ liệu mô phỏng đi vào thư mục riêng `var/ares_sim`, không lẫn dữ liệu thật.

---

## 3. Kiến trúc

```
Camera ─► VisionTracker ─► (YOLO, luồng riêng, không chặn) ─► ZoneMap (điểm giữa-đáy bbox → Khu 1..4)
              │                          │
              │                          └─► ActiveLearner (conf 0.35–0.55 → hàng đợi → /pending)
              ▼
        SharedState ◄── MqttBus ◄── ESP32 #1 (DHT22 trong/ngoài, BH1750 cửa sổ + trần)
              │
   ┌──────────┴───────────┐
   ▼                      ▼
Bộ đệm 30 s           Bộ đệm 10 phút
LightingController    HvacController (pre-cooling, setpoint động, fail-safe)
   │ PWM 4 vùng           │ lệnh IR (nguyên độ, deadband ±0,7)
   └──────► MqttBus ──────┴──► ESP32 #2 (PWM LED + IR blaster)

DBManager (SQLite/PostgreSQL): telemetry_logs · habit_profiles · active_learning_queue · system_events
NightlyJobs (00:00): gom thói quen → dọn dữ liệu cũ → retrain → đăng ký model
Dashboard (FastAPI): giám sát · biểu đồ · duyệt ảnh · model · thói quen
```

| Module | Vai trò | Ghi chú |
|---|---|---|
| `vision_tracker.py` | Đọc khung hình, chạy detector, xếp người vào vùng, đẩy vào bộ đệm | Luồng riêng; mất camera thì báo "mất", không đoán |
| `control_engine.py` | `LightingController`, `HvacController` (hàm thuần) + `ControlEngine` (keo dính) | Mọi quyết định có ghi chú tiếng Việt |
| `formulas.py` | Công thức đúng như spec, có test từng công thức | |
| `buffers.py` | Trung bình có trọng số thời gian + độ phủ dữ liệu | "Không có dữ liệu" ≠ "phòng trống" |
| `mqtt_client.py` | paho-mqtt ≥ 2.0, LWT, lệnh retain | Mất broker thì tự nối lại ngầm |
| `db_manager.py` | 4 bảng, ghi bất đồng bộ qua hàng đợi | Lỗi ghi được đếm, không làm sập vòng điều khiển |
| `active_learner.py`, `retrain.py`, `train_job.py`, `model_registry.py`, `nightly.py` | Học chủ động và retrain | Mục 6 |
| `simulation.py`, `simrun.py` | Thế giới giả + chạy không đầu + báo cáo | Mục 9 |
| `runtime.py` | Lắp ráp và chạy các luồng | `status()` cấp dữ liệu cho dashboard |

### Quan hệ với phần cũ của yolo-al

| Cũ | Mới |
|---|---|
| `realtime/camera.py` | **Dùng lại** (`CvFrameSource` bọc nó) |
| `realtime/detector.py` (ngưỡng conf riêng) | `ares/detector.py`: chạy conf thấp 0,35 để còn thấy dải bất định 0,35–0,55; tự nhận ultralytics ≥ 8.4 (`quantize=16`) hay cũ (`half=True`); chỉ lớp `person` |
| `collector/ worker/ review/ dataset/ training/` + Redis | **Giữ nguyên, chạy độc lập.** Vòng học chủ động của ARES là một vòng riêng, nhẹ hơn (SQLite + duyệt trên dashboard ARES). **Chưa có cầu nối tự động** giữa hai vòng — nếu muốn gộp, cần quyết định riêng (xem mục 10) |

Các nguyên tắc của Workflow v2 vẫn được giữ trong vòng ARES: **chỉ nhãn do người duyệt** mới vào tập huấn luyện,
lưu **ảnh gốc full-frame** (JPEG q95, không resize), luôn train **từ checkpoint gốc** (không nối tiếp qua nhiều đêm),
trộn **replay 70 % cũ / 30 % mới**, tập giữ lại chia **theo nhóm thời gian** (ngày + khối 3 giờ) để không rò rỉ,
và model mới luôn được **đo** trên tập giữ lại (từ 2026-11-01 mặc định hoán đổi ngay, cổng chỉ cảnh báo — mục 13).

---

## 4. Cấu hình

- `ares_config/ares.env` (mẫu: `ares.env.example`), biến môi trường tiền tố `ARES_`. Mọi ngưỡng trong spec đều là tham số, không hardcode.
- `ares_config/zones.json`: đa giác từng vùng (toạ độ chuẩn hoá 0..1 theo khung hình), `capacity`, hệ số hiệu chuẩn `k`, `n_max`.
- `ares_config/schedule.json`: lịch lớp (thứ 2 = Thứ Hai … 8 = Chủ nhật), ngày huỷ/nghỉ. Sửa khi đang chạy: tự nạp lại sau ~30 giây.
- **Cả hai tệp trong repo là MẪU** (2×2 vùng, 4 buổi/ngày). Phải thay bằng phòng thật trước khi chạy thật.

---

## 5. MQTT — hợp đồng với hai ESP32

Tiền tố mặc định `ares`, `<room>` = `ARES_ROOM_ID` (vd `P101`).

| Topic | Chiều | Nội dung |
|---|---|---|
| `ares/<room>/sensors` | ESP32 #1 → ARES | `{"temp_in":27.1,"temp_out":33.0,"humidity":68,"lux_window":5200,"lux_zones":[115,120,95,40]}` (chấp nhận `lux_ceiling` thay cho `lux_zones`). Giá trị ngoài dải vật lý bị bỏ, không làm hỏng cả gói |
| `ares/<room>/cmd/led` | ARES → ESP32 #2 | `{"pwm":[45,50,80,15],"fade_ms":2000}` — PWM % từng vùng. **retain** |
| `ares/<room>/cmd/ac` | ARES → ESP32 #2 | `{"power":1,"temp":25,"fan":2,"mode":"cool"}` — nhiệt độ **nguyên độ**. **retain** |
| `ares/<room>/status/<thiết bị>` | thiết bị → ARES | `online` / `offline`, retain + LWT |
| `ares/<room>/status/ares` | ARES → mọi người | `online` / `offline`, retain + LWT |

Ghi chú cho firmware ESP32 (chưa có trong repo):

1. **ESP32 #2 phải tự bảo vệ khi mất ARES.** ARES gửi lại trạng thái đèn mỗi `ARES_LED_HEARTBEAT_S` (60 s). Nếu quá vài phút không có lệnh/heartbeat, nên rơi về mức an toàn cố định (ví dụ 70 % đèn) thay vì giữ nguyên hoặc tắt, để lớp học không tối đen.
2. Lệnh `retain` nên được áp dụng ngay khi khởi động lại/nối lại broker.
3. Điều hòa IR là **một chiều**: ARES coi trạng thái điều hòa = trạng thái lệnh gần nhất đã gửi, không có phản hồi. Nếu ai đó bấm remote tay, ARES sẽ không biết. Deadband ±0,7 °C và giãn cách tối thiểu 5 phút giữa hai lệnh không khẩn cấp giúp tránh bíp liên tục.
4. Broker: `docker compose --profile ares up -d` (Mosquitto). **Bắt buộc có tài khoản** — làm theo hướng dẫn đầu `deploy/mosquitto/mosquitto.conf` để tạo tệp mật khẩu, rồi điền `ARES_MQTT_USERNAME/PASSWORD`.

---

## 6. Hành vi điều khiển và các quyết định đã chọn

### Chiếu sáng (mỗi 15 s, bộ đệm 30 s)
- Vùng "có người" khi trung bình 30 s ≥ `occ_on_threshold` = **0,35** (khoảng 1/3 số khung có người) — một lần nhận nhầm đơn lẻ không bật đèn.
- Vùng có người: đích 400 lux (450 lux khi > 8 người); vùng trống ≤ 3 phút: giữ mức cũ (GRACE); trống > 3 phút: hạ về nền 10–15 % PWM, đích 100 lux; **cả phòng** trống > 3 phút: 0 %.
- `L_actual = lux_trần × k` (k ≈ 3,0–3,5, hiệu chuẩn từng vùng). Daylight harvesting nằm tự nhiên trong vòng lặp: nắng làm `L_actual` cao → PWM giảm.
- Cảm biến lux hỏng > 5 phút → PWM dự phòng; **mất camera trong giờ học** → coi mọi vùng có người (thà thừa điện còn hơn lớp tối); mất camera ngoài giờ → giữ nguyên.

### Điều hòa (mỗi 30 s kiểm kế hoạch, cập nhật setpoint theo `hvac_interval_min`, bộ đệm 10 phút)
- **Pre-cooling:** nhìn trước `max(30, lead_max)` phút; nếu xác suất có người > 0,75 (thói quen đã học, hoặc lịch cho 0,9 khi chưa đủ dữ liệu) thì bật lúc `t_class − t_lead`, `t_lead = (T_in − 25,5)/R_cool + 0,8·max(0, T_out − 30)`, kẹp 0–60 phút. `R_cool` cập nhật bằng EMA (0,8/0,2) sau mỗi lần pre-cool.
- **Fail-safe:** đến lớp + 15 phút mà trung bình 10 phút vẫn "trống" → tắt điều hòa và ghi sự kiện.
- **Setpoint động:** `clamp(round(26,5 − 2,5·N/N_max − 0,5[RH>70] − 0,5[vùng quá đông]), 22, 27)`, quạt theo tải; **deadband ±0,7 °C**; làm tròn nửa lên.
- Người dùng nhìn thấy tất cả lý do trong "ghi chú" của từng quyết định trên dashboard.

### Học chủ động và retrain ban đêm
- Detection có conf 0,35–0,55 → lưu **ảnh gốc** + nhãn tạm vào `pending`. **Không còn trần 50 ảnh/ngày** (`ARES_AL_DAILY_CAP=0`) — việc giảm số lượng giao cho lọc PCA hai giai đoạn (mục 8b). Vẫn giữ giãn cách ≥ 60 s, trần giờ và bỏ ảnh trùng khít (aHash).
- Dashboard: **Duyệt / Sửa bbox / Loại** (công cụ vẽ-kéo-đổi cỡ-xoá bbox, phím tắt A/S/R/Del/mũi tên).
- 00:00 (`ARES_RETRAIN_TIME`): `NightlyJobs` gom thói quen → dọn dữ liệu cũ → retrain. Retrain chạy ở **tiến trình con** (không tranh GIL với vision, chết một mình nếu CUDA lỗi), có timeout. Nếu máy tắt đúng nửa đêm, việc đêm được **bù** trong vòng 6 giờ sau giờ hẹn (không train giữa giờ học).
- Model mới có trạng thái `candidate → production | rejected | archived`. **Mặc định `ARES_AUTO_PROMOTE=always`** (theo workflow 2026-11-01): train xong là hoán đổi ngay. Cổng chất lượng (`recall`, `mAP50` không kém model cũ quá `gate_tolerance` 0,01 trên tập giữ lại + replay) vẫn được **tính và lưu**; nếu model mới kém hơn thì phát cảnh báo `MODEL_GATE_WARN` — bấm "Quay lui phiên bản trước" ở tab Model. Muốn chặn model kém như trước: `ARES_AUTO_PROMOTE=gated`.
- Hoán đổi **không ngắt nhận diện**: model mới được nạp ở luồng nền, vision vẫn chạy model cũ tới khi nạp xong rồi mới đổi (sự kiện `MODEL_SWAPPED`).
- Trong lúc train, camera chính tự hạ xuống `ARES_VISION_FPS_TRAINING` (1 FPS), camera phụ xuống `ARES_HUB_FPS_TRAINING`; train xong tự trả về `ARES_VISION_FPS` (15) — sự kiện `FPS_THROTTLE` / `FPS_RESTORED`. Bộ đệm là trung bình theo thời gian nên đổi FPS không làm lệch số người.
- **Nên đặt `ARES_REPLAY_DIR`** (thư mục YOLO có `images/` và `labels/` của dataset gốc). Không có replay thì chỉ train trên ảnh người duyệt — dễ quên (catastrophic forgetting); hệ thống sẽ cảnh báo.

---

## 7. CSDL

Mặc định SQLite (WAL) trong `storage_root`; đặt `ARES_DB_URL=postgresql+psycopg://...` để dùng PostgreSQL.

- `telemetry_logs` — mỗi phút: `occ_total` (NULL = **mất camera**, không phải "0 người"), `occ_zones`, nhiệt/ẩm, lux, `ac_state`, `led_pwm`.
- `habit_profiles` — `(day_of_week 2..8, time_slot 15 phút)`: `occ_probability`, `expected_headcount`, `avg_cooling_rate`. Gom lại mỗi nửa đêm từ ~8 tuần gần nhất, bỏ các dòng mất camera.
- `active_learning_queue` — `PENDING | APPROVED | REJECTED | TRAINED`, đường dẫn ảnh/nhãn, lý do loại.
- `system_events` — nhật ký sự kiện (khởi động, pre-cool, fail-safe, retrain, promote…); cũng là chỗ lưu dấu "việc đêm đã chạy" để khởi động lại không chạy lặp.

---

## 8. Dashboard

`http://127.0.0.1:8100` — giao diện tiếng Việt, một tệp HTML, không CDN, sáng/tối theo hệ thống. Bảy tab: Giám sát, **Camera**, Biểu đồ, Duyệt ảnh, Model, Thói quen, **Thiết bị** (chỉnh delay / vùng điều chỉnh — mục 13).

**Bảo mật** (ảnh trong dashboard có người thật):
- Mặc định chỉ nghe `127.0.0.1`. Mở ra ngoài (`ARES_DASHBOARD_HOST=0.0.0.0`) mà **không có** `ARES_DASHBOARD_TOKEN` thì ARES **từ chối khởi động**.
- Có token: gửi qua header `X-ARES-Token` hoặc `?token=` (lần đầu sẽ đặt cookie HttpOnly, SameSite=Strict).
- Mọi request ghi (POST) cần thêm header `X-ARES: 1` (chống CSRF); `image_id` được kiểm tra chặt (không đi lên thư mục khác).
- Token nằm trên HTTP thường, **không mã hoá** — chỉ nên dùng trong LAN tin cậy hoặc qua reverse proxy HTTPS.

API thêm: `/api/cameras`, `/api/cameras/{id}/frame.jpg`, `POST /api/cameras/rescan`, `/api/pca/status`, `POST /api/pca/fit|triage|release`, `/api/pca/train-preview`, `POST /api/al/items/{id}/restore`.

API: `/api/status`, `/api/frame.jpg`, `/api/history`, `/api/events`, `/api/habits`, `/api/al/stats`, `/api/al/items[/{id}[/image|approve|edit|reject]]`, `/api/models`, `POST /api/models/rollback`, `POST /api/models/{version}/promote`, `POST /api/retrain`.

---

## 8b. Nhiều camera (lưới CCTV) và lọc PCA hai giai đoạn

**Tab Camera.** ARES tự đếm camera và xếp lưới: 1 cam → toàn màn hình, 2 → 2×1, 4 → 2×2, 6 → 3×2, 9 → 3×3, 16 → 4×4 (cột = ⌈√n⌉). Mỗi ô có tên, chấm online/offline và **số người** camera đó đếm được; bấm vào ô để phóng to, `Esc` để thoát; có thể ép bố cục khác và nút "Dò lại camera". Camera điều khiển (`main`) vẫn do VisionTracker giữ (không mở trùng thiết bị); camera phụ gồm USB cắm thêm (tự dò) và danh sách trong `ares_config/cameras.json` (xem `cameras.example.json`; mật khẩu RTSP bị che trên API). Một luồng suy luận chạy vòng tròn qua các camera phụ (`ARES_HUB_FPS` vòng/giây, `ARES_HUB_FPS_TRAINING` khi đang train), ảnh khó từ **mọi** camera đều vào hàng học chủ động. Camera phụ **chỉ đếm người để hiển thị và học**, không tham gia điều khiển đèn/điều hòa.

**Lọc PCA.** Ảnh thô (embedding `hist_v1` 512 chiều, lưu float16) được giữ nguyên; PCA (tái dùng `worker/pca.py`: fit một lần rồi đóng băng) chỉ dùng để so khoảng cách.
1. *Trước khi người duyệt*: ảnh mới là `CANDIDATE` → sau ~60 s gom mẻ, chạy farthest-first: ảnh gần một ảnh đã có (đang chờ/đã duyệt/đã train/đã loại) hơn bán kính → **xoá ngay** cả file lẫn dòng CSDL (mặc định `ARES_DUPLICATE_TTL_DAYS=0`, theo workflow — lọc nhầm thì không lấy lại được); đặt `> 0` để giữ dạng `DUPLICATE` chừng ấy ngày và khôi phục được từ tab Duyệt ảnh. Còn lại → `PENDING`.
2. *Sau khi duyệt, trước khi train*: chỉ trong các ảnh APPROVED mới, giữ tập đại diện (ảnh đã sửa bbox **luôn** được giữ; ảnh đã train làm "hạt giống" nên cảnh đã train không bị chọn lại). Ảnh bị bỏ vẫn nằm trong kho, không vào train **cũng không vào tập giữ lại** (tránh rò rỉ). Dưới 20 ảnh mới thì giữ hết.
- Khởi động lạnh: PCA tự fit khi đủ 300 khung mẫu (mỗi camera lấy 1 khung/30 s); chưa đủ thì ảnh chờ ở `CANDIDATE`. Dashboard có nút "Khởi tạo PCA ngay" (≥ 60 mẫu) và "Bỏ qua lọc".
- Bán kính tự hiệu chỉnh theo phân phối khoảng cách láng giềng gần nhất của chính camera (phân vị 0,5 và 0,8, có sàn 0,01/0,02) — **là heuristic, chưa kiểm chứng trên dữ liệu thật của bạn**; chỉnh `ARES_PCA_Q_REVIEW` / `ARES_PCA_Q_TRAIN` nếu lọc quá tay hay quá nhẹ.
- Hạn chế: embedding màu-histogram mô tả *bố cục/ánh sáng cảnh*, không biết "có người hay không". Hai cảnh khác người nhưng cùng màu có thể bị coi là trùng. Muốn chặt hơn nên đổi sang embedding của chính YOLO (backbone) — chưa làm.

---

## 9. Mô phỏng và kết quả

`python -m ares simulate` chạy nhiều ngày trong vài chục giây (đồng hồ giả, vật lý phòng giả, detector giả có bỏ sót ~10–20 % và dương tính giả ~12 %/khung) và so với **baseline**: đèn 100 % + điều hòa 25 °C từ 30 phút trước buổi đầu đến 30 phút sau buổi cuối mỗi ngày có lớp.

Một lần chạy mẫu (4 ngày từ thứ Sáu 2026-11-06 05:30, seed 42, bước 5 s, lịch mẫu `schedule.json`):

| | ARES | Baseline | Chênh |
|---|---|---|---|
| Đèn | 2,44 kWh | 5,88 kWh | −58,5 % |
| Điều hòa | 30,20 kWh | 31,24 kWh | −3,3 % |
| Tổng | 32,64 kWh | 37,12 kWh | −12,1 % |

Khi có người: 0 phút > 26,5 °C; thiếu sáng (< 300 lux) chỉ 6 phút-vùng; sai số đếm người trung bình ≈ 2 người.

**Đọc các số này thế nào cho đúng:**
- Đây là kết quả của **mô hình giả định** (hệ số truyền nhiệt, công suất điều hòa, quỹ đạo người…), dùng để kiểm tra logic điều khiển — **không phải số đo thực tế** và không nên đưa vào báo cáo như hiệu quả tiết kiệm của phòng thật.
- Tiết kiệm điều hòa nhỏ vì lịch mẫu có các buổi học liền nhau gần như cả ngày; tiết kiệm đèn lớn vì baseline giả định đèn 100 % cả ngày. Hãy đo baseline thật của phòng trước khi so sánh.
- Bước mô phỏng > 5 s (ngoài dải vision 1–5 s của spec) làm mỗi lần nhận nhầm của detector giả bị "giữ" quá lâu nên đèn nhấp nháy giả; báo cáo sẽ in cảnh báo.
- Cùng seed cho kết quả giống hệt nhau (có test).

---

## 10. Chỗ khác/diễn giải so với spec — cần bạn xác nhận

| # | Spec | Thực hiện | Lý do |
|---|---|---|---|
| 1 | `PWM(t) = PWM(t−1) + Kp·e/ΔL_max` | `PWM = prev + 100·Kp·e/ΔL_max` (PWM tính theo %) | Spec trộn đơn vị; cộng thẳng `Kp·e/ΔL_max` (≤ 1) vào PWM % thì đèn nhích < 1 %/nhịp, mất hàng chục phút mới sáng |
| 2 | Kp = 0,5 | Kp = 0,5, riêng **bước đầu khi vùng vừa có người** dùng 1,0 | Đèn lên nhanh khi người vào; sau đó 0,5 chống nhấp nháy |
| 3 | "Nếu N_zone > 0" bật đèn | Cần trung bình 30 s ≥ 0,35 | Một lần nhận nhầm không bật đèn cả vùng |
| 4 | Fail-safe: `N_total == 0` lúc +15 phút | Trung bình 10 phút < 0,25 | Chịu được nhận nhầm lẻ tẻ (nếu đòi đúng 0 thì một nhận nhầm giữ điều hòa chạy cả buổi) |
| 5 | Pre-cool "30 phút trước giờ học" | Nhìn trước `max(30, lead_max)` phút | `t_lead` có thể tới 45–60 phút; nhìn 30 phút thì lên kế hoạch quá muộn |
| 6 | Setpoint động | Không **tăng** setpoint trong khoảng −20…+15 phút quanh giờ vào lớp; không hạ quạt khi đang pre-cool | Tránh nới nhiệt đúng lúc người sắp vào/đang vào lớp (giờ giải lao làm đếm người = 0 tạm thời) |
| 7 | Học chủ động: trần 50 ảnh/ngày | **Bỏ trần theo yêu cầu mới của bạn**, thay bằng lọc PCA 2 giai đoạn (mục 8b); thêm cổng chất lượng + holdout theo thời gian | Giữ nguyên tắc v2 (xem mục 3) |
| 8 | **Công thức setpoint ↔ bảng tiện nghi** | Giữ nguyên công thức | Xem ngay dưới |

**Mâu thuẫn trong spec (không tự sửa):** công thức `T_setpoint` ở mục 6C cho ≈ 25 °C khi phòng đầy ~75 % (26,5 − 2,5·0,75 ≈ 24,6 → 25), nhưng bảng tiện nghi ở mục 5 ghi 23–24 °C cho > 70 % công suất. Chỉ chạm 23–24 °C khi cộng đủ phạt độ ẩm và vùng đông. Nếu muốn khớp bảng thì phải tăng `ARES_PEOPLE_GAIN` (ví dụ 3,5) hoặc hạ `ARES_T_BASE` — đây là quyết định về tiện nghi/điện năng của bạn, nên chưa đổi.

---

## 11. Việc cần làm trước khi chạy thật

1. Thay `zones.json` bằng đa giác khớp góc camera thật (dùng `python -m ares zones --snapshot`), đo lux bằng máy để chỉnh `k` từng vùng.
2. Thay `schedule.json` bằng lịch thật.
3. Viết firmware ESP32 theo hợp đồng mục 5, chạy broker, thử một chu trình đèn + IR bằng tay trước khi để ARES điều khiển.
4. Đặt `ARES_REPLAY_DIR`, kiểm tra `ARES_WEIGHTS` và `ARES_DEVICE` (`python -m ares check`).
5. Chạy thử vài ngày ở chế độ quan sát (xem dashboard, đối chiếu số đếm người với thực tế) trước khi tin vào điều khiển.
6. Điều hòa phân loại IR chỉ một chiều — đo `R_cool` thực tế vài tuần đầu và theo dõi sự kiện `FAILSAFE_*`.

## 12. Kiểm thử

```powershell
pytest tests\ares -q          # 244 test ARES (≈ 3 phút, gồm mô phỏng nhiều giờ)
pytest -q                     # cả test cũ (73)
```

Test ARES dùng thư mục tạm riêng, không đụng `.env` hay dữ liệu thật.

---

## 13. Cập nhật 2026-11-01 — workflow mới + chống bật/tắt liên tục

Sơ đồ workflow: Camera → điều tiết FPS (bình thường full FPS, khi train 1 FPS) → YOLO → bộ đệm → điều khiển đèn/điều hòa;
ảnh khó (conf 0,35–0,55) → lọc PCA lần 1 (trùng thì **xoá**) → người duyệt 100 % → lọc PCA lần 2 → train (hạ FPS) →
**hoán đổi model ngay**, không ngắt dịch vụ → tab Model có lịch sử + Rollback.

| Thay đổi | Ở đâu | Ghi chú |
|---|---|---|
| Điều tiết FPS theo trạng thái train | `ares/fps.py`, `runtime.py` | `ARES_VISION_FPS=15`, `ARES_VISION_FPS_TRAINING=1`, `ARES_HUB_FPS=5`, `ARES_HUB_FPS_TRAINING=1`. Camera thật chờ khung **mới** nên không bao giờ đếm lại cùng khung. `ARES_VISION_INTERVAL_S` giờ chỉ dùng cho mô phỏng |
| Train xong hoán đổi ngay | `retrain.py`, `vision_tracker.py` | `auto_promote=always` mặc định; cổng chất lượng chỉ cảnh báo; nạp model nền |
| PCA lần 1 xoá ảnh trùng ngay | `pca_filter.py` | `duplicate_ttl_days=0` mặc định |
| Chống bật/tắt liên tục | `control_engine.py` | Bảng dưới |
| Tab **Thiết bị** trên dashboard | `tuning.py`, `dashboard/` | Chỉnh mọi delay / vùng điều chỉnh / FPS khi đang chạy, có hiệu lực ở nhịp kế tiếp, lưu `<storage_root>/device_tuning.json` (đè lên `ares.env`), có "Khôi phục mặc định". API: `GET/POST /api/tuning`, `POST /api/tuning/reset` |
| Khung hình trực tiếp mượt hơn | `index.html`, `overlay_interval_s=0.2` | Tab Giám sát tải ảnh liên tục (~5 ảnh/s); nhãn FPS trên thanh tiêu đề, chuyển cam khi đang train |

### Delay và vùng điều chỉnh (mặc định)

| Thiết bị | Tham số | Mặc định | Tác dụng |
|---|---|---|---|
| Đèn | `occ_on_threshold` / `occ_off_threshold` | 0,35 / 0,15 | Hysteresis: số người dao động quanh ngưỡng không làm vùng bật/tắt liên tục |
| | `light_on_delay_s` | 10 s | Trễ bật (thực tế cộng tối đa 1 nhịp đèn 15 s). Một lần nhận nhầm không bật đèn |
| | `empty_dim_after_s` / `room_off_after_s` | 180 / 180 s | Trễ hạ về nền / trễ tắt cả phòng (tách riêng được) |
| | `lux_deadband` | ±30 lux | Vùng chết: đủ gần lux đích thì giữ PWM |
| | `pwm_max_step` | 15 %/nhịp | Đổi từ từ (bước đầu khi vừa có người vẫn lên thẳng) |
| | `pwm_occupied_min/max`, `idle_pwm_min/max` | 0–100 %, 10–15 % | Dải điều chỉnh |
| | `led_publish_min_delta` | 2 % | Đổi nhỏ hơn thì không gửi lệnh |
| Điều hòa | `deadband` | ±0,7 °C | Như spec |
| | `min_cmd_interval_s` | 300 s | Giãn cách 2 lệnh IR không khẩn cấp |
| | `ac_min_on_s` / `ac_min_off_s` | 600 / 300 s | Bảo vệ máy nén. Min-off áp dụng cả pre-cool (pre-cool chỉ lùi vài phút, không bị huỷ). Fail-safe "lớp trống sau 15 phút" vẫn tắt được dù chưa đủ min-on |
| | `ac_on_delay_s` / `ac_off_delay_s` | 120 / 300 s | Trễ bật theo hiện diện / trễ tắt khi trống (cộng sau cửa sổ 10 phút) |
| | `ac_keep_on_before_class_min` | 30 phút | Phòng trống nhưng sắp vào lớp: **không tắt**, hạ về nhiệt độ pre-cool + quạt mạnh (sự kiện `PRECOOL_KEEP_ON`) thay vì tắt rồi pre-cool lại sau vài phút |
| | `ac_setpoint_max_step` | 2 °C/lệnh | Đổi setpoint từ từ |
| | `t_min`/`t_max`, `fan_min`/`fan_max`, `fan_hysteresis` | 22–27 °C, 1–3, ±5 % | Dải điều chỉnh + vùng chết mức quạt |

### So sánh mô phỏng 4 ngày (cùng kịch bản, seed 42, bước 5 s) — số từ mô hình giả định, không phải đo thật

| | Trước | Sau |
|---|---|---|
| Đèn: số lần vùng đổi sáng ↔ tắt/nền | 240 | 96 |
| Đèn: số lần PWM đổi chiều tăng/giảm | 6 026 | 504 |
| Đèn: tổng quãng PWM thay đổi | 29 804 % | 7 911 % |
| Điều hòa: số lần bật/tắt | 8 | 4 (bỏ được ca "tắt giờ nghỉ trưa rồi 8 phút sau pre-cool lại") |
| Có người mà > 26,5 °C | 0 phút | 0 phút |
| Vùng có người mà < 300 lux | 5,5 phút-vùng | 8,75 phút-vùng (giá của trễ bật 10 s) |
| Điện đèn / điều hòa | 2,44 / 30,20 kWh | 2,30 / 30,26 kWh |

Test cũ cho công thức spec chạy với cấu hình "không lớp chống bật/tắt" (`tests/ares/conftest.py::NO_ANTICHATTER`);
các hành vi mới có test riêng ở `tests/ares/test_anti_chatter.py`.

Đánh đổi đã chấp nhận (người dùng chọn): bỏ cổng chặn model và xoá ngay ảnh trùng làm mất hai lưới an toàn —
model kém vẫn lên production (có cảnh báo, rollback tay) và ảnh bị PCA lọc nhầm không khôi phục được.
