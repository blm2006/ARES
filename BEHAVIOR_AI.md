# ARES · Nhận diện hành vi bất thường (behavior_ai)

> Module `ares/behavior/`. Phát hiện **té ngã**, **ngã gục / nằm bất động**, **ngồi gục bất động**, **đánh nhau / xô đẩy**
> trên mọi camera ARES đang có. Cảnh báo hiện ở trang **/behavior** của dashboard ARES, kèm **clip 5 giây trước + 10 giây sau**.
> Ngày viết: 14/11/2026.

## 1. Mục tiêu và phạm vi

| Hạng mục | Đã chốt |
|---|---|
| Gắn vào | ARES (`python -m ares run`), chạy **độc lập** với điều khiển đèn/điều hòa: không đọc, không ghi gì của `control_engine` |
| Camera | Theo danh sách camera ARES đang có (camera điều khiển `main` + camera phụ trong hub); bật/tắt từng camera bằng `ARES_BEHAVIOR_CAMERAS` |
| Hành vi | Té ngã · Nằm xuống đột ngột / nằm dưới sàn (2–5 s, kể cả còn cử động) · Nằm bất động ≥ 15 s · Ngồi gục bất động ≥ 15 s · Đánh nhau/xô đẩy |
| Cảnh báo | Chỉ trên dashboard (khung đỏ, danh sách cảnh báo, nhật ký sự kiện ARES). Telegram/Zalo/âm thanh: làm sau |
| Bằng chứng | Clip 5 s trước + 10 s sau, ảnh chụp lúc phát hiện, chuỗi khung xương |
| Model | Weights pretrained + thư viện pip; **code tích hợp và code train tự viết** |
| Train thêm | Script riêng `python -m ares.behavior train`, người dùng tự chạy |
| Dữ liệu | Dataset công khai (script tự tải) + video tự quay + sự kiện đã duyệt trên dashboard |

**Giới hạn quan trọng:** camera **không chẩn đoán đột quỵ**. Hệ thống chỉ phát hiện *dấu hiệu nhìn thấy được* (ngã, nằm/gục
bất động lâu). Người nằm nghỉ trên giường/ghế dài hay ngủ gục trên bàn cũng có thể bị báo → đây là thiết kế
"thà báo nhầm còn hơn bỏ sót"; người trực bấm **Báo nhầm** để vừa tắt cảnh báo vừa tạo dữ liệu train.

## 2. Chọn model (tìm trên GitHub)

| Lựa chọn | Có sẵn | Tự train thêm | Kết luận |
|---|---|---|---|
| **Ultralytics YOLO11-pose** (`yolo11n-pose.pt`, COCO-17 điểm khớp) — github.com/ultralytics/ultralytics | pip `ultralytics` (đã cài), weights đã có trong `C:\yolo-al` | Có (`yolo pose train`) | **Dùng** để lấy khung xương |
| ST-GCN (Yan et al. 2018) — github.com/yysijie/st-gcn | Code cũ, PyTorch 0.4 | Có | Lấy **ý tưởng** → tự viết bản nhỏ `STGCNLite` |
| MMAction2 / PYSKL (ST-GCN++, PoseC3D pretrained NTU-60, có lớp "falling", "punching") | pip `mmaction2` + `mmcv` | Có | **Không dùng lúc này**: `mmcv` rất khó cài trên Windows + Python 3.12; có thể thay vào sau |
| VideoMAE / X3D (video thô) | HuggingFace | Có | Nặng GPU, nhạy nền/ánh sáng; khung xương bền hơn với camera cố định |

→ Kiến trúc chọn: **YOLO11-pose (pretrained) → luật tự viết + ST-GCN-lite tự viết, train trên khung xương.**
Luật chạy được ngay không cần train; model càng nhiều dữ liệu càng tốt và *bổ sung* cho luật (chế độ `hybrid`).

## 3. Luồng xử lý

```
VisionTracker (main) ─┐ offer(cam, ảnh, t)   chỉ giữ ảnh MỚI NHẤT/camera, không copy, không chặn
CameraHub (phụ)  ─────┘
        │
        ▼  luồng "ares-behavior", tối đa ARES_BEHAVIOR_FPS vòng/giây (mặc định 10)
  gom ảnh mới của mọi camera thành 1 lô → YOLO11-pose (1 model, chung GPU)
        │
        ▼  mỗi camera
  Tracker tự viết (IoU + khoảng cách tâm; giữ người đang nằm tới 15 s khi model bỏ sót)
        │
        ▼  mỗi người: đặc trưng tư thế + chuyển động  (skeleton.py)
  Luật (analyzer.py)  +  STGCNLite mỗi 0,5 s nếu đã có model (model.py)
        │
        ▼  sự kiện
  EventStore (sqlite) + thư mục bằng chứng + ClipRecorder (luồng ghi riêng)
  emit → nhật ký sự kiện ARES  ·  trang /behavior  ·  badge "Hành vi" trên dashboard chính
```

Không có luồng nào chờ luồng khác: YOLO-pose chậm thì vòng hành vi tự bỏ khung cũ, đèn/điều hòa không bị ảnh hưởng.
Lúc ARES đang retrain YOLO ban đêm, FPS vision bị hạ xuống 1 → nhận diện hành vi cũng thưa đi (đã biết, chấp nhận).

## 4. Đặc trưng và luật

Mọi khoảng cách/tốc độ chia cho **chiều cao tham chiếu** của từng người (phân vị 80 cạnh dài khung người gần đây) nên
không phụ thuộc người đứng gần hay xa camera.

| Đặc trưng | Cách tính |
|---|---|
| Góc thân | vector hông → vai so với phương thẳng đứng (0° đứng, 90° nằm; vai thấp hơn hông → ≥ 75°) |
| Tư thế | `lying` (góc ≥ 60°, hoặc khung rộng/cao ≥ 1,4 khi không thấy thân) · `slumped` (đầu ngang/dưới vai + đang ngồi/cúi 20–60°) · `sitting` · `upright` |
| Chuyển động | độ dịch của **2 khớp cử động nhiều nhất** giữa trung bình 0,3 s gần nhất và 0,3 s cách đó ~0,85 s (khử rung của model). Trước đây lấy trung bình cả 17 khớp nên chỉ cử động tay/đầu bị pha loãng và tính nhầm "bất động". Đã thử: rung ±1,6% chiều cao/khung vẫn là "bất động"; cử động tay ≥ 5% chiều cao là "cử động" |
| Tốc độ tay/chân | trung vị 3 khung của tốc độ lớn nhất trong 4 điểm cổ tay/cổ chân |

| Sự kiện | Điều kiện (mặc định) |
|---|---|
| **Té ngã** | đang đứng/ngồi → nằm trong ≤ 1,5 s, hông tụt ≥ 0,35 × chiều cao, rồi ≥ 60% thời gian 1 s sau vẫn nằm |
| **Tư thế hiển thị** (không phải cảnh báo) | Đứng · Đang đi (tâm khung người dịch > 0,3 chiều cao/giây, có trễ 0,18) · Ngồi · Nằm · Ngồi gục · Thấy nửa người (không thấy hông + gối thì không đoán đứng/ngồi). Ngồi = đùi gần nằm ngang; nhận cả người **ngồi sẵn từ đầu** (đùi ngắn hơn nửa thân, không phụ thuộc chiều cao tham chiếu) |
| **Nằm xuống đột ngột / nằm dưới sàn** | nằm liên tục **5 s** → báo **dù vẫn cử động**; nếu ≤ 3 s trước đó còn đứng/ngồi (nằm xuống đột ngột, không đủ tiêu chí té ngã) → chỉ cần **2 s**. Đã báo té ngã cho người đó thì không báo trùng. Ghi rõ "vẫn còn cử động" / "ít cử động" |
| **Nằm bất động** | tư thế nằm + chuyển động < 0,05 liên tục **15 s** (chỉ đếm khi đo được khớp); cử động > 1,5 × ngưỡng là đếm lại từ đầu. Sau cú ngã → ghi "sau cú ngã", mức khẩn cấp |
| **Ngồi gục bất động** | tư thế `slumped` + bất động 15 s |
| **Đánh nhau / xô đẩy** | 2 người cách nhau < 1,5 × chiều cao **và** tay/chân vung > 1,5 chiều cao/giây, tích luỹ ≥ 0,8 s |
| Chống lặp | cùng người (hoặc cùng chỗ) + cùng loại: không báo lại trong 60 s |

Khi có model (`hybrid`): `p(lying)`/`p(slumped)` ≥ 0,7 ghi đè tư thế; `p(fall)` ≥ 0,7 hai lần liên tiếp = bắt đầu cú ngã;
`p(fight)` ≥ 0,7 khi hai người sát nhau = cộng dồn thời gian đánh nhau. Chế độ `model` chỉ dùng tín hiệu model
(chưa có model thì tạm dùng luật).

### Số đo thật (đã chạy trên máy dev, CPU)

* **Luật đánh nhau** trên 240 clip Surveillance Camera Fight (2 giây/clip): bắt **~32%** clip đánh nhau, báo nhầm **~5%** clip thường.
  Luật chỉ là phương án dự phòng → **nên train model**.
* **STGCNLite** train 40 epoch trên 300 clip đó (chia theo clip 80/20): cửa sổ 2 s đạt **F1 0,73** (fight: precision 0,79, recall 0,68).
  Dữ liệu ít nên còn overfit; thêm URFD + video tự quay + sự kiện đã duyệt sẽ tốt hơn.
* Luật té ngã/bất động/ngồi gục được kiểm bằng khung xương tổng hợp (`tests/ares/test_behavior.py`); **chưa đo trên video ngã thật**
  vì máy dev không tải được URFD — việc đầu tiên trên máy bạn là chạy `video` trên vài clip URFD (mục 7).

## 5. Model STGCNLite

* Đầu vào `(C=5, T=16, V=17, M=2)`: x, y, độ tin cậy, vx, vy · 16 bước lấy đều trong 2 s gần nhất (lấy mẫu **theo thời gian**,
  FPS camera khác nhau vẫn dùng chung model) · người chính + người gần nhất.
* Chuẩn hoá: gốc = trung điểm hông người chính ở bước đầu (giữ được chuyển động tụt xuống khi ngã), chia chiều cao tham chiếu.
* 5 khối GCN (không gian, đồ thị 18 cạnh COCO + trọng số cạnh học được) + TCN kernel 5; ghép đặc trưng 2 người → 5 lớp:
  `normal, fall, lying, slumped, fight`. ~0,3 triệu tham số.
* Train: AdamW + OneCycle, label smoothing 0,05, trọng số lớp √(nghịch tần suất); tăng cường lật trái/phải, co giãn,
  xoay ±12°, che khớp 6%, nhiễu. Giữ epoch có macro-F1 val cao nhất. Ghi `models/behavior-<giờ>.pt/.txt/.json`, chép thành `current.pt`.

## 6. Dữ liệu

| Nguồn | Lấy | Nhãn |
|---|---|---|
| UR Fall Detection (30 ngã + 40 sinh hoạt, cam0 song song sàn, 30 fps) — fenix.ur.edu.pl/~mkepski/ds/uf.html | `download` tự tải ~3 GB | theo từng khung từ csv: −1 bình thường · 0 đang ngã · 1 nằm |
| Surveillance Camera Fight (150 + 150 clip, MIT) — github.com/sayibet/fight-detection-surv-dataset | **đã chép sẵn** vào `var/ares/behavior/datasets/raw/fight-surv/` | cả clip: fight / normal |
| Video tự quay | chép vào `datasets/raw/custom/<lớp>/` | cả clip 1 nhãn (nên cắt 2–10 s đúng hành vi) |
| Sự kiện trên dashboard | bấm **Đúng** / **Báo nhầm** | Đúng → loại sự kiện; Báo nhầm → normal |
| Le2i (đăng ký tay) | tải tay, cắt đoạn vào `custom/fall` / `custom/normal` | — |

Chưa có dataset công khai cho **ngồi gục** → luật lo phần này; muốn model học, nhóm quay vài chục clip `custom/slumped`.

## 7. Lệnh

```bat
:: 1) Thử ngay trên video (không cần chạy ARES) — xuất video có khung xương + danh sách sự kiện
.venv\Scripts\python -m ares.behavior video samples\ten_video.mp4
.venv\Scripts\python -m ares.behavior video var\ares\behavior\datasets\raw\fight-surv\fight --no-write
.venv\Scripts\python -m ares.behavior video clip.mp4 --still-s 10 --debug      :: in tư thế/chuyển động mỗi giây để chỉnh ngưỡng

:: 2) Dữ liệu + train
.venv\Scripts\python -m ares.behavior download            :: URFD (fight đã có sẵn)
.venv\Scripts\python -m ares.behavior extract             :: YOLO-pose -> keypoints\*.npz (GPU ~ vài phút)
.venv\Scripts\python -m ares.behavior train               :: 60 epoch, tự chép thành current.pt

:: 3) Chạy ARES như cũ — module tự bật
.venv\Scripts\python -m ares run      :: rồi mở http://127.0.0.1:8100/behavior (hoặc tab "Hành vi")
```

Kết quả thử video: `var\ares\behavior\video_tests\`. Model: `var\ares\behavior\models\`. Sự kiện: `var\ares\behavior\events\`.

## 8. Dashboard `/behavior`

* **Cảnh báo chưa xử lý**: ảnh + mô tả + nút *Xem clip*, *Đã xử lý*, *Đúng*, *Báo nhầm*; tiêu đề tab nhấp nháy khi có cảnh báo.
* **Camera đang phân tích**: ảnh có khung xương (xanh bình thường · cam đang theo dõi, đếm giây bất động · đỏ đã báo động).
* **Lịch sử**: lọc theo loại; trình phát clip bằng chuỗi ảnh (chạy mọi trình duyệt, không phụ thuộc codec) + tải `clip.mp4`.
* Dashboard chính có thêm tab **Hành vi** với badge số cảnh báo mới. Dùng chung token/chống CSRF của ARES.

API: `GET /api/behavior/status|events|events/{id}|events/{id}/snapshot.jpg|events/{id}/frames/{n}.jpg|events/{id}/clip.mp4|cameras/{cam}/frame.jpg`,
`POST /api/behavior/events/{id}/ack|verdict`, `POST /api/behavior/ack-all`.

## 9. Cấu hình (`.env` hoặc `ares_config/ares.env`, tiền tố `ARES_BEHAVIOR_`)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ENABLED` | true | bật/tắt cả module |
| `CAMERAS` | all | `all` hoặc `main,usb1,...` |
| `WEIGHTS` | yolo11n-pose.pt | máy mạnh: `yolo11s-pose.pt` (chính xác hơn, chậm hơn) |
| `FPS` | 10 | vòng phân tích/giây (mỗi vòng xử lý mọi camera theo lô) |
| `MODE` | hybrid | rules · model · hybrid |
| `MODEL_THR` | 0.7 | xác suất tối thiểu để model tự báo |
| `STILL_S` | 15 | giây bất động trước khi báo "nằm/ngồi gục bất động" |
| `STILL_MOTION` | 0.05 | ngưỡng bất động (chiều cao/giây, đo trên 2 khớp cử động nhiều nhất) |
| `LYING_S` | 5 | nằm liên tục chừng này giây thì báo "nằm dưới sàn" dù còn cử động (0 = tắt — vd camera nhìn giường) |
| `SUDDEN_LYING_S` | 2 | vừa đứng/ngồi rồi nằm xuống đột ngột: báo sau chừng này giây |
| `FALL_WINDOW_S` / `FALL_DROP` / `FALL_CONFIRM_S` | 1.5 / 0.35 / 1.0 | luật té ngã |
| `FIGHT_DIST` / `FIGHT_SPEED` / `FIGHT_S` | 1.5 / 1.5 / 0.8 | luật đánh nhau |
| `COOLDOWN_S` | 60 | chống báo lặp |
| `CLIP_PRE_S` / `CLIP_POST_S` / `CLIP_FPS` / `CLIP_WIDTH` | 5 / 10 / 8 / 640 | clip bằng chứng |
| `KEEP_DAYS` | 30 | xoá sự kiện **chưa duyệt** cũ hơn N ngày (đã duyệt thì giữ làm dữ liệu) |

## 10. File

Mới: `ares/behavior/` (`skeleton, tracker, analyzer, model, pose, store, monitor, routes, datasets, train, run_video, __main__`),
`ares/dashboard/static/behavior.html`, `tests/ares/test_behavior.py`, file này.
Sửa nhỏ (bản gốc ở `Claude outputs\backup_truoc_behavior_2026-11-14\`): `ares/config.py` (thêm mục cấu hình),
`ares/runtime.py` (tạo/khởi động/dừng BehaviorMonitor, tên camera, badge), `ares/vision_tracker.py` + `ares/cameras.py`
(mỗi khung gọi `behavior.offer`, 3 dòng), `ares/dashboard/app.py` (đăng ký route), `ares/dashboard/static/index.html` (tab Hành vi),
`ares_config/ares.env.example`. Toàn bộ 252 test ARES chạy qua.

## 11. Việc tiếp theo

1. Chạy `download` → `extract` → `train` trên GPU; xem báo cáo `.txt`, nhất là recall lớp `fall`.
2. Thử `video` trên vài clip URFD và video quay tại phòng; chỉnh `STILL_MOTION`, `FALL_DROP` theo góc camera thật.
3. Quay clip `custom/slumped` (ngồi gục) và `custom/normal` (cúi nhặt đồ, ngồi xuống nhanh, nằm nghỉ có cử động) để giảm báo nhầm.
4. Sau đó: Telegram bot (nhắn kèm ảnh), âm thanh trên dashboard, có thể thay ST-GCN-lite bằng model pretrained NTU qua MMAction2.


## AI bổ sung: cháy / khói, hút thuốc (18/11/2026)

Mã: `ares/hazards/`. Cài cùng `CAI_AI_BAO_HO.bat` (`python -m ares.hazards setup`, có kiểm SHA-256). Chạy 1 lần/giây/camera
(`ARES_HAZARD_FPS`), cả camera trực tiếp và Thử video. Tắt: `ARES_HAZARDS=off`; chỉ 1 mục: `ARES_HAZARDS=fire`.

| Mục | Model | Khi nào báo | Giấy phép |
|---|---|---|---|
| Cháy / khói (`FIRE`, khẩn cấp) | huggingface.co/rabahdev/fire-smoke-yolov8n — YOLOv8n trên D-Fire | lửa thấy ≥ 3/5 lần kiểm gần nhất; khói (dễ nhầm bụi gỗ) ≥ 4/5. Cùng camera không báo lại trong 120 giây | AGPL-3.0 |
| Hút thuốc (`SMOKING`) | huggingface.co/Beehzod/smoking-detection-finetuned — YOLO11m, lớp cigarette, ảnh 960 px | điếu thuốc cách mũi / cổ tay của người < 0,45 × chiều cao người, ≥ 3/6 lần kiểm | MIT |

Đã kiểm: video xưởng gỗ 102 giây không báo nhầm cháy/khói hay hút thuốc (model cháy chỉ có 1 khung lẻ "khói", bị bộ lọc 4/5 loại);
ảnh cháy công khai: nhận lửa + khói; ảnh người hút thuốc: thấy điếu thuốc ở 8/16 ảnh -> cần **train dần** tại chỗ.

Đã tìm nhưng KHÔNG thêm: model đánh nhau dạng ảnh đơn (file cần thư viện `dill` để mở, có thể chạy mã lạ khi nạp; ARES đã có
phát hiện đánh nhau bằng khung xương + ST-GCN); model cháy "home fire" (báo khói nhầm ở mọi khung video xưởng gỗ); dao/súng
(model COCO nhận nhầm nhiều, chưa có model mở đủ tin cậy).

**Train dần.** Mỗi cảnh báo cháy/khói, hút thuốc, đồ bảo hộ lưu thêm `raw.jpg` (ảnh gốc) + `dets.json` (hộp model thấy).
Dashboard → Hành vi → Train lại AI → "Cháy / khói + hút thuốc": Đúng = ảnh có vật (hộp của model), Báo nhầm = ảnh nhãn rỗng;
fine-tune từ model đang dùng (đóng băng backbone, lr nhỏ), rồi **cổng kiểm định**: chỉ dùng model mới nếu số ảnh Báo nhầm còn bị báo
giảm mà không sót thêm ảnh Đúng. Model dùng nằm ở `behavior/hazards/<mục>/current.pt` (bản trước: `prev_<giờ>.pt`), nạp ngay.
Cần ≥ 2 ảnh Đúng và ≥ 2 ảnh Báo nhầm; thêm ảnh tự gán nhãn ở `behavior/hazards/<mục>/extra/images|labels`.
Đồ bảo hộ Báo nhầm = model SÓT món đồ, không tự biết hộp ở đâu -> ảnh gom vào `behavior/hazards/ppe/can_gan_nhan/` để gán nhãn.
