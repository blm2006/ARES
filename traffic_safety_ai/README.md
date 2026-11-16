# Traffic Safety AI — giám sát an toàn giao thông khu vực trường học (chạy local)

Hệ thống AI chạy trên máy cá nhân/máy chủ nội bộ, phân tích video **camera cố định (CCTV)** và **camera hành trình (POV)**
quanh cổng trường để ghi nhận **16 tình huống** vi phạm / nguy cơ, đọc **biển số xe Việt Nam**, lập **phiếu ghi nhận**
theo các trường của trang tra cứu "phạt nguội", thống kê **lưu lượng – tốc độ – điểm nóng** và có **trang quản trị**
để người có thẩm quyền xác minh.

> **AI không có thẩm quyền xử phạt.** Mọi hồ sơ ở trạng thái *chờ xác minh*; căn cứ và mức phạt in trên phiếu chỉ để
> **tham khảo** (cần đối chiếu văn bản gốc). Phiếu không mang quốc huy, tên cơ quan nhà nước hay hình thức quyết định.

Gói **độc lập** với yolo-al/ARES (không import chéo) — chép thư mục `traffic_safety_ai/` sang chỗ khác vẫn chạy.

---

## 0. Cách dễ nhất: Studio (mở như app — thả video, tự vẽ, bấm chạy)

Bấm đúp **`MO_APP.bat`** (hoặc `python -m traffic_safety_ai studio`, hoặc mục **S** trong `TRAFFIC.bat`).
Một cửa sổ ứng dụng mở ra tại `http://127.0.0.1:8200/studio`:

1. **Kéo-thả video** vào cửa sổ (nhiều tệp cùng lúc được). Video được chép vào `data/studio/videos`.
   Tệp rất lớn: dùng ô "thêm theo đường dẫn" để không phải chép.
2. **Chọn thứ cần vẽ** ở cột phải rồi bấm lên hình: vỉa hè, vạch qua đường, làn + hướng đi, cấm dừng/đỗ,
   lòng đường, vạch dừng đèn đỏ + ô đèn, cặp vạch đo tốc độ, hiệu chuẩn mét... Thanh vàng phía trên hình
   chỉ từng bước. Kéo chấm trắng để chỉnh, `Delete` để xoá, `Ctrl+Z` hoàn tác, lăn chuột để phóng to.
   Danh sách "Tình huống AI sẽ kiểm tra" cập nhật ngay theo hình đã vẽ (thiếu gì thì ghi rõ thiếu gì).
3. **Lưu & chạy phân tích**: chạy ở nền, có thanh tiến độ và khung hình trực tiếp; xong thì có nút
   mở hồ sơ vi phạm, báo cáo và tải video chú thích.

Mỗi video có một file hình vẽ riêng `data/studio/scenes/<tên video>.yaml` (đúng định dạng `SiteConfig`,
dùng lại được với lệnh `run --scene`). Tên tệp được bỏ dấu tiếng Việt khi chép vào để OpenCV trên Windows
đọc chắc chắn. Giữ cửa sổ đen (máy chủ nội bộ) mở trong lúc dùng; đóng nó là tắt Studio.

**Trang "Xử lý vi phạm"** (`/vi-pham`, mở từ Studio hoặc trang tổng quan): mỗi vi phạm là một thẻ (lỗi, loại xe,
biển số, địa điểm, giờ); bấm vào thẻ để xem minh chứng bên phải — clip vi phạm, ảnh trước/trong/sau, ảnh biển số —
rồi xác nhận, loại bỏ hoặc sửa biển. Lọc theo ngày, loại lỗi, loại xe, camera; tìm theo biển số.

**Video chú thích gọn**: xe bình thường chỉ có khung mảnh, không chữ; chỉ xe vi phạm mới có nhãn (biển số hoặc loại xe)
và băng tên lỗi hiện ~5 giây. Cần soi lại nhận diện thì bật "Ghi tên mọi xe lên video" trong Studio hoặc `run --labels full`.

## 1. Chạy thử ngay (không cần camera, không cần model)

```
C:\yolo-al> TRAFFIC.bat            (chọn 2 — mô phỏng)
hoặc
C:\yolo-al> .venv\Scripts\python -m traffic_safety_ai demo --open
```

Bộ mô phỏng tự vẽ một cổng trường (CCTV, 46 giây) và một tuyến camera hành trình (POV, 30 giây) với 21 tình huống
dàn dựng, cho chạy qua **toàn bộ** pipeline (tracking, rule, hợp nhất biển số, bằng chứng, báo cáo). Kết quả mong đợi:
CCTV 15 hồ sơ (RB-01…RB-12, HZ-02, HZ-04, hai mức của RB-06), POV 6 hồ sơ (RB-03, RB-07, RB-08, HZ-01, HZ-02, HZ-03),
**0 báo giả** với các xe chấp hành đúng. *Mô phỏng chỉ kiểm tra logic, không phản ánh độ chính xác trên video thật.*

## 2. Cài đặt

Dùng lại venv của gói yolo-al (Python 3.12, torch 2.6 + CUDA 12.4, ultralytics 8.4):

```
cd C:\yolo-al
.venv\Scripts\python -m pip install -r traffic_safety_ai\requirements.txt
.venv\Scripts\python -m traffic_safety_ai check
```

`check` in phiên bản thư viện, GPU, model nạp được và **bảng 16 tình huống BẬT/TẮT kèm lý do** (thiếu model nào,
thiếu khai báo vùng nào). Thiếu model → tình huống đó tự tắt, hệ thống vẫn chạy phần còn lại.

## 3. Quy trình triển khai cho một camera

1. **Khai báo vùng/vạch** (cửa sổ OpenCV, chuột + phím):
   `python -m traffic_safety_ai calibrate --source cong_A.mp4 --scene traffic_safety_ai\configs\cong_A.yaml`
   — vỉa hè, vạch qua đường, làn + mũi tên hướng đi, vùng cấm dừng/đỗ, vùng tầm nhìn, lòng đường, vạch dừng (+ đèn),
   vạch đếm, cặp vạch đo tốc độ, 4 điểm hiệu chuẩn mặt đường (hình chữ nhật biết kích thước, vd ô vạch kẻ 3,5 × 10 m).
2. **Kiểm tra bằng mắt**: `python -m traffic_safety_ai preview --source cong_A.mp4 --scene ...\cong_A.yaml`
   → ảnh có toàn bộ vùng/vạch + **lưới 1 m** của hiệu chuẩn (lưới méo = hiệu chuẩn sai).
3. **Chỉnh tham số** trong YAML: giới hạn tốc độ **theo biển báo tại chỗ**, giờ áp dụng cấm dừng, ngưỡng từng rule.
4. **Chạy**: `python -m traffic_safety_ai run --source cong_A.mp4 --scene ...\cong_A.yaml --show --save-video`
   (thư mục video, `rtsp://...`, số webcam đều được; thêm `--dashboard` để xem trực tiếp trên trình duyệt).
   Giờ ghi trên phiếu lấy từ `--start-time "2026-11-04 06:45:00"`, hoặc tự đoán từ tên tệp (`..._20261104_064500.mp4`).
5. **Duyệt hồ sơ**: `python -m traffic_safety_ai dashboard` → http://127.0.0.1:8200 — xác nhận / loại bỏ / sửa biển số
   (có nhật ký ai sửa gì, lúc nào).
6. **Báo cáo**: `python -m traffic_safety_ai report --day 2026-11-04 --open` → HTML + 3 tệp CSV (mở bằng Excel).

`TRAFFIC.bat` (đặt ở `C:\yolo-al\` hoặc ngay trong `traffic_safety_ai\`) có menu cho tất cả các bước trên.

**Thử trên 2 video mẫu** trong `C:\yolo-al\samples\` (đã có cấu hình sẵn):

```
python -m traffic_safety_ai run --source "samples\vid xe chạy.mp4" --scene traffic_safety_ai\configs\mau_video_un_tac_hn.yaml --show --save-video
python -m traffic_safety_ai run --source samples\Download.mp4 --scene traffic_safety_ai\configs\mau_video_cao_toc_dem.yaml --show --save-video
```

Video ùn tắc quay tay → hệ thống phát hiện camera rung và tạm ngưng rule hướng/tốc độ (đúng thiết kế); video ban đêm chỉ
khai báo vạch đếm (khung có nhiều đường, hai chiều xe). Đây là video mạng xã hội để thử nhận diện/đếm, không phải CCTV thật.

## 4. Mười sáu tình huống

| Mã | Tình huống | Kênh | Cần thêm | Phân loại |
|---|---|---|---|---|
| RB-01 | Không đội mũ bảo hiểm (người lái / người ngồi sau; trừ trẻ < 6 tuổi) | CCTV | `models/helmet.pt` | Vi phạm |
| RB-02 | Dùng điện thoại khi lái | CCTV, POV | model pose (tự tải) | Nghi vấn |
| RB-03 | Đi ngược chiều | CCTV, POV | làn + mũi tên (CCTV) | Vi phạm / nghi vấn |
| RB-04 | Đi trên vỉa hè / sai làn / đè vạch liền | CCTV (POV cần `lane_seg.pt`) | vùng vỉa hè, làn | Vi phạm |
| RB-05 | Vượt đèn đỏ (2 vạch: vạch dừng + vạch xác nhận) | CCTV, POV | vạch dừng + ô đèn | Vi phạm |
| RB-06 | Chở quá số người (3 người = nghi vấn vì ngoại lệ trẻ < 12 tuổi; ≥ 4 = vi phạm) | CCTV | — | Vi phạm / nghi vấn |
| RB-07 | Không nhường người đi bộ (TTC + PET) | CCTV, POV | vạch qua đường | Nghi vấn / nguy cơ |
| RB-08 | Phanh gấp / bám đuôi (chỉ số nguy cơ) | CCTV, POV | hiệu chuẩn mét (CCTV) | Nguy cơ |
| RB-09 | Chở hàng cồng kềnh | CCTV | tách nền hoặc `cargo.pt` | Nghi vấn |
| RB-10 | Quá tốc độ (≥ 5 km/h + sai số so với biển báo) | CCTV | hiệu chuẩn hoặc cặp vạch đo | Vi phạm (ước lượng) |
| RB-11 | Dừng/đỗ sai: trước cổng, trên vạch, vỉa hè (*bổ sung*) | CCTV | vùng cấm dừng/đỗ | Vi phạm |
| RB-12 | Đi dàn hàng ngang ≥ 3 xe (*bổ sung*) | CCTV | — | Vi phạm |
| HZ-01 | Rẽ/chuyển làn không xi-nhan | POV | — | Nghi vấn |
| HZ-02 | Khuất tầm nhìn (xe đỗ che lối ra cổng; xe lớn bên đường) | CCTV, POV | vùng tầm nhìn (CCTV) | Nguy cơ |
| HZ-03 | Người/xe lao ra bất ngờ từ sau vật che | POV, CCTV | lòng đường (CCTV) | Nguy cơ |
| HZ-04 | Học sinh qua đường ngoài vạch → dữ liệu đề xuất vạch/đèn (*bổ sung*) | CCTV | lòng đường | Nguy cơ |

Căn cứ tham khảo từng tình huống nằm ở `configs/rules_catalog.yaml` (NĐ 168/2024/NĐ-CP đã sửa đổi bởi NĐ 238/2026/NĐ-CP,
Luật TTATGTĐB 2024). Khi văn bản thay đổi chỉ cần sửa tệp này, không sửa code.

## 5. Đầu ra

**Mỗi hồ sơ** (`data/evidence/<ngày>/<mã hồ sơ>/`):

- `composite.jpg` — **phiếu ghi nhận** 1920×1250: toàn cảnh có khoanh vùng, ảnh trước/sau ~1 giây, ảnh phương tiện,
  ảnh biển số + biển đọc được, và các trường như tra cứu phạt nguội: *Biển kiểm soát · Màu biển · Loại phương tiện ·
  Thời gian vi phạm · Địa điểm vi phạm · Hành vi vi phạm · Trạng thái · Đơn vị phát hiện · Nơi giải quyết*, kèm căn cứ,
  mức phạt tham khảo, số đo (tốc độ, thời điểm đèn đỏ đã bật, TTC...), SHA-256 ảnh gốc;
- `composite_public.jpg` — như trên nhưng **làm mờ** mặt người và biển số của người không liên quan;
- `record.html` (in được, ảnh nhúng sẵn), `record.json`, `snapshot_raw.jpg`, `before.jpg`, `after.jpg`, `vehicle.jpg`,
  `plate.jpg`, `clip.webm/mp4` (có ~2 giây trước sự kiện), mã **SHA-256** của từng tệp (phát hiện chỉnh sửa).

**Mã hồ sơ**: `VP<ngày>-<camera>-<số thứ tự>`. **CSDL** `data/traffic.db` (SQLite). **Nhật ký lần chạy** `data/runs/<id>/`
(events.csv, video có chú thích). **Báo cáo** `data/reports/...`: thống kê theo tình huống/giờ, **lưu lượng 15 phút**
quy đổi xe con (TCVN 4054:2005), **giờ cao điểm + PHF**, **tốc độ V15/V50/V85**, **bản đồ điểm nóng** nguy cơ, danh sách hồ sơ.

## 6. Đọc biển số

Dò biển trên ảnh cắt xe (YOLO `plate.pt`, hoặc bộ dò cổ điển) → đọc ký tự → **hợp nhất nhiều khung**: mỗi lần đọc được
chấm theo độ tin OCR × độ nét × đúng mẫu biển VN, bỏ phiếu cả chuỗi và từng vị trí ký tự → một lần đọc sai 1–2 ký tự
không làm sai kết quả (trong mô phỏng: 16/16 biển đúng dù ~50% lần đọc có lỗi).
Kiểm tra theo mẫu biển TT 79/2024/TT-BCA (seri 20 chữ cái, mã tỉnh 34 tỉnh/thành sau sắp xếp 2025 — biển cũ vẫn
hợp lệ), sửa nhầm lẫn theo vị trí (0↔D, 8↔B, 5↔S...), nhận màu biển (trắng/vàng/xanh/đỏ).

Cách đọc ký tự (`configs/models.yaml → plate_ocr.backend`): `yolo_chars` (tự huấn luyện `plate_chars.pt`, hợp biển 2 dòng),
`fast_plate_ocr` (cài thêm), `easyocr`, hoặc `none` (chỉ lưu ảnh biển để người duyệt đọc và nhập).

## 7. Những điểm đã thay đổi so với đặc tả ban đầu

| Đặc tả | Đã làm | Lý do |
|---|---|---|
| Debounce theo frame (+1/−0,5) | Theo **thời gian** (+Δt / −0,5Δt, ngưỡng giây) | Không phụ thuộc FPS, `--stride` |
| Vận tốc = hiệu 2 điểm + EMA | **Hồi quy có loại ngoại lai** trên 1 s, kèm sai số | Hộp rung 2 px ở 30 fps = 60 px/s ảo |
| Ngưỡng tốc độ cố định | Theo **biển báo tại chỗ** + biên sai số; ghi "ước lượng" | Camera thường không phải thiết bị đo đã kiểm định |
| Vượt đèn đỏ 1 vạch | **2 vạch** + thời điểm cắt vạch nội suy + đèn đỏ đã bật bao lâu | Xe dừng đè vạch không bị báo |
| Vỉa hè: chạm là báo | Phải **đi dọc** vỉa hè ≥ n mét; lối vào cổng/nhà được miễn | Luật cho phép đi qua vỉa hè để vào nhà, cơ quan |
| Chở 3 người = vi phạm | 3 người = **nghi vấn** (ngoại lệ trẻ < 12 tuổi), ≥ 4 = vi phạm | Khoản 1 Điều 33 Luật TTATGTĐB 2024 |
| Mũ bảo hiểm: mọi người | Trừ **trẻ < 6 tuổi** (ước lượng chiều cao, ghi nghi vấn) | Ngoại lệ trong nghị định |
| POV: trừ ego-motion bằng dịch ảnh | **Hình học mặt đường** (X, Z mét) + tốc độ nở của nền + yaw | "Nở" khi tiến tới không bị hiểu nhầm là rẽ |
| Rule bật cứng | Bật/tắt theo **năng lực** (model, vùng khai báo) + lý do | Thiếu model không làm sai kết quả |
| — | Phát hiện **camera rung/xoay** → tạm ngưng rule hướng/tốc độ; bộ đếm bù trôi khung | Thử trên video quay tay: 38 báo giả → 0 |
| — | **Đồng thuận** RB-03: nhiều xe cùng "ngược chiều" = camera/khai báo sai | Chống báo giả hàng loạt |
| — | RB-11, RB-12, HZ-04, đếm lưu lượng, V85, điểm nóng, dashboard, mô phỏng | Phục vụ an toàn cổng trường và quy hoạch |
| — | Không lưu biển số xe **không vi phạm**; ảnh công khai làm mờ người khác | Luật Bảo vệ dữ liệu cá nhân 2025 |

## 8. Quyền riêng tư & pháp lý

- Dữ liệu chỉ nằm trên máy; dashboard mặc định chỉ nghe `127.0.0.1`. Mở ra mạng LAN (`--host 0.0.0.0`) thì **bắt buộc mã
  truy cập** (tự sinh nếu không khai báo); thao tác ghi cần header chống CSRF; nhật ký thao tác lưu trong CSDL.
- Biển số của xe không vi phạm bị xoá khi xe rời khung (`privacy.store_plates_of_non_violators: false`).
- `privacy.retention_days` + lệnh `cleanup --yes` để xoá hồ sơ quá hạn; `zones.privacy_mask` để luôn che cửa sổ nhà dân.
- Đặt camera và xử lý hình ảnh cần tuân thủ Luật Bảo vệ dữ liệu cá nhân (91/2025/QH15) và thông báo cho phụ huynh,
  học sinh, người dân khu vực. Việc cung cấp dữ liệu cho cơ quan chức năng theo NĐ 135/2021/NĐ-CP.

## 9. Hiệu năng

- Mặc định `yolo11s.pt` ở `imgsz 1280` (vật nhỏ từ góc cao). Máy yếu: `imgsz 960`, `--stride 2`.
- Tăng tốc: `python -m traffic_safety_ai export-models --which base --format engine` (TensorRT FP16) rồi sửa
  `models.yaml → base.weights` sang tệp `.engine`; hoặc `--format onnx`.
- Ba luồng: đọc video (không chặn), xử lý, ghi bằng chứng/video (nền).
- **Máy không có PyTorch / máy yếu / mini PC**: `--models traffic_safety_ai\configs\models_cpu_onnx.yaml` — model nền
  dạng ONNX (`models/yolov8n_960.onnx`) chạy bằng **OpenCV DNN**, chỉ cần `opencv-python` + `numpy`. Đo trên máy của
  bạn (máy ảo Linux 2 nhân của Claude desktop): ~6 khung/s ở 960 với `--stride 3`. Tạo ONNX từ model khác:
  `export-models --which base --format onnx --imgsz 960`.

### Tối ưu tháng 11/2026 (sau khảo sát các dự án GitHub cùng loại)

Ý tưởng tham khảo, viết lại theo cấu trúc của gói (không chép mã): roboflow/supervision (LineZone), obss/SAHI (chia ô),
ultralytics solutions (đếm/đo tốc độ theo vùng), BoT-SORT (agnostic NMS + bù chuyển động camera).

| Thay đổi | Ở đâu | Hiệu quả đo được |
|---|---|---|
| **Chỉ chạy model nền trong vùng đã khai báo** (`base.roi: auto`, mặc định; lề 15% mỗi phía) | `core/regions.py` | Video nút giao 640×360, CPU: **6,2 → 9,0 khung/s (+45%)**, sự kiện giữ nguyên |
| **Giới hạn model ảnh cắt ≤ 6 lần/giây mỗi đối tượng** (`max_hz`) cho pose/điện thoại, mũ | `core/attributes.py` | Pose trước chạy 15 lần/s/người lái ở video 30 fps; luật chỉ cần ≥ 3 lần trong 1,5 s |
| **Hồi quy 2 cột một lần**; CCTV bỏ tính tốc độ nở hộp (chỉ POV dùng) | `core/kinematics.py` | Bớt khoảng một nửa phép khớp mỗi khung |
| **Gộp hộp trùng khác lớp** (bán tải = ô tô + xe tải) | `core/detector.py` | Hết lỗi một xe 2 hồ sơ vượt đèn đỏ |
| **Đếm có trễ**: cắt vạch → chờ xác nhận, sang hẳn bên kia mới tính; quay lại thì huỷ; mất dấu khi đang qua vẫn tính | `analytics/counting.py` | Xe máy đứng đè vạch đếm lúc ùn tắc không còn bị đếm/đổi chiều sai |
| **Chia ô kiểu SAHI** (`base.tiles: [2, 2]`, mặc định tắt) | `core/regions.py` | Video đường đôi: xe nhận được ở xa **3,8 → 6,0/khung (+58%)**, nhưng chậm ~4,5 lần → chỉ nên bật khi có GPU |

Ví dụ bật chia ô trên máy có GPU (`configs/models.yaml`):

```yaml
base:
  weights: yolo11s.pt
  imgsz: 1280
  roi: auto          # auto | full | [x1, y1, x2, y2]
  tiles: [2, 1]      # 2 cột × 1 hàng trên vùng ROI, chồng lấn 20%
```

### Rà soát lỗi logic 11/11/2026

Đọc lại toàn bộ mã (lõi, 16 luật, đếm, bằng chứng, CSDL, dashboard) và sửa các lỗi sau; mỗi lỗi có bài kiểm thử trong
`tests/test_review_fixes.py`.

| Chỗ | Lỗi | Sửa |
|---|---|---|
| RB-11, HZ-02 | Đồng hồ "đã đỗ bao lâu" gắn với ID → xe bị che, tracker cấp ID mới là đếm lại từ 0 (xe đỗ trước cổng gần như không bao giờ đủ 60 s); đổi ID sau khi đã báo thì lập hồ sơ thứ hai; camera rung cũng làm đồng hồ về 0 | Đồng hồ gắn với **vị trí** (`StationaryMemory`): ID mới ở đúng chỗ cũ trong 10 s kế thừa mốc thời gian và trạng thái đã báo; camera rung giữ nguyên đồng hồ |
| RB-07 (PET) | Khi chưa hiệu chuẩn, lưới ô tính theo chiều cao hộp của TỪNG đối tượng → ô của người và ô của xe khác cỡ, không so được | So khoảng cách thật giữa chân xe và vệt người đi bộ vừa rời |
| RB-03 | Đồng thuận tính theo từng khung: 2 xe đi ngược trong làn trống bị bỏ sót; đường vắng (mỗi lúc 1 xe) thì khai báo ngược mũi tên vẫn báo giả đều đều | Tính theo **xe khác nhau trong 20 s**: ≥ 3 xe và > 50% đi ngược mới coi là khai báo sai |
| RB-04 | Chạm vạch liền rồi quay về làn cũ vẫn bị báo "đè vạch" | Cắt lần hai = huỷ; chỉ báo khi đã sang hẳn bên kia |
| RB-05 | Xe được xác nhận đúng lúc camera rung → mất hồ sơ; vạch dừng trỏ tới đèn không tồn tại → luật im lặng không chạy | Camera rung: tạm dừng, không tiêu mất ứng viên; xác nhận cả khi xe ĐÃ ở bên kia vạch xác nhận; `check` báo rõ đèn không tồn tại |
| RB-12 | Nhóm thêm/bớt một xe thành "nhóm mới" → nhiều hồ sơ cho một nhóm; trạng thái không bao giờ được dọn | Khoá nhóm ổn định (trùng ≥ 2 thành viên), dọn khi xe rời khung |
| Giờ hiệu lực | `active_hours: ["22:00-05:00"]` (qua nửa đêm) không bao giờ đúng | Xử lý khoảng qua nửa đêm |
| Tracker | Nguồn RTSP bỏ khung: thời gian nhảy nhưng tracker đếm theo số khung → giữ track "ma" quá lâu, dự đoán sai | Bước dự đoán và hạn giữ track theo thời gian thật |
| Pipeline | Danh sách xe vi phạm không bao giờ được dọn → mỗi khung hợp nhất biển số cho MỌI xe từng vi phạm (chậm dần khi chạy nhiều giờ); `new_track_ids` luôn rỗng | Dọn khi xe rời khung; biển số của hồ sơ đang chờ được giữ tới lúc ghi xong |
| Bằng chứng | Vùng riêng tư (`privacy_mask`) chỉ che ở ảnh tổng quan, KHÔNG che ở ảnh trước/sau, ảnh xe và clip; clip ghi sai tốc độ khung (video 25 fps → clip chạy chậm 20%) | Che ở mọi sản phẩm; ghi đúng tốc độ khung thực |
| Cấu hình | Lối vào được miễn của RB-04 không co giãn theo kích thước video; vạch thiếu điểm gây lỗi giữa chừng lúc chạy | Co giãn đủ; kiểm tra số điểm, khoảng cách > 0, ô đèn hợp lệ ngay khi nạp |
| Vạch đo tốc độ | Cắt vạch b trước a để lại trạng thái treo; vẫn đo khi camera rung | Bỏ lần cắt sai thứ tự; tạm dừng khi camera rung; dọn theo track |
| Báo cáo / dashboard | Bản đồ điểm nóng của báo cáo theo ngày gộp cả các ngày khác; mã truy cập có ký tự ngoài ASCII gây lỗi 500 | Lọc theo ngày; so sánh mã theo byte |

## 10. Giới hạn đã biết (nên đọc trước khi dùng thật)

- Model COCO **bỏ sót nhiều xe máy** khi đường quá đông (thử trên video ùn tắc Hà Nội: chỉ nhận ~20/100+ xe). Nên dùng
  `yolo11m/l` hoặc **huấn luyện thêm trên dữ liệu giao thông Việt Nam** (`training/README.md`).
- Chưa có `helmet.pt`, `plate_chars.pt` → RB-01 tắt, biển số chỉ lưu ảnh. Đây là hai model nên huấn luyện đầu tiên.
- Tốc độ đo bằng camera là **ước lượng** (phụ thuộc hiệu chuẩn); không dùng thay thiết bị đo đã kiểm định.
- Camera CCTV phải **cố định chắc**; video quay tay (TikTok) gần như toàn bộ rule hướng/tốc độ bị tạm ngưng.
- Khai báo sai vùng → kết quả sai theo (ví dụ một đa giác làn phủ cả hai chiều xe chạy sẽ báo ngược chiều chiều kia).
  Luôn kiểm tra bằng `preview` và vài phút video trước khi chạy dài.
- POV cần chiều cao gắn camera và đường chân trời gần đúng (`configs/pov_config.yaml`).

## 11. Kiểm thử

```
cd C:\yolo-al
.venv\Scripts\python -m pytest traffic_safety_ai\tests -q      (113 bài, ~30 giây, không cần GPU/video)
```

Gồm: hình học, homography, Kalman/tracker, động học, biển số VN, đèn tín hiệu, ego-motion, từng rule với thế giới tổng hợp,
CSDL, phiếu, báo cáo, dashboard (bảo mật đường dẫn, CSRF, token), công cụ khai báo, CLI, mô phỏng đầu-cuối CCTV + POV.

## 12. Cấu trúc

```
traffic_safety_ai/
  cli.py, __main__.py, main.py     dòng lệnh
  configs/                         cctv_config.yaml, pov_config.yaml, models.yaml, rules_catalog.yaml (căn cứ pháp lý)
  core/                            detector, tracker (ByteTrack tự viết), kalman, kinematics, homography, ego_motion,
                                   association (người–xe), traffic_light, attributes (mũ, điện thoại, hàng, xi-nhan), pipeline
  rules/                           16 rule + registry (bật/tắt theo năng lực) + catalog
  lpr/                             biển số VN: mẫu biển, màu biển, hợp nhất nhiều khung, bộ đọc
  evidence/                        CSDL, phiếu, ghi bằng chứng, làm mờ
  analytics/                       đếm lưu lượng, tốc độ, báo cáo
  dashboard/                       FastAPI + trang web
  tools/roi_calibrator.py          khai báo vùng/vạch, ảnh kiểm tra
  sim/                             mô phỏng CCTV + POV
  training/                        huấn luyện model bổ sung + mẫu dataset
  tests/                           kiểm thử tự động
  TRAFFIC.bat, start_traffic.ps1   chạy nhanh trên Windows
```

Thuật toán (ByteTrack, bù chuyển động camera kiểu BoT-SORT, TTC/PET, PCU/PHF/V85...) được **viết lại** theo ý tưởng
công khai; không chép mã nguồn của dự án khác.
