# Huấn luyện model bổ sung

Model nền COCO (`yolo11s.pt` / `yolov8n.pt`) đã đủ cho: xe, người, đèn tín hiệu → chạy được RB-03, RB-04 (CCTV), RB-05,
RB-06, RB-07, RB-08, RB-10, RB-11, RB-12, HZ-01…HZ-04. Các tình huống còn lại cần model riêng; thiếu model thì tình huống
đó **tự tắt** và lệnh `check` ghi rõ lý do.

| Model | Dùng cho | Lệnh | Dữ liệu |
|---|---|---|---|
| `models/helmet.pt` | RB-01 không đội mũ | `train --task helmet` | ảnh cắt người đi xe máy, lớp `helmet`, `no_helmet` |
| `models/plate.pt` | dò biển số (thay bộ dò cổ điển) | `train --task plate` | ảnh cắt xe, lớp `plate_1line`, `plate_2line` |
| `models/plate_chars.pt` | đọc ký tự biển (OCR) | `train --task plate_chars` | ảnh biển đã cắt, mỗi ký tự 1 hộp, tên lớp = ký tự |
| `models/cargo.pt` | RB-09 hàng cồng kềnh (không bắt buộc với CCTV) | `train --task cargo` | hộp bao phần hàng |
| `yolo11n-pose.pt` | RB-02 điện thoại (tư thế tay) | tự tải | — |
| `models/lane_seg.pt` | RB-04 kênh POV | `train --task lane_seg` | đa giác road/sidewalk/crosswalk/solid_line |
| `models/vehicle_orientation.pt` | tăng độ tin RB-03 POV | `train --task orientation` | thư mục `front/`, `rear/` |

## Quy trình đề xuất

1. **Thu dữ liệu tại chỗ**: chạy `run` trên video cổng trường; ảnh cắt xe/biển nằm trong thư mục bằng chứng.
   Dữ liệu thật tại địa điểm luôn tốt hơn dữ liệu tải về (góc camera, ánh sáng, loại mũ/nón hay gặp).
2. **Gán nhãn**: CVAT, Label Studio hoặc Roboflow (xuất định dạng YOLO). Với biển số, gán **đúng ký tự** cho từng
   hộp; biển 2 dòng vẫn gán từng ký tự — hệ thống tự ghép theo dòng.
3. **Sửa file mẫu** trong `training/datasets/*.yaml` (đường dẫn TUYỆT ĐỐI tới dữ liệu).
4. **Huấn luyện** (máy có GPU):
   ```
   python -m traffic_safety_ai train --task helmet --data training/datasets/helmet.yaml --epochs 80
   ```
   Xong, `best.pt` được chép vào `traffic_safety_ai/models/` đúng tên trong `configs/models.yaml`.
5. **Kiểm tra**: `python -m traffic_safety_ai check` → tình huống liên quan chuyển sang BẬT.
6. (Tuỳ chọn) **Tăng tốc**: `python -m traffic_safety_ai export-models --which base,helmet --format engine`
   (TensorRT FP16, cần GPU NVIDIA) hoặc `--format onnx`.

## Nguồn tham khảo (chỉ lấy ý tưởng / dữ liệu có giấy phép phù hợp)

- Bộ dữ liệu dò biển số VN công khai trên HuggingFace (ví dụ `neitpm/vn-plate-detection-benchmark`) — kiểm tra giấy phép
  trước khi dùng, chuyển nhãn sang YOLO nếu cần.
- Thư viện `fast-plate-ocr` (đọc biển 1 lần, không cần dò ký tự) — đặt `plate_ocr.backend: fast_plate_ocr` trong
  `configs/models.yaml` (cài thêm: `pip install fast-plate-ocr[onnx]`).
- Không chép mã nguồn của dự án khác vào gói này; các thuật toán ở đây được viết lại.

## Lưu ý quyền riêng tư

Ảnh người/biển số dùng để huấn luyện là dữ liệu cá nhân (Luật Bảo vệ dữ liệu cá nhân 2025): chỉ dùng cho mục đích
đã thông báo, lưu trên máy nội bộ, xoá khi không còn cần.
