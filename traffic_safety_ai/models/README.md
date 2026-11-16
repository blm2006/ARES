# Thư mục model

Đặt các model bổ sung vào đây (tên đúng như `configs/models.yaml`):

| Tệp | Dùng cho |
|---|---|
| `helmet.pt` | RB-01 không đội mũ bảo hiểm |
| `plate.pt` | dò biển số (thiếu → bộ dò cổ điển, kém chính xác) |
| `plate_chars.pt` | đọc ký tự biển số (thiếu → chỉ lưu ảnh biển để người duyệt đọc) |
| `cargo.pt` | RB-09 hàng cồng kềnh (không bắt buộc với CCTV) |
| `lane_seg.pt` | RB-04 cho camera hành trình |
| `vehicle_orientation.pt` | tăng độ tin cậy RB-03 cho camera hành trình |

Model nền (`yolo11s.pt`, `yolo11n-pose.pt`) được Ultralytics tự tải lần đầu; máy không có Internet thì chép
`yolov8n.pt` của gói yolo-al vào thư mục cha (`C:\yolo-al\yolov8n.pt`) — cấu hình đã khai báo dự phòng.

Cách huấn luyện: xem `training/README.md`. Kiểm tra: `python -m traffic_safety_ai check`.
