"""AI bổ sung cho phần Hành vi: các model YOLO phát hiện VẬT/HIỆN TƯỢNG nguy hiểm, chạy cạnh YOLO-pose.

    cháy / khói  (FIRE)     -> báo cả camera, không gắn với người
    hút thuốc    (SMOKING)  -> điếu thuốc ở gần miệng / tay của 1 người -> báo người đó

Luồng: khung hình -> YOLO của từng mục (1 lần/giây/camera, ARES_HAZARD_FPS) -> xác nhận theo thời gian
(phải thấy ở nhiều lần kiểm gần nhất, lọc khung báo nhầm lẻ tẻ) -> sự kiện đi chung đường với té ngã / đồ bảo hộ
(dashboard, clip, Đúng / Báo nhầm). Ảnh gốc + hộp của mỗi sự kiện được lưu để TRAIN DẦN model (python -m ares.hazards train).

Model: tải bằng CAI_AI_BAO_HO.bat (python -m ares.hazards setup). Model đã train lại từ đánh giá nằm ở
<storage_root>/behavior/hazards/<mục>/current.pt và được ưu tiên dùng.

Lệnh:  python -m ares.hazards setup | check | train [--only fire]
"""

from __future__ import annotations

# mục -> thông tin. "labels": tên lớp của model -> nhãn hiển thị (lớp không có ở đây thì bỏ qua).
HAZARDS = {
    "fire": {
        "name": "Cháy / khói", "kind": "FIRE", "weights": "hazard_fire_smoke_dfire_y8n.pt",
        "url": "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt",
        "sha256": "b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f",
        "source": "huggingface.co/rabahdev/fire-smoke-yolov8n (YOLOv8n, dữ liệu D-Fire 21.000 ảnh)",
        "license": "AGPL-3.0", "labels": {"fire": "LỬA", "smoke": "KHÓI"},
        "per_person": False, "imgsz": 640,
    },
    "smoking": {
        "name": "Hút thuốc", "kind": "SMOKING", "weights": "hazard_smoking_y11m.pt",
        "url": "https://huggingface.co/Beehzod/smoking-detection-finetuned/resolve/main/best.pt",
        "sha256": "3f3ea54995cba4bbbd82960dce5c6be2f16cf8fe024c2a5892800a0d3f44b9c1",
        "source": "huggingface.co/Beehzod/smoking-detection-finetuned (YOLO11m, lớp cigarette)",
        "license": "MIT", "labels": {"cigarette": "THUỐC LÁ"},
        "per_person": True, "imgsz": 960,  # điếu thuốc là vật rất nhỏ: cho model nhìn ảnh to hơn
    },
}

KIND_OF = {k: v["kind"] for k, v in HAZARDS.items()}
