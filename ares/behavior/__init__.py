"""Nhận diện hành vi bất thường cho ARES: té ngã, nằm bất động (nghi ngất/đột quỵ), ngồi gục, đánh nhau/xô đẩy.

Luồng xử lý (mỗi camera ARES đang có):

    khung hình -> YOLO-pose (17 điểm khớp COCO) -> bám vết người (tracker.py)
               -> đặc trưng tư thế + chuyển động (skeleton.py)
               -> luật (rules.py)  +  model chuỗi khung xương ST-GCN-lite nếu đã train (model.py)
               -> sự kiện -> dashboard /behavior + clip bằng chứng (store.py, monitor.py)

Module này KHÔNG điều khiển đèn/điều hòa và không đọc trạng thái của control_engine.
Camera chỉ phát hiện DẤU HIỆU (ngã, nằm/gục bất động lâu) — không chẩn đoán bệnh.

Lệnh:  python -m ares.behavior download | extract | train | video <file>
"""

KINDS = ("FALL", "LYING_DOWN", "LYING_STILL", "SLUMPED", "FIGHT")

KIND_VI = {
    "FALL": "Té ngã",
    "LYING_DOWN": "Đột ngột nằm xuống / nằm dưới sàn",
    "LYING_STILL": "Ngã gục / nằm bất động",
    "SLUMPED": "Ngồi gục bất động",
    "FIGHT": "Đánh nhau / xô đẩy",
}

# Mức độ hiển thị: nằm bất động sau cú ngã được nâng lên "error" (khẩn cấp) ở monitor.
KIND_SEVERITY = {"FALL": "error", "LYING_DOWN": "error", "LYING_STILL": "error", "SLUMPED": "warn", "FIGHT": "error"}

# Lớp của model phân loại chuỗi khung xương (thứ tự cố định — đừng đổi khi đã có model đã train).
CLASSES = ("normal", "fall", "lying", "slumped", "fight")
CLASS_VI = {"normal": "Bình thường", "fall": "Đang ngã", "lying": "Nằm", "slumped": "Ngồi gục", "fight": "Đánh nhau"}
