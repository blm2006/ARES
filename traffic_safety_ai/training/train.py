"""Huấn luyện model bổ sung bằng Ultralytics YOLO, với tham số mặc định hợp lý cho từng bài toán.

    python -m traffic_safety_ai train --task helmet --data training/datasets/helmet.yaml --epochs 80

Xong sẽ chép best.pt vào traffic_safety_ai/models/<tên chuẩn>.pt để hệ thống tự nhận (xem configs/models.yaml).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from traffic_safety_ai import MODELS_DIR, PACKAGE_ROOT

# task → (model khởi tạo, imgsz, tên file đích, loại bài toán, ghi chú)
TASKS: dict[str, tuple[str, int, str, str, str]] = {
    "helmet": ("yolo11n.pt", 224, "helmet.pt", "detect",
               "Ảnh cắt vùng đầu/thân trên người đi xe máy; lớp: helmet, no_helmet"),
    "plate": ("yolo11n.pt", 416, "plate.pt", "detect",
              "Ảnh cắt phương tiện; lớp: plate (có thể tách plate_1line / plate_2line)"),
    "plate_chars": ("yolo11n.pt", 256, "plate_chars.pt", "detect",
                    "Ảnh biển số đã cắt; mỗi ký tự 1 hộp, tên lớp đúng ký tự: 0–9, A–Z (không dấu)"),
    "cargo": ("yolo11n.pt", 640, "cargo.pt", "detect", "Ảnh toàn cảnh/ảnh cắt xe máy; lớp: cargo"),
    "phone": ("yolo11n.pt", 320, "phone.pt", "detect", "Ảnh cắt người lái; lớp: phone"),
    "lane_seg": ("yolo11n-seg.pt", 640, "lane_seg.pt", "segment",
                 "Ảnh camera hành trình; lớp đa giác: road, sidewalk, crosswalk, solid_line"),
    "orientation": ("yolo11n-cls.pt", 224, "vehicle_orientation.pt", "classify",
                    "Thư mục ảnh cắt xe chia 2 lớp: front/ (thấy đầu xe), rear/ (thấy đuôi xe)"),
}


def train(task: str, data: str, epochs: int = 100, imgsz: int | None = None, batch: int = 16,
          model: str | None = None, device: str = "0", name: str | None = None) -> int:
    from ultralytics import YOLO

    if task not in TASKS:
        raise SystemExit(f"task không hợp lệ: {task}. Chọn một trong: {', '.join(TASKS)}")
    init, sz, target, kind, note = TASKS[task]
    data_path = Path(data)
    if not data_path.exists() and (PACKAGE_ROOT / data).exists():
        data_path = PACKAGE_ROOT / data
    if not data_path.exists():
        raise SystemExit(f"Không thấy dataset: {data}")
    print(f"Huấn luyện '{task}' ({kind}) — {note}")
    print(f"  model khởi tạo: {model or init} | imgsz {imgsz or sz} | epochs {epochs} | batch {batch}")
    m = YOLO(model or init)
    kw = dict(data=str(data_path), epochs=epochs, imgsz=imgsz or sz, batch=batch, device=device,
              project=str(PACKAGE_ROOT / "runs_train"), name=name or task, exist_ok=True, patience=30,
              plots=True, verbose=False)
    if kind == "detect":
        # biển số / ký tự: KHÔNG lật ngang (chữ bị ngược); mũ bảo hiểm thì lật được
        kw["fliplr"] = 0.0 if task in ("plate", "plate_chars") else 0.5
    res = m.train(**kw)
    save_dir = Path(getattr(res, "save_dir", None) or getattr(m.trainer, "save_dir", ""))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print("Không thấy best.pt — xem log huấn luyện.")
        return 1
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_DIR / target
    if dst.exists():
        dst.rename(dst.with_suffix(".bak.pt"))
    shutil.copy2(best, dst)
    print(f"Đã chép {best} → {dst}")
    print("Chạy 'python -m traffic_safety_ai check' để xác nhận tình huống liên quan đã BẬT.")
    return 0
