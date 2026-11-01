"""Tiến trình con: fine-tune YOLO + đo model ứng viên và model đang chạy trên các tập giữ lại.

    python -m ares.train_job <run_dir>/job.json

Đọc job.json (do ares.retrain ghi), ghi result.json. Không import gì từ phần còn lại của ARES
ngoài thư viện chuẩn + ultralytics, để chạy độc lập và chết một mình nếu CUDA lỗi.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path


def _device(want: str) -> str:
    if want != "auto":
        return want
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _write_yaml(path: Path, run_dir: Path, train: str, val: str, names: dict) -> Path:
    lines = [f"path: {json.dumps(str(run_dir))}", f"train: {json.dumps(train)}", f"val: {json.dumps(val)}", "names:"]
    lines += [f"  {int(k)}: {json.dumps(str(v))}" for k, v in sorted(names.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _eval(weights: str, data_yaml: Path, imgsz: int, device: str, person_cls: int = 0) -> dict:
    """Đo riêng lớp 'person' (tập replay do người dùng cấp có thể chứa lớp khác)."""
    from ultralytics import YOLO

    box = YOLO(weights).val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False, plots=False).box
    try:
        idx = list(box.ap_class_index).index(person_cls)
        p, r, ap50, _ = box.class_result(idx)
        return {"map50": float(ap50), "recall": float(r), "precision": float(p)}
    except ValueError:  # tập không có nhãn person: dùng trung bình tất cả lớp có dữ liệu
        return {"map50": float(box.map50), "recall": float(box.mr), "precision": float(box.mp)}


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    run_dir = Path(job["run_dir"])
    result_file = run_dir / "result.json"
    t0 = time.time()
    try:
        from ultralytics import YOLO

        device = _device(job.get("device", "auto"))
        base = YOLO(job["base_weights"])  # LUÔN từ checkpoint gốc
        names = dict(base.names)  # giữ nguyên đầu 80 lớp COCO => nạp đủ trọng số, chỉ nhãn 'person' được dạy thêm

        data_yaml = _write_yaml(run_dir / "data.yaml", run_dir, job["train_list"], job["val_list"], names)
        kwargs = dict(
            data=str(data_yaml), epochs=int(job["epochs"]), imgsz=int(job["imgsz"]), batch=int(job["batch"]),
            lr0=float(job["lr0"]), optimizer="AdamW", project=str(run_dir), name="train", exist_ok=True,
            workers=int(job.get("workers", 2)), seed=0, plots=False, patience=10,
            hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=0.0, translate=0.08, scale=0.4, shear=0.0,
            flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, close_mosaic=5,
        )
        if int(job.get("freeze", 0)) > 0:
            kwargs["freeze"] = int(job["freeze"])
        try:
            base.train(device=device, **kwargs)
        except Exception as e:  # GPU lỗi/hết VRAM -> thử lại bằng CPU thay vì bỏ cuộc
            if device == "cpu":
                raise
            print(f"[train_job] GPU lỗi ({e}); thử lại bằng CPU", flush=True)
            base = YOLO(job["base_weights"])
            device = "cpu"
            base.train(device=device, **kwargs)

        trainer = base.trainer
        best = Path(getattr(trainer, "best", "") or "")
        if not best.exists():
            best = Path(trainer.save_dir) / "weights" / "last.pt"
        if not best.exists():
            raise FileNotFoundError("Không thấy best.pt/last.pt sau khi train")

        metrics: dict = {"candidate": {}, "current": {}}
        for key, lst in (("human", job.get("val_human_list")), ("replay", job.get("val_replay_list"))):
            if not lst:
                continue
            y = _write_yaml(run_dir / f"data_val_{key}.yaml", run_dir, lst, lst, names)
            metrics["candidate"][key] = _eval(str(best), y, int(job["imgsz"]), device)
            metrics["current"][key] = _eval(job["current_weights"], y, int(job["imgsz"]), device)
        out = {"ok": True, "weights": str(best), "metrics": metrics, "device": device,
               "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        out = {"ok": False, "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 1)}
    result_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Cách dùng: python -m ares.train_job <job.json>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
