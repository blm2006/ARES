"""Train dần model AI bổ sung (cháy/khói, hút thuốc) từ đánh giá Đúng / Báo nhầm, CÓ CỔNG KIỂM ĐỊNH.

1. Dựng bộ dữ liệu YOLO từ sự kiện đã đánh giá (feedback.build_dataset).
2. Fine-tune từ model đang dùng: đóng băng 10 lớp đầu (backbone), learning rate nhỏ, ít vòng — để học cảnh nơi lắp
   mà không quên những gì model đã biết.
3. Cổng: so model cũ và mới trên TOÀN BỘ ảnh đã đánh giá — số ảnh Báo nhầm mà model vẫn báo (báo nhầm còn lại) và
   số ảnh Đúng mà model còn thấy (độ nhạy). Chỉ dùng model mới nếu không tệ hơn ở cả hai và tốt hơn ở ít nhất một.
4. Dùng: chép thành behavior/hazards/<mục>/current.pt (bản trước giữ lại prev_<giờ>.pt), nạp lại không cần khởi động.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

MIN_POS, MIN_NEG = 2, 2


def _score(weights: Path, items: list[dict], conf: float, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(weights))
    pos = [it for it in items if it["pos"]]
    neg = [it for it in items if not it["pos"]]
    hit = lambda its: sum(1 for r in m.predict([str(i["img"]) for i in its], imgsz=imgsz, conf=conf, device=device,  # noqa: E731
                                               verbose=False) if r.boxes is not None and len(r.boxes)) if its else 0
    return {"recall": hit(pos), "n_pos": len(pos), "false_alarms": hit(neg), "n_neg": len(neg)}


def train_one(cfg, name: str, epochs: int = 30, device: str = "") -> dict:
    from ultralytics import YOLO

    from ares.behavior.pose import resolve_device
    from ares.hazards import HAZARDS
    from ares.hazards.detector import imgsz_for, weights_for
    from ares.hazards.feedback import build_dataset

    label = HAZARDS[name]["name"]
    base = weights_for(cfg, name)
    if not base.exists():
        return {"ok": False, "name": name, "message": f"{label}: chưa cài model — chạy CAI_AI_BAO_HO.bat."}
    root = Path(cfg.behavior_dir) / "hazards" / name
    root.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    ds = build_dataset(cfg, name, {int(k): str(v) for k, v in names.items()}, root / "dataset")
    c = ds["counts"]
    n_pos = sum(1 for it in ds["items"] if it["pos"])
    n_neg = len(ds["items"]) - n_pos
    if n_pos < MIN_POS or n_neg < MIN_NEG:
        return {"ok": False, "name": name, "counts": c,
                "message": f"{label}: chưa đủ ảnh để train — có {n_pos} ảnh Đúng, {n_neg} ảnh Báo nhầm (cần ≥ {MIN_POS} mỗi loại). "
                           "Đánh giá thêm cảnh báo, hoặc thêm ảnh tự gán nhãn vào "
                           f"behavior/hazards/{name}/extra/."}
    dev = resolve_device(device or cfg.behavior_device or cfg.device)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"{label}: train {c['train']} ảnh, kiểm {c['val']} ảnh ({n_pos} có vật, {n_neg} không) trên {dev}")
    isz = imgsz_for(cfg, name)
    model.train(data=str(ds["yaml"]), epochs=int(epochs), imgsz=isz, batch=8, freeze=10, lr0=0.002,
                lrf=0.2, warmup_epochs=0, mosaic=0.5, close_mosaic=0, device=dev, workers=0, plots=False,
                project=str(root / "runs"), name=stamp, exist_ok=True, verbose=False, patience=0)
    run = root / "runs" / stamp / "weights"
    new = run / "best.pt" if (run / "best.pt").exists() else run / "last.pt"
    if not new.exists():
        return {"ok": False, "name": name, "counts": c, "message": f"{label}: train không tạo được model."}
    before = _score(base, ds["items"], cfg.hazard_conf, isz, dev)
    after = _score(new, ds["items"], cfg.hazard_conf, isz, dev)
    better = (after["false_alarms"] <= before["false_alarms"] and after["recall"] >= before["recall"]
              and (after["false_alarms"] < before["false_alarms"] or after["recall"] > before["recall"]))
    rec = {"t": time.time(), "stamp": stamp, "counts": c, "before": before, "after": after, "promoted": better,
           "base": str(base)}
    hist = root / "history.json"
    try:
        old = json.loads(hist.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = []
    hist.write_text(json.dumps((old + [rec])[-50:], ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    msg = (f"{label}: báo nhầm còn {after['false_alarms']}/{after['n_neg']} (trước {before['false_alarms']}), "
           f"vẫn thấy {after['recall']}/{after['n_pos']} (trước {before['recall']}). ")
    if not better:
        return {"ok": True, "name": name, "promoted": False, "before": before, "after": after, "counts": c,
                "message": msg + "Model mới KHÔNG tốt hơn — giữ model cũ."}
    cur = root / "current.pt"
    if cur.exists():
        shutil.copyfile(cur, root / f"prev_{stamp}.pt")
    shutil.copyfile(new, cur)
    return {"ok": True, "name": name, "promoted": True, "before": before, "after": after, "counts": c,
            "message": msg + "Đã dùng model mới."}


def train_all(cfg, epochs: int = 30, only: str = "") -> list[dict]:
    from ares.hazards.detector import wanted
    from ares.hazards.feedback import collect_ppe_for_labeling

    out = [train_one(cfg, n, epochs) for n in wanted(cfg) if not only or n == only]
    k = collect_ppe_for_labeling(cfg)
    if k:
        out.append({"ok": True, "name": "ppe", "promoted": False,
                    "message": f"Đồ bảo hộ: thêm {k} ảnh báo nhầm vào behavior/hazards/ppe/can_gan_nhan/ — cần gán nhãn "
                               "(vd bằng Label Studio / Roboflow) rồi mới train lại được model đồ bảo hộ."})
    return out
