"""Dữ liệu train dần cho các model PHÁT HIỆN VẬT (AI bổ sung + đồ bảo hộ).

Mỗi sự kiện cháy/khói, hút thuốc, vi phạm đồ bảo hộ lưu thêm vào thư mục sự kiện:
    raw.jpg    ảnh GỐC lúc phát hiện (chưa vẽ gì, cạnh dài ≤ 1280 px)
    dets.json  các hộp model thấy trên đúng ảnh đó (tọa độ theo raw.jpg)

Người trực bấm Đúng / Báo nhầm trên dashboard -> đánh giá về máy AI -> ``build_dataset`` biến thành mẫu YOLO:
    Đúng      -> ảnh + hộp model đã thấy (mẫu dương — model "nhớ" cảnh ở nơi lắp)
    Báo nhầm  -> ảnh + nhãn RỖNG (mẫu âm — model học rằng cảnh này KHÔNG có lửa / khói / điếu thuốc)
Vi phạm đồ bảo hộ bị Báo nhầm nghĩa là model SÓT món đồ (không tự biết hộp ở đâu) -> ảnh được gom vào
``behavior/hazards/ppe/can_gan_nhan/`` để người gán nhãn rồi mới train được (không tự train).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

MAX_SIDE = 1280


def save_sample(d: Path, raw: np.ndarray, ev, ca) -> None:
    import cv2

    h, w = raw.shape[:2]
    sc = min(1.0, MAX_SIDE / max(h, w))
    img = cv2.resize(raw, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1.0 else raw
    cv2.imwrite(str(d / "raw.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    out = {"kind": ev.kind, "w": img.shape[1], "h": img.shape[0], "t": ev.t, "hazard": {}, "ppe": []}
    if ca is not None:
        if abs(ca.hz.t - ev.t) < 1e-6:  # hộp của AI bổ sung lấy trên CHÍNH khung này
            out["hazard"] = {name: [[x.cls, *[round(v * sc, 1) for v in x.box], round(x.conf, 3)] for x in dets]
                             for name, dets in ca.hz.dets.items()}
        if ca.ppe_on and ev.t - ca.ppe_t <= 1.0:
            out["ppe"] = [[x.item, int(x.worn), *[round(v * sc, 1) for v in x.box], round(x.conf, 3)] for x in ca.ppe_dets]
    (d / "dets.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def _split(eid: str) -> str:
    return "val" if int(hashlib.md5(eid.encode()).hexdigest(), 16) % 5 == 0 else "train"


def reviewed_samples(cfg, kind: str) -> list[dict]:
    """Sự kiện đã đánh giá có ảnh gốc: [{id, verdict, img, dets}]"""
    from ares.behavior.store import EventStore

    root = Path(cfg.behavior_dir)
    if not (root / "behavior.db").exists():
        return []
    st = EventStore(root)
    try:
        rows = [r for r in st.reviewed() if r["kind"] == kind]
    finally:
        st.close()
    out = []
    for r in rows:
        d = root / r["dir"]
        if (d / "raw.jpg").exists() and (d / "dets.json").exists():
            out.append({"id": r["id"], "verdict": r["verdict"], "img": d / "raw.jpg",
                        "dets": json.loads((d / "dets.json").read_text(encoding="utf-8"))})
    return out


def build_dataset(cfg, name: str, names: dict[int, str], out: Path) -> dict:
    """Tạo bộ dữ liệu YOLO cho model của mục ``name`` (cháy/khói, hút thuốc) từ sự kiện đã đánh giá.

    names: lớp của model đang dùng (chỉ số -> tên). Thêm ảnh tự gán nhãn: đặt vào
    behavior/hazards/<mục>/extra/images + extra/labels (định dạng YOLO, đúng chỉ số lớp của model)."""
    from ares.hazards import HAZARDS

    kind = HAZARDS[name]["kind"]
    idx = {v.lower(): k for k, v in names.items()}
    if out.exists():
        shutil.rmtree(out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True)
        (out / "labels" / sp).mkdir(parents=True)
    n = {"pos": 0, "neg": 0, "extra": 0, "train": 0, "val": 0}
    items = []
    for s in reviewed_samples(cfg, kind):
        boxes = [b for b in (s["dets"].get("hazard") or {}).get(name, []) if b[0].lower() in idx and b[5] >= cfg.hazard_conf]
        if s["verdict"] == "true" and not boxes:
            continue  # Đúng nhưng không còn hộp nào (hiếm): bỏ, không đoán
        lines = []
        if s["verdict"] == "true":
            W, H = s["dets"]["w"], s["dets"]["h"]
            for c, x1, y1, x2, y2, _ in boxes:
                lines.append(f"{idx[c.lower()]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            n["pos"] += 1
        else:
            n["neg"] += 1
        sp = _split(s["id"])
        shutil.copyfile(s["img"], out / "images" / sp / f"{s['id']}.jpg")
        (out / "labels" / sp / f"{s['id']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n[sp] += 1
        items.append({"img": out / "images" / sp / f"{s['id']}.jpg", "pos": bool(lines)})
    extra = Path(cfg.behavior_dir) / "hazards" / name / "extra"
    for img in sorted((extra / "images").glob("*.jpg")) if (extra / "images").exists() else []:
        lab = extra / "labels" / (img.stem + ".txt")
        sp = _split(img.stem)
        shutil.copyfile(img, out / "images" / sp / f"x_{img.name}")
        txt = lab.read_text(encoding="utf-8") if lab.exists() else ""
        (out / "labels" / sp / f"x_{img.stem}.txt").write_text(txt, encoding="utf-8")
        n["extra"] += 1
        n[sp] += 1
        items.append({"img": out / "images" / sp / f"x_{img.name}", "pos": bool(txt.strip())})
    if n["val"] == 0 and n["train"] > 1:  # tập nhỏ: đưa 1 ảnh sang val để Ultralytics chạy được
        f = next((out / "images" / "train").glob("*.jpg"))
        f.rename(out / "images" / "val" / f.name)
        (out / "labels" / "train" / (f.stem + ".txt")).rename(out / "labels" / "val" / (f.stem + ".txt"))
        for it in items:
            if it["img"] == out / "images" / "train" / f.name:
                it["img"] = out / "images" / "val" / f.name
        n["val"], n["train"] = 1, n["train"] - 1
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
                    + "".join(f"  {k}: {v}\n" for k, v in sorted(names.items())), encoding="utf-8")
    return {"counts": n, "yaml": yaml, "items": items}


def collect_ppe_for_labeling(cfg) -> int:
    """Vi phạm đồ bảo hộ bị Báo nhầm -> chép ảnh gốc ra thư mục chờ gán nhãn. Trả số ảnh mới."""
    dst = Path(cfg.behavior_dir) / "hazards" / "ppe" / "can_gan_nhan"
    dst.mkdir(parents=True, exist_ok=True)
    k = 0
    for s in reviewed_samples(cfg, "PPE"):
        if s["verdict"] != "false":
            continue
        f = dst / f"{s['id']}.jpg"
        if not f.exists():
            shutil.copyfile(s["img"], f)
            (dst / f"{s['id']}.json").write_text(json.dumps(s["dets"], ensure_ascii=False), encoding="utf-8")
            k += 1
    return k
