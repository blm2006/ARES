"""Train model ST-GCN-lite trên khung xương đã tách (+ sự kiện đã duyệt trên dashboard).

    python -m ares.behavior train                  # mặc định 60 epoch, tự chọn GPU
    python -m ares.behavior train --epochs 100 --no-promote

Kết quả: <storage_root>/behavior/models/behavior-YYYYmmdd-HHMM.pt (+ báo cáo .txt). Mặc định chép thành
current.pt để ARES dùng ngay ở lần khởi động sau (ARES_BEHAVIOR_MODE=hybrid|model).

Dữ liệu:
  * datasets/keypoints/*.npz  (python -m ares.behavior extract)
  * sự kiện trên dashboard đã bấm "Đúng" (nhãn = loại sự kiện) hoặc "Báo nhầm" (nhãn = bình thường) — đây là cách
    model "tự học thêm" từ chính camera của bạn.
Chia train/val THEO CLIP (không để cửa sổ của cùng một clip lọt vào cả hai phía).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from ares.behavior import CLASSES
from ares.behavior.model import SPAN_S, T_STEPS, augment, build_net, build_window

KIND_TO_CLASS = {"FALL": "fall", "LYING_DOWN": "lying", "LYING_STILL": "lying", "SLUMPED": "slumped", "FIGHT": "fight"}
STEP_S = 0.25


def _ref_h(box: np.ndarray) -> float:
    ok = np.isfinite(box[:, 0])
    if not ok.any():
        return 100.0
    b = box[ok]
    return float(np.percentile(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]), 80))


def _window_label(lab: np.ndarray) -> int | None:
    """Nhãn cửa sổ từ nhãn từng khung. None = mơ hồ, bỏ."""
    n = len(lab)
    if n == 0:
        return None
    cnt = {c: int((lab == i).sum()) for i, c in enumerate(CLASSES)}
    if cnt["fight"] >= 0.5 * n:
        return CLASSES.index("fight")
    if cnt["fall"] >= max(2, 0.15 * n):
        return CLASSES.index("fall")
    for c in ("lying", "slumped", "normal"):
        if cnt[c] >= 0.6 * n:
            return CLASSES.index(c)
    return None


def windows_from_clip(d: dict) -> tuple[list[np.ndarray], list[int]]:
    times, labels, kp, box = d["times"], d["labels"], d["kp"], d["box"]
    P = kp.shape[1]
    xs, ys = [], []
    refs = [_ref_h(box[:, p]) for p in range(P)]
    vis = [np.isfinite(kp[:, p, 0, 0]) for p in range(P)]
    t_first, t_last = float(times[0]), float(times[-1])
    ends = np.arange(min(t_first + 0.6 * SPAN_S, t_last), t_last + 1e-6, STEP_S)  # clip ngắn (~2 s) vẫn có vài cửa sổ
    for te in ends:
        sel = (times > te - SPAN_S) & (times <= te)
        if sel.sum() < 4:
            continue
        y = _window_label(labels[sel])
        if y is None:
            continue
        mains = [p for p in range(P) if vis[p][sel].mean() >= 0.5][:2]
        for m in mains:
            others = [p for p in range(P) if p != m and vis[p][sel].any()]
            nb = None
            if others:
                cm = np.nanmean(box[sel, m], axis=0)
                cmc = ((cm[0] + cm[2]) / 2, (cm[1] + cm[3]) / 2)

                def dist(p):
                    b = np.nanmean(box[sel, p], axis=0)
                    return np.hypot((b[0] + b[2]) / 2 - cmc[0], (b[1] + b[3]) / 2 - cmc[1])
                nb = min(others, key=dist)
            ok = vis[m]
            t2 = k2 = None
            if nb is not None:
                ok2 = vis[nb]
                t2, k2 = times[ok2], kp[ok2, nb]
            xs.append(build_window(times[ok], kp[ok, m], te, refs[m], t2, k2))
            ys.append(y)
    return xs, ys


def windows_from_event(ev: dict, root: Path) -> tuple[list[np.ndarray], list[int]]:
    p = root / ev["dir"] / "keypoints.npz"
    if not p.exists() or "t0" not in (z := np.load(p)):
        return [], []
    t_ev = float(z["t_event"][0])
    t0, k0, b0 = z["t0"], z["kp0"], z["box0"]
    t1 = z["t1"] if "t1" in z else None
    k1 = z["kp1"] if "kp1" in z else None
    ref = _ref_h(b0)
    if ev["kind"] in ("FIRE", "SMOKING"):  # AI bổ sung: train model YOLO riêng (python -m ares.hazards train)
        return [], []
    if ev["verdict"] == "false":
        y, lo, hi = CLASSES.index("normal"), t_ev - 10, t_ev
    else:
        c = KIND_TO_CLASS.get(ev["kind"])
        if c is None:
            return [], []
        y = CLASSES.index(c)
        lo, hi = {"fall": (t_ev - 1.5, t_ev - 0.25), "fight": (t_ev - 1.5, t_ev)}.get(c, (t_ev - 10, t_ev))
    xs, ys = [], []
    for te in np.arange(max(lo, float(t0[0]) + SPAN_S * 0.75), hi + 1e-6, STEP_S):
        xs.append(build_window(t0, k0, te, ref, t1, k1))
        ys.append(y)
    return xs, ys


def load_all(cfg) -> list[tuple[str, list, list]]:
    from ares.behavior.store import EventStore

    root = Path(cfg.behavior_dir)
    groups = []
    for f in sorted((root / "datasets" / "keypoints").glob("*.npz")):
        with np.load(f) as z:
            d = {k: z[k] for k in ("times", "labels", "kp", "box")}
        xs, ys = windows_from_clip(d)
        if xs:
            groups.append((f.stem, xs, ys))
    if (root / "behavior.db").exists():
        st = EventStore(root)
        try:
            for ev in st.reviewed():
                xs, ys = windows_from_event(ev, root)
                if xs:
                    groups.append((f"event__{ev['id']}", xs, ys))
        finally:
            st.close()
    return groups


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    K = len(CLASSES)
    cm = np.zeros((K, K), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    per = {}
    f1s = []
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": round(float(prec), 3), "recall": round(float(rec), 3), "f1": round(float(f1), 3),
                  "support": int(cm[i].sum())}
        if cm[i].sum():
            f1s.append(f1)
    return {"macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4), "accuracy": round(float((y == p).mean()), 4),
            "per_class": per, "confusion": cm.tolist()}


def train(cfg, epochs: int = 60, batch: int = 128, lr: float = 2e-3, val_frac: float = 0.2, seed: int = 0,
          promote: bool = True, device: str = "") -> Path | None:
    import torch
    from torch import nn

    from ares.behavior.pose import resolve_device

    rng = np.random.default_rng(seed)
    groups = load_all(cfg)
    if not groups:
        print("Chưa có dữ liệu. Chạy: python -m ares.behavior download  rồi  python -m ares.behavior extract")
        return None
    # ---- chia theo clip, cố gắng mỗi lớp có mặt ở val
    order = list(range(len(groups)))
    rng.shuffle(order)
    val_idx, cls_seen = set(), set()
    n_val = max(1, int(round(val_frac * len(groups))))
    for i in order:  # ưu tiên đưa vào val những clip có lớp val chưa có
        lab = set(groups[i][2])
        if len(val_idx) < n_val and (lab - cls_seen or len(val_idx) < n_val // 2):
            val_idx.add(i)
            cls_seen |= lab
    tr_x = [x for i, g in enumerate(groups) if i not in val_idx for x in g[1]]
    tr_y = [y for i, g in enumerate(groups) if i not in val_idx for y in g[2]]
    va_x = [x for i, g in enumerate(groups) if i in val_idx for x in g[1]]
    va_y = [y for i, g in enumerate(groups) if i in val_idx for y in g[2]]
    if not tr_x or not va_x:
        print("Quá ít clip để chia train/val (cần >= 2 clip).")
        return None
    Xtr, Ytr = np.stack(tr_x), np.asarray(tr_y)
    Xva, Yva = np.stack(va_x), np.asarray(va_y)
    cnt = np.bincount(Ytr, minlength=len(CLASSES))
    print(f"{len(groups)} clip -> train {len(Xtr)} cửa sổ ({len(groups) - len(val_idx)} clip), val {len(Xva)} ({len(val_idx)} clip)")
    print("  số cửa sổ train theo lớp: " + ", ".join(f"{c}={n}" for c, n in zip(CLASSES, cnt)))
    missing = [c for c, n in zip(CLASSES, cnt) if n == 0]
    if missing:
        print(f"  CHÚ Ý: chưa có dữ liệu cho lớp {missing} — model sẽ không nhận ra các lớp này (luật vẫn chạy). "
              "Thêm video vào datasets/raw/custom/<lớp>/ hoặc duyệt sự kiện trên dashboard.")

    dev = resolve_device(device or cfg.behavior_device or cfg.device)
    torch.manual_seed(seed)
    net = build_net(len(CLASSES)).to(dev)
    w = np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0.0)
    w = w / w[w > 0].mean()
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev), label_smoothing=0.05)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    steps = max(1, epochs * int(np.ceil(len(Xtr) / batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    Xva_t = torch.from_numpy(Xva).to(dev)

    def evaluate():
        net.eval()
        with torch.no_grad():
            pr = torch.cat([net(Xva_t[i:i + 512]).argmax(1) for i in range(0, len(Xva_t), 512)]).cpu().numpy()
        net.train()
        return _metrics(Yva, pr)

    best, best_state, t0 = None, None, time.time()
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(np.stack([augment(Xtr[j], rng) for j in idx])).to(dev)
            yb = torch.from_numpy(Ytr[idx]).long().to(dev)
            loss = lossf(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        m = evaluate()
        if best is None or m["macro_f1"] >= best["macro_f1"]:
            best, best_state = m, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  epoch {ep:3d}/{epochs}  loss {tot / len(Xtr):.4f}  val macro-F1 {m['macro_f1']:.3f}  acc {m['accuracy']:.3f}"
                  f"  ({time.time() - t0:.0f}s)")

    out_dir = Path(cfg.behavior_dir) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    meta = {"classes": list(CLASSES), "T": T_STEPS, "span": SPAN_S, "arch": "stgcn-lite-v1", "metrics": best,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
            "train_counts": {c: int(n) for c, n in zip(CLASSES, cnt)}, "clips": [g[0] for g in groups],
            "val_clips": [groups[i][0] for i in sorted(val_idx)]}
    path = out_dir / f"behavior-{stamp}.pt"
    torch.save({"state_dict": best_state, "meta": meta}, path)
    lines = [f"Model hành vi {path.name} — val macro-F1 {best['macro_f1']:.3f}, accuracy {best['accuracy']:.3f}", "",
             f"{'lớp':10s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'số mẫu':>7s}"]
    for c in CLASSES:
        pc = best["per_class"][c]
        lines.append(f"{c:10s} {pc['precision']:9.3f} {pc['recall']:7.3f} {pc['f1']:6.3f} {pc['support']:7d}")
    lines += ["", "Ma trận nhầm lẫn (hàng = thật, cột = dự đoán): " + " ".join(CLASSES)]
    lines += ["  " + " ".join(f"{v:6d}" for v in row) + f"   {CLASSES[i]}" for i, row in enumerate(best["confusion"])]
    report = "\n".join(lines)
    (out_dir / f"behavior-{stamp}.txt").write_text(report + "\n", encoding="utf-8")
    (out_dir / f"behavior-{stamp}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + report)
    if promote:
        shutil.copyfile(path, out_dir / "current.pt")
        print(f"\nĐã đặt làm model đang dùng: {out_dir / 'current.pt'} (khởi động lại ARES để nạp).")
    else:
        print(f"\nĐã lưu {path} (chưa dùng). Muốn dùng: chép thành {out_dir / 'current.pt'} hoặc đặt ARES_BEHAVIOR_CLASSIFIER.")
    return path
