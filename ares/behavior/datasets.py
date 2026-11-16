"""Tải dataset công khai + tách khung xương để train model hành vi.

    python -m ares.behavior download              # URFD (30 ngã + 40 sinh hoạt) + Surveillance Camera Fight
    python -m ares.behavior download --urfd-max 5 # tải ít để thử nhanh
    python -m ares.behavior extract               # chạy YOLO-pose trên mọi video đã tải -> keypoints/*.npz

Thư mục (mặc định <storage_root>/behavior/datasets):
    raw/urfd/            fall-XX-cam0-rgb.zip, adl-XX-cam0-rgb.zip, urfall-cam0-falls.csv, urfall-cam0-adls.csv
    raw/fight-surv/      fight/*.mp4, noFight/*.mp4
    raw/custom/<lớp>/    VIDEO CỦA BẠN, cả clip mang 1 nhãn: normal | fall | lying | slumped | fight
                         (ví dụ raw/custom/slumped/nhom_quay_01.mp4). Nên cắt clip ngắn 2–10 giây đúng hành vi.
    keypoints/           kết quả tách khung xương (.npz), train.py đọc ở đây

Nguồn:
  * UR Fall Detection Dataset — Kwolek & Kepski, Univ. of Rzeszow: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
    Nhãn từng khung trong urfall-cam0-*.csv: -1 không nằm, 0 đang ngã, 1 nằm trên sàn. Camera cam0 song song sàn, 30 fps.
  * Surveillance Camera Fight Dataset — Aktı et al. 2019: https://github.com/sayibet/fight-detection-surv-dataset
    150 clip đánh nhau + 150 clip bình thường, ~2 giây, từ camera giám sát.
  * Le2i (Univ. Bourgogne) cần đăng ký/tải tay: giải nén video vào raw/custom/fall hoặc raw/custom/normal tuỳ đoạn.
Dataset dùng cho học tập/nghiên cứu — kiểm tra giấy phép của từng bộ trước khi dùng thương mại.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import shutil
import ssl
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from ares.behavior import CLASSES

log = logging.getLogger(__name__)

URFD_BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
FIGHT_ZIPS = (
    "https://github.com/sayibet/fight-detection-surv-dataset/archive/refs/heads/master.zip",
    "https://github.com/sayibet/fight-detection-surv-dataset/archive/refs/heads/main.zip",
)
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv")
MAX_TRACKS = 4


def dirs(cfg) -> tuple[Path, Path]:
    base = Path(cfg.behavior_dir) / "datasets"
    raw, kp = base / "raw", base / "keypoints"
    for c in CLASSES:
        (raw / "custom" / c).mkdir(parents=True, exist_ok=True)
    kp.mkdir(parents=True, exist_ok=True)
    return raw, kp


# ============================================================================ tải
def _fetch(url: str, dst: Path, retries: int = 3) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  đã có  {dst.name}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ARES-behavior/1.0"})
            try:
                resp = urllib.request.urlopen(req, timeout=60)
            except (ssl.SSLError, urllib.error.URLError) as e:
                if url.startswith("https://") and "SSL" in str(e).upper():
                    resp = urllib.request.urlopen(urllib.request.Request(url.replace("https://", "http://", 1),
                                                                         headers=req.headers), timeout=60)
                else:
                    raise
            total = int(resp.headers.get("Content-Length") or 0)
            got, t0 = 0, time.time()
            with open(part, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        sys.stdout.write(f"\r  tải    {dst.name}: {got / 1e6:6.1f}/{total / 1e6:.1f} MB")
                    else:
                        sys.stdout.write(f"\r  tải    {dst.name}: {got / 1e6:6.1f} MB")
                    sys.stdout.flush()
            sys.stdout.write(f"  ({time.time() - t0:.0f}s)\n")
            part.replace(dst)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"\n  lỗi    {dst.name} (lần {attempt}/{retries}): {e}")
            time.sleep(2 * attempt)
    return False


def download(cfg, urfd: bool = True, urfd_max: int = 0, fight: bool = True) -> None:
    raw, _ = dirs(cfg)
    if urfd:
        d = raw / "urfd"
        print(f"UR Fall Detection -> {d}  (mỗi file zip ~20–100 MB)")
        for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
            _fetch(URFD_BASE + name, d / name)
        n_f = min(30, urfd_max) if urfd_max else 30
        n_a = min(40, urfd_max) if urfd_max else 40
        for i in range(1, n_f + 1):
            _fetch(f"{URFD_BASE}fall-{i:02d}-cam0-rgb.zip", d / f"fall-{i:02d}-cam0-rgb.zip")
        for i in range(1, n_a + 1):
            _fetch(f"{URFD_BASE}adl-{i:02d}-cam0-rgb.zip", d / f"adl-{i:02d}-cam0-rgb.zip")
    if fight:
        d = raw / "fight-surv"
        if any(d.rglob("*.mp4")):
            print(f"Fight dataset đã có ở {d}")
        else:
            print(f"Surveillance Camera Fight -> {d}")
            z = raw / "_fight.zip"
            if any(_fetch(u, z) for u in FIGHT_ZIPS):
                with zipfile.ZipFile(z) as zf:
                    for m in zf.namelist():
                        low = m.lower()
                        if not low.endswith(VIDEO_EXT):
                            continue
                        parent = Path(m).parent.name.lower()  # .../fight/fi001.mp4 hoặc .../noFight/nofi001.mp4
                        cls = {"nofight": "noFight", "fight": "fight"}.get(parent)
                        if cls is None:
                            continue
                        out = d / cls / Path(m).name
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(m) as src, open(out, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                z.unlink(missing_ok=True)
                print(f"  giải nén xong: {len(list((d / 'fight').glob('*')))} fight, {len(list((d / 'noFight').glob('*')))} noFight")
            else:
                print("  Không tải được — tải tay từ GitHub rồi giải nén vào", d)
    print("Xong. Bước tiếp: python -m ares.behavior extract")


# ============================================================================ đọc khung hình
def _num_key(name: str) -> int:
    nums = re.findall(r"\d+", Path(name).stem)
    return int(nums[-1]) if nums else 0


def iter_zip_frames(path: Path, src_fps: float, target_fps: float):
    import cv2

    with zipfile.ZipFile(path) as zf:
        names = sorted((n for n in zf.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))), key=_num_key)
        step_t, next_t = 1.0 / target_fps, 0.0
        for i, n in enumerate(names):
            t = i / src_fps
            if t + 1e-6 < next_t:
                continue
            next_t += step_t
            img = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield i + 1, t, img  # số thứ tự khung (1-based) khớp cột frame của csv URFD


def iter_video_frames(path: Path, target_fps: float):
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not (1.0 <= fps <= 240.0):
        fps = 25.0
    i, next_t, step_t = 0, 0.0, 1.0 / target_fps
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if t + 1e-6 < next_t:
                continue
            next_t += step_t
            yield i, t, img
    finally:
        cap.release()


def _urfd_labels(raw: Path) -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {}
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        p = raw / "urfd" / name
        if not p.exists():
            continue
        for row in csv.reader(io.StringIO(p.read_text(encoding="utf-8", errors="ignore"))):
            if len(row) < 3:
                continue
            try:
                out.setdefault(row[0].strip(), {})[int(float(row[1]))] = int(float(row[2]))
            except ValueError:
                continue  # dòng tiêu đề
    return out


URFD_MAP = {-1: "normal", 0: "fall", 1: "lying"}


def list_sources(raw: Path) -> list[dict]:
    """Mọi clip có thể tách: {name, kind: zip|video, path, label (cả clip) | None, urfd_seq}."""
    out = []
    for z in sorted((raw / "urfd").glob("*-cam0-rgb.zip")):
        seq = z.name.replace("-cam0-rgb.zip", "")
        out.append({"name": f"urfd__{seq}", "kind": "zip", "path": z, "label": None, "urfd_seq": seq})
    for cls, lab in (("fight", "fight"), ("noFight", "normal")):
        for v in sorted((raw / "fight-surv" / cls).glob("*")):
            if v.suffix.lower() in VIDEO_EXT:
                out.append({"name": f"fight__{cls}__{v.stem}", "kind": "video", "path": v, "label": lab})
    for cls in CLASSES:
        for v in sorted((raw / "custom" / cls).rglob("*")):
            if v.suffix.lower() in VIDEO_EXT:
                out.append({"name": f"custom__{cls}__{v.stem}", "kind": "video", "path": v, "label": cls})
    return out


# ============================================================================ tách khung xương
def extract_clip(pose, frames, labels_fn, batch: int = 16) -> dict | None:
    """frames: iterator (frame_no, t, img). Trả mảng đã bám vết, sẵn để lưu npz."""
    from ares.behavior.tracker import Tracker

    tracker = Tracker(keep_s=1.0, keep_lying_s=4.0)
    times, labels, per_frame = [], [], []
    buf = []

    def flush():
        res = pose.infer([b[2] for b in buf])
        for (fno, t, _), (kps, boxes, confs) in zip(buf, res):
            trs = tracker.update(t, kps, boxes, confs)
            per_frame.append({tr.id: (tr.last.kp, tr.last.box) for tr in trs})
            times.append(t)
            labels.append(labels_fn(fno))
        buf.clear()

    for item in frames:
        buf.append(item)
        if len(buf) >= batch:
            flush()
    if buf:
        flush()
    if not times:
        return None
    hits: dict[int, int] = {}
    for fr in per_frame:
        for tid in fr:
            hits[tid] = hits.get(tid, 0) + 1
    keep = sorted(hits, key=lambda k: -hits[k])[:MAX_TRACKS]
    F, P = len(times), max(1, len(keep))
    kp = np.full((F, P, 17, 3), np.nan, np.float32)
    box = np.full((F, P, 4), np.nan, np.float32)
    for f, fr in enumerate(per_frame):
        for p, tid in enumerate(keep):
            if tid in fr:
                kp[f, p], box[f, p] = fr[tid]
    return {"times": np.asarray(times, np.float64), "labels": np.asarray(labels, np.int16), "kp": kp, "box": box}


def extract(cfg, force: bool = False, fps: float = 10.0, only: str = "") -> None:
    from ares.behavior.monitor import resolve_path
    from ares.behavior.pose import PoseEstimator
    from ares.config import REPO_ROOT

    raw, kpdir = dirs(cfg)
    sources = [s for s in list_sources(raw) if not only or only in s["name"]]
    if not sources:
        print(f"Không thấy video nào trong {raw}. Chạy 'python -m ares.behavior download' hoặc chép video vào raw/custom/<lớp>/")
        return
    pose = PoseEstimator(resolve_path(cfg.behavior_weights, REPO_ROOT), cfg.behavior_device or cfg.device,
                         cfg.behavior_imgsz, cfg.behavior_conf, cfg.half)
    urfd = _urfd_labels(raw)
    print(f"Tách khung xương {len(sources)} clip bằng {Path(pose.weights).name} trên {pose.device}, {fps:g} khung/giây -> {kpdir}")
    t_all = time.time()
    for k, s in enumerate(sources, start=1):
        out = kpdir / f"{s['name']}.npz"
        if out.exists() and not force:
            continue
        t0 = time.time()
        if s["kind"] == "zip":
            lab = urfd.get(s["urfd_seq"], {})
            if not lab:
                print(f"  [{k}/{len(sources)}] {s['name']}: thiếu nhãn trong csv URFD — bỏ qua")
                continue
            idx = {c: i for i, c in enumerate(CLASSES)}
            frames = iter_zip_frames(s["path"], 30.0, fps)
            labels_fn = (lambda fno, lab=lab: idx[URFD_MAP.get(lab.get(fno, -1), "normal")])
        else:
            frames = iter_video_frames(s["path"], fps)
            ci = CLASSES.index(s["label"])
            labels_fn = (lambda fno, ci=ci: ci)
        try:
            arr = extract_clip(pose, frames, labels_fn)
        except Exception as e:  # noqa: BLE001
            print(f"  [{k}/{len(sources)}] {s['name']}: lỗi {e}")
            continue
        if arr is None:
            print(f"  [{k}/{len(sources)}] {s['name']}: không đọc được khung hình")
            continue
        np.savez_compressed(out, src=np.array(s["name"]), **arr)
        seen = int(np.isfinite(arr["kp"][:, 0, 0, 0]).sum())
        print(f"  [{k}/{len(sources)}] {s['name']}: {len(arr['times'])} khung, thấy người {seen}, {time.time() - t0:.1f}s")
    print(f"Xong sau {time.time() - t_all:.0f}s. Bước tiếp: python -m ares.behavior train")
