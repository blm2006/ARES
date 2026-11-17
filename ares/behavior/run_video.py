"""Thử nhận diện hành vi trên file video (không cần chạy ARES).

    python -m ares.behavior video samples\\te_nga.mp4
    python -m ares.behavior video samples\\ --still-s 10 --debug       (cả thư mục, in trạng thái từng giây để chỉnh ngưỡng)

Ghi ra <storage_root>/behavior/video_tests/<tên>_behavior.mp4 (khung xương + nhãn) và <tên>_events.json.
Dùng đúng lõi Analyzer như khi chạy trong ARES, thời gian lấy theo video (không theo đồng hồ máy).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ares.behavior import KIND_VI
from ares.behavior.datasets import VIDEO_EXT, iter_video_frames


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def load_models(cfg):
    """Nạp YOLO-pose (+ model hành vi đã train nếu có) — dùng chung cho thử video bằng lệnh và job từ dashboard."""
    from ares.behavior.monitor import resolve_path
    from ares.behavior.pose import PoseEstimator, resolve_device
    from ares.config import REPO_ROOT

    device = resolve_device(cfg.behavior_device or cfg.device)
    pose = PoseEstimator(resolve_path(cfg.behavior_weights, REPO_ROOT), device, cfg.behavior_imgsz, cfg.behavior_conf, cfg.half)
    clf, cpath = None, None
    cpath = resolve_path(cfg.behavior_classifier, REPO_ROOT) if cfg.behavior_classifier else Path(cfg.behavior_dir) / "models" / "current.pt"
    if cfg.behavior_mode != "rules" and cpath.exists():
        from ares.behavior.model import load_classifier

        net, meta = load_classifier(cpath, pose.device)
        clf = (net, meta, pose.device)
    return pose, clf, cpath


def analyze(cfg, pose, clf, path: Path, out_mp4: Path | None = None, fps: float = 10.0, frames_zip: Path | None = None,
            max_w: int = 0, progress=None, should_stop=None, debug: bool = False, ppe=None, ppe_rules=None) -> dict:
    """Chạy Analyzer trên 1 video. Trả {events, frames, fps, duration, people_max, people_series, processing_s,
    status_runs, posture_s}.

    out_mp4: ghi video kết quả · frames_zip: ghi từng khung JPEG (0001.jpg...) để trình duyệt phát được ở mọi máy
    progress(tỉ lệ 0..1, số khung): gọi ~mỗi giây · should_stop(): True thì dừng giữa chừng (raise InterruptedError).
    """
    import zipfile

    import cv2

    from ares.behavior.analyzer import POSE_KEYS, Analyzer, summarize

    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    duration = total / src_fps if src_fps and total > 0 else 0.0
    an = Analyzer(cfg, pose, clf, ppe)  # ppe: PPEDetector hoặc None; ppe_rules: luật đồ bảo hộ riêng cho video này
    if ppe_rules is not None:
        from ares.ppe import clean_rules

        an.ppe_rules["video"] = clean_rules(ppe_rules)
    draw = out_mp4 is not None or frames_zip is not None
    vw, zf, events, n, t0, last_dbg, last_p = None, None, [], 0, time.time(), -1.0, 0.0
    series, people_max, sec, sec_max = [], 0, -1, 0
    runs: list = []  # [khung bắt đầu, mức 0/1/2, câu trạng thái] — chỉ ghi khi trạng thái đổi (thanh trạng thái + dải thời gian)
    pose_s = {k: 0.0 for k in POSE_KEYS}  # tổng thời gian (người × giây) ở từng tư thế
    if frames_zip is not None:
        zf = zipfile.ZipFile(frames_zip, "w", zipfile.ZIP_STORED)
    try:
        for _, t, img in iter_video_frames(path, fps):
            if should_stop is not None and should_stop():
                raise InterruptedError("đã huỷ")
            if max_w and img.shape[1] > max_w:
                img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])), interpolation=cv2.INTER_AREA)
            res = an.process([("video", img, t)], draw=draw)[0]
            n += 1
            k = len(res.tracks)
            people_max = max(people_max, k)
            lv, txt = summarize(res.tracks)
            if (not runs or runs[-1][1] != lv or runs[-1][2] != txt) and len(runs) < 5000:
                runs.append([n, lv, txt])
            for v in res.tracks:
                if v.pose in pose_s:
                    pose_s[v.pose] += 1.0 / fps
            if int(t) != sec:
                if sec >= 0:
                    series.append([sec, sec_max])
                sec, sec_max = int(t), k
            else:
                sec_max = max(sec_max, k)
            if draw and res.image is not None:
                if out_mp4 is not None:
                    if vw is None:
                        h, w = res.image.shape[:2]
                        for fourcc in ("avc1", "mp4v"):  # avc1 (H.264) phát được trên trình duyệt; không có thì mp4v
                            vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                            if vw.isOpened():
                                break
                    vw.write(res.image)
                if zf is not None:
                    ok, buf = cv2.imencode(".jpg", res.image, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if ok:
                        zf.writestr(f"{n:04d}.jpg", buf.tobytes())
            for ev in res.events:
                d = {"video": Path(path).name, "t": round(ev.t, 2), "time": _mmss(ev.t), "kind": ev.kind, "label": KIND_VI[ev.kind],
                     "tracks": ev.track_ids, "source": ev.source, "detail": ev.detail, "frame": n}
                events.append(d)
                if progress is None:
                    print(f"  [{_mmss(ev.t)}] {KIND_VI[ev.kind]:24s} người {ev.track_ids} ({ev.source}) {ev.detail or ''}")
            if debug and t - last_dbg >= 1.0:
                last_dbg = t
                for v in res.tracks:
                    pr = "" if not v.probs else " model " + max(v.probs, key=v.probs.get)
                    print(f"    {_mmss(t)} #{v.id:<3d} tư thế {v.posture:8s} chuyển động {v.motion if v.motion is not None else '—':<6} "
                          f"bất động {v.still_s:5.1f}s {v.text}{pr}")
            if progress is not None and time.time() - last_p >= 1.0:
                last_p = time.time()
                progress(min(0.99, t / duration) if duration else 0.0, n)
        if sec >= 0:
            series.append([sec, sec_max])
    finally:
        if vw is not None:
            vw.release()
        if zf is not None:
            zf.close()
    return {"events": events, "frames": n, "fps": fps, "duration": round(duration or n / fps, 2), "people_max": people_max,
            "people_series": series, "processing_s": round(time.time() - t0, 1),
            "status_runs": runs, "posture_s": {k: round(v, 1) for k, v in pose_s.items()},
            "ppe": {"rules": an.rules_for("video"), "ready": ppe is not None}}


def run(cfg, path: Path, out_dir: Path | None = None, fps: float = 10.0, debug: bool = False, write: bool = True) -> list[dict]:
    paths = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in VIDEO_EXT) if Path(path).is_dir() else [Path(path)]
    if not paths:
        print(f"Không thấy video trong {path}")
        return []
    pose, clf, cpath = load_models(cfg)
    ppe = None
    if cfg.ppe_required or cfg.ppe_forbidden:
        from ares.ppe.load import load_detector

        ppe, info = load_detector(cfg)
        print("AI bảo hộ: " + (f"{info['weights']} — nhận ra {', '.join(info['items'])}" if ppe else
                               (info.get("error") or info.get("hint", "chưa cài"))))
    print(f"Pose {Path(pose.weights).name} trên {pose.device} · chế độ {cfg.behavior_mode}"
          f"{' + model ' + cpath.name if clf else ' (chỉ luật)'} · bất động >= {cfg.behavior_still_s:g}s · {fps:g} khung/giây")
    out_dir = Path(out_dir or Path(cfg.behavior_dir) / "video_tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_events = []
    for p in paths:
        out_mp4 = out_dir / f"{p.stem}_behavior.mp4"
        r = analyze(cfg, pose, clf, p, out_mp4 if write else None, fps, debug=debug, ppe=ppe)
        dt, n, events = r["processing_s"], r["frames"], r["events"]
        print(f"{p.name}: {n} khung trong {dt:.1f}s ({n / dt if dt else 0:.1f} khung/giây xử lý) — {len(events)} sự kiện"
              + (f" -> {out_mp4}" if write else ""))
        (out_dir / f"{p.stem}_events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
        all_events += events
    return all_events
