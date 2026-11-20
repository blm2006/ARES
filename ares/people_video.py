"""Thử video cho phần NGƯỜI + MÔI TRƯỜNG: chia vùng, gán đèn (mạch + cổng) vào vùng, mô phỏng bật/tắt đèn theo người.

Dashboard (mục Người & môi trường > Thử video) gửi kèm bố trí:

    {"zones":  [{"id": "z1", "name": "Bàn học", "poly": [[x, y], ...] (0..1 theo khung hình), "entry": false}, ...],
     "lights": [{"id": "l1", "name": "Đèn 1", "zones": ["z1"], "device_id": 3, "output": "relay"|"led", "level": 100}, ...],
     "off_delay_s": 10, "min_people": 1}

Mỗi khung: YOLO tìm người -> bám vết (người mới chỉ sinh ở rìa khung hoặc trong vùng "cửa ra vào") -> điểm giữa chân
-> thuộc vùng nào -> đèn của vùng có người thì BẬT; vùng vắng đủ off_delay_s giây thì TẮT.
Trả kết quả cùng dạng với thử video hành vi (events, status_runs, people_series, khung ảnh) + thống kê đèn/vùng
và light_runs để dashboard điều khiển mạch thật đúng nhịp khi phát lại.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np

from ares.behavior.tracker import Tracker, _in_poly
from ares_server.layout import clean_layout  # noqa: F401 — dùng chung với server (thuần Python)

ZONE_COLS = [(230, 160, 60), (90, 200, 90), (200, 110, 220), (60, 170, 240), (180, 180, 60), (120, 120, 240),
             (60, 200, 200), (240, 120, 150)]  # BGR
LIGHT_ON = (40, 220, 255)  # vàng


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


class LightSim:
    """Trạng thái đèn theo số người trong vùng, có trễ tắt (vắng off_delay_s giây mới tắt)."""

    def __init__(self, layout: dict):
        self.layout = layout
        self.on = {li["id"]: False for li in layout["lights"]}
        self.last_seen = {li["id"]: -1e18 for li in layout["lights"]}
        self.on_s = {li["id"]: 0.0 for li in layout["lights"]}
        self.switches = {li["id"]: 0 for li in layout["lights"]}

    def step(self, t: float, zone_counts: dict[str, int], dt: float) -> list[tuple[dict, bool]]:
        changed = []
        for li in self.layout["lights"]:
            lid = li["id"]
            n = sum(zone_counts.get(z, 0) for z in li["zones"])
            if n >= self.layout["min_people"]:
                self.last_seen[lid] = t
            want = t - self.last_seen[lid] <= self.layout["off_delay_s"]
            if want != self.on[lid]:
                self.on[lid] = want
                self.switches[lid] += 1
                changed.append((li, want))
            if self.on[lid]:
                self.on_s[lid] += dt
        return changed


def render(img: np.ndarray, layout: dict, polys: list, counts: dict, views: list, sim: LightSim) -> np.ndarray:
    """Vùng (tô mờ, vàng nếu đèn của vùng đang bật) + khung người MẢNH + bảng đèn + thanh trạng thái."""
    import cv2

    from ares.behavior.analyzer import _blit, text_patch

    out = img.copy()
    H, W = out.shape[:2]
    fs = max(11, min(15, W // 54))
    lit_zone = {z for li in layout["lights"] if sim.on[li["id"]] for z in li["zones"]}
    over = out.copy()
    for k, (z, pts) in enumerate(zip(layout["zones"], polys)):
        col = LIGHT_ON if z["id"] in lit_zone else ZONE_COLS[k % len(ZONE_COLS)]
        cv2.fillPoly(over, [pts], col)
    cv2.addWeighted(over, 0.22, out, 0.78, 0, out)
    for k, (z, pts) in enumerate(zip(layout["zones"], polys)):
        col = LIGHT_ON if z["id"] in lit_zone else ZONE_COLS[k % len(ZONE_COLS)]
        cv2.polylines(out, [pts], True, col, 1, cv2.LINE_AA)
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        lab = f"{z['name']} · {counts.get(z['id'], 0)} người" + (" · cửa" if z["entry"] else "")
        _blit(out, text_patch(lab, fs - 1, (255, 255, 255), tuple(int(c * 0.6) for c in col), pad=3), x + 2, y + 2, alpha=0.9)
    for v in views:  # người: khung 1px + chấm chân
        x1, y1, x2, y2 = (int(c) for c in v["box"])
        col = (80, 200, 80) if v["zone"] else (170, 170, 170)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 1)
        cv2.circle(out, ((x1 + x2) // 2, y2), 2, (0, 0, 255), -1, cv2.LINE_AA)
        p = text_patch(f"{v['id']}", max(10, fs - 3), (255, 255, 255), col, pad=2)
        _blit(out, p, x1, max(0, y1 - p.shape[0]))
    # bảng đèn góc phải
    y = fs + 16
    for li in layout["lights"]:
        on = sim.on[li["id"]]
        txt = f"{li['name']}: {'BẬT' if on else 'tắt'}" + (f" {li['level']}%" if on and li["output"] == "led" else "")
        p = text_patch(txt, fs - 1, (20, 20, 20) if on else (235, 235, 235), LIGHT_ON if on else (70, 70, 70), pad=3)
        _blit(out, p, W - p.shape[1] - 4, y, alpha=0.95)
        y += p.shape[0] + 3
    n_on = sum(sim.on.values())
    top = fs + 12
    strip = out[:top]
    strip[:] = (strip * 0.35).astype(np.uint8)
    head = f"{len(views)} người" + (f" · đèn bật {n_on}/{len(sim.on)}" if sim.on else "") + \
        "".join(f" · {z['name']}: {counts.get(z['id'], 0)}" for z in layout["zones"][:4])
    _blit(out, text_patch(head, fs, (140, 235, 140) if not n_on else LIGHT_ON, (0, 0, 0), pad=6), 4, 2, alpha=0.999)
    return out


def analyze_people(cfg, pose, path: Path, layout: dict | None, out_mp4: Path | None = None, fps: float = 10.0,
                   frames_zip: Path | None = None, max_w: int = 0, progress=None, should_stop=None) -> dict:
    """pose: PoseEstimator (dùng phần phát hiện người của YOLO-pose) hoặc vật có .infer(list ảnh) cùng dạng."""
    import cv2

    from ares.behavior.datasets import iter_video_frames

    layout = clean_layout(layout)
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    duration = total / src_fps if src_fps and total > 0 else 0.0
    sim = LightSim(layout)
    tracker: Tracker | None = None
    polys: list = []
    events, runs, series = [], [], []
    zone_ps = {z["id"]: 0.0 for z in layout["zones"]}
    zone_max = {z["id"]: 0 for z in layout["zones"]}
    light_runs = {li["id"]: [] for li in layout["lights"]}
    frame_t: list[float] = []  # thời điểm (giây, theo video gốc) của từng khung phân tích -> web phát video gốc mượt + vẽ đè
    tracks: list[list] = []  # từng khung: [[id, x1, y1, x2, y2 (0..1), vùng]]
    n, people_max, sec, sec_max, t0, last_p = 0, 0, -1, 0, time.time(), 0.0
    vw = None
    zf = zipfile.ZipFile(frames_zip, "w", zipfile.ZIP_STORED) if frames_zip is not None else None
    dt = 1.0 / fps
    try:
        for _, t, img in iter_video_frames(path, fps):
            if should_stop is not None and should_stop():
                raise InterruptedError("đã huỷ")
            if max_w and img.shape[1] > max_w:
                img = cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1])), interpolation=cv2.INTER_AREA)
            H, W = img.shape[:2]
            if tracker is None:
                polys = [np.array([[int(x * W), int(y * H)] for x, y in z["poly"]], np.int32) for z in layout["zones"]]
                tracker = Tracker(gate=bool(getattr(cfg, "behavior_track_gate", True)),
                                  warmup_s=float(getattr(cfg, "behavior_track_warmup_s", 2.0)),
                                  edge=float(getattr(cfg, "behavior_track_edge", 0.06)),
                                  confirm_s=float(getattr(cfg, "behavior_track_confirm_s", 4.0)),
                                  recover_s=float(getattr(cfg, "behavior_track_recover_s", 6.0)),
                                  entry_zones=[[tuple(p) for p in pl.tolist()] for z, pl in zip(layout["zones"], polys) if z["entry"]])
            kps, boxes, confs = pose.infer([img])[0]
            tracker.update(t, kps, boxes, confs, (W, H))
            n += 1
            views, counts = [], {z["id"]: 0 for z in layout["zones"]}
            for tr in tracker.tracks.values():
                if tr.last_t != t:
                    continue
                b = tr.last.box
                fx, fy = float(b[0] + b[2]) / 2, float(b[3])
                zone = ""
                for z, pl in zip(layout["zones"], polys):
                    if _in_poly(fx, fy, pl.tolist()):
                        counts[z["id"]] += 1
                        zone = zone or z["id"]
                views.append({"id": tr.id, "box": [float(x) for x in b], "zone": zone})
            if len(frame_t) < 40000:
                zi = {z["id"]: i for i, z in enumerate(layout["zones"])}
                frame_t.append(round(float(t), 3))
                tracks.append([[int(v["id"]), round(v["box"][0] / W, 4), round(v["box"][1] / H, 4), round(v["box"][2] / W, 4),
                                round(v["box"][3] / H, 4), zi.get(v["zone"], -1)] for v in views])
            k = len(views)
            people_max = max(people_max, k)
            for zid, c in counts.items():
                zone_ps[zid] += c * dt
                zone_max[zid] = max(zone_max[zid], c)
            for li, on in sim.step(t, counts, dt):
                light_runs[li["id"]].append([n, 1 if on else 0])
                who = sum(counts.get(z, 0) for z in li["zones"])
                names = ", ".join(z["name"] for z in layout["zones"] if z["id"] in li["zones"]) or "chưa gán vùng"
                events.append({"t": round(t, 2), "time": _mmss(t), "kind": "LIGHT_ON" if on else "LIGHT_OFF",
                               "label": f"{'Bật' if on else 'Tắt'} {li['name']}", "tracks": [], "source": names,
                               "detail": {"light": li["id"], "people": who,
                                          "text": f"{who} người trong {names}" if on else
                                          f"{names} vắng {layout['off_delay_s']:g} giây"}, "frame": n})
            n_on = sum(sim.on.values())
            lv = 1 if n_on else 0
            txt = (f"{k} người · đèn bật {n_on}/{len(sim.on)}" if sim.on else f"{k} người") + \
                "".join(f" · {z['name']}: {counts[z['id']]}" for z in layout["zones"])
            if (not runs or runs[-1][1] != lv or runs[-1][2] != txt) and len(runs) < 5000:
                runs.append([n, lv, txt])
            if int(t) != sec:
                if sec >= 0:
                    series.append([sec, sec_max])
                sec, sec_max = int(t), k
            else:
                sec_max = max(sec_max, k)
            if out_mp4 is not None or zf is not None:
                im = render(img, layout, polys, counts, views, sim)
                if out_mp4 is not None:
                    if vw is None:
                        for fourcc in ("avc1", "mp4v"):
                            vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*fourcc), fps, (im.shape[1], im.shape[0]))
                            if vw.isOpened():
                                break
                    vw.write(im)
                if zf is not None:
                    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if ok:
                        zf.writestr(f"{n:04d}.jpg", buf.tobytes())
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
    dur = max(duration, n / fps)
    lights_stat = {li["id"]: {"name": li["name"], "on_s": round(min(dur, sim.on_s[li["id"]]), 1),
                              "pct": round(min(100.0, 100 * sim.on_s[li["id"]] / dur), 1) if dur else 0.0,
                              "switches": sim.switches[li["id"]]} for li in layout["lights"]}
    saved = round(100 - sum(v["pct"] for v in lights_stat.values()) / len(lights_stat), 1) if lights_stat else None
    return {"kind": "people", "events": events[:500], "frames": n, "fps": fps, "duration": round(dur, 2),
            "people_max": people_max, "people_series": series, "processing_s": round(time.time() - t0, 1),
            "status_runs": runs, "layout": layout, "light_runs": light_runs, "lights_stat": lights_stat,
            "zones_stat": {z["id"]: {"name": z["name"], "person_s": round(zone_ps[z["id"]], 1), "max": zone_max[z["id"]]}
                           for z in layout["zones"]},
            "energy_saved_pct": saved, "frame_t": frame_t, "tracks": tracks, "track_stats": dict(tracker.stats) if tracker else {}}
