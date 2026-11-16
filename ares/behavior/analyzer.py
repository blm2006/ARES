"""Lõi phân tích hành vi: pose -> bám vết -> luật (+ model) -> sự kiện. Không có luồng, không ghi đĩa.

Dùng chung cho chạy thật trong ARES (monitor.py) và thử trên file video (run_video.py), nên thời gian luôn được
truyền vào (không đọc đồng hồ hệ thống).

Luật (mọi ngưỡng ở config.py, tiền tố ARES_BEHAVIOR_):
  * TÉ NGÃ       : đang đứng/ngồi -> nằm trong <= fall_window_s, hông tụt >= fall_drop x chiều cao người,
                   rồi vẫn nằm thêm fall_confirm_s (lọc cúi nhặt đồ / ngồi xuống nhanh).
  * NẰM XUỐNG     : nằm liên tục lying_s giây (mặc định 5 s) -> báo NGAY CẢ KHI VẪN CỬ ĐỘNG; nếu vừa đứng/ngồi
                   rồi nằm xuống đột ngột (trong sudden_window_s) thì chỉ cần sudden_lying_s (mặc định 2 s).
                   Không báo trùng nếu đã báo TÉ NGÃ cho người đó.
  * NẰM BẤT ĐỘNG : tư thế nằm + cử động < still_motion liên tục still_s giây (mặc định 15 s). Cử động đo trên các
                   khớp cử động nhiều nhất (pose_motion) — cử động tay/đầu là đếm lại từ đầu.
  * NGỒI GỤC     : ngồi/cúi, đầu thấp ngang vai trở xuống + bất động still_s giây.
  * ĐÁNH NHAU    : 2 người sát nhau (< fight_dist x chiều cao) và có tay/chân vung nhanh (> fight_speed)
                   tích luỹ >= fight_s giây.
Model ST-GCN-lite (nếu đã train) bổ sung tín hiệu: hybrid = luật HOẶC model; model = chỉ model.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from statistics import median

import numpy as np

from ares.behavior import CLASSES, KIND_VI
from ares.behavior.skeleton import EDGES, KP_CONF, LIMBS, frame_features, pose_motion, visible_mean
from ares.behavior.tracker import Track, Tracker

log = logging.getLogger(__name__)

DOWN = ("lying", "slumped")
EVAL_EVERY_S = 0.5  # model phân loại chạy lại mỗi chừng này giây cho mỗi người
PROB_FRESH_S = 1.2


@dataclass
class BehaviorEvent:
    kind: str
    cam_id: str
    t: float
    track_ids: list[int]
    score: float
    center: tuple[float, float]
    source: str  # rules | model | rules+model
    detail: dict = field(default_factory=dict)

    @property
    def label_vi(self) -> str:
        return KIND_VI.get(self.kind, self.kind)


@dataclass
class TrackView:
    id: int
    box: list[float]
    posture: str
    label: str
    level: int  # 0 bình thường, 1 theo dõi, 2 báo động
    still_s: float
    motion: float | None
    probs: dict | None


@dataclass
class CamResult:
    cam_id: str
    t: float
    tracks: list[TrackView]
    events: list[BehaviorEvent]
    image: np.ndarray | None  # ảnh đã vẽ (thu nhỏ về clip_width) — None nếu không yêu cầu vẽ


class CameraAnalyzer:
    """Trạng thái của MỘT camera."""

    def __init__(self, cam_id: str, cfg):
        self.cam_id, self.cfg = cam_id, cfg
        self.tracker = Tracker()
        self.pairs: dict[tuple[int, int], dict] = {}
        self.fired: deque = deque(maxlen=200)  # (kind, t, center, ref_h, track_id)
        self.last_t = 0.0

    # -------------------------------------------------------------- tiện ích
    def _cooldown_ok(self, kind: str, t: float, tr_ids: list[int], center, ref_h: float) -> bool:
        cd = self.cfg.behavior_cooldown_s
        for k, ft, fc, fh, fid in self.fired:
            if k == "FIGHT" == kind and t - ft <= self.cfg.behavior_fight_cam_cooldown_s:
                return False  # cùng một vụ ẩu đả: nhiều cặp người, chỉ báo một lần cho cả camera
            if k != kind or t - ft > cd:
                continue
            if fid in tr_ids:
                return False
            if np.hypot(center[0] - fc[0], center[1] - fc[1]) < 0.8 * max(fh, ref_h):  # cùng chỗ (track bị đổi id)
                return False
        return True

    def _fire(self, out: list, kind: str, t: float, trs: list[Track], score: float, source: str, **detail) -> None:
        cs = [((tr.last.box[0] + tr.last.box[2]) / 2, (tr.last.box[1] + tr.last.box[3]) / 2) for tr in trs]
        center = (float(np.mean([c[0] for c in cs])), float(np.mean([c[1] for c in cs])))
        rh = float(np.mean([tr.ref_h for tr in trs]))
        ids = [tr.id for tr in trs]
        if not self._cooldown_ok(kind, t, ids, center, rh):
            return
        for tr in trs:
            self.fired.append((kind, t, center, rh, tr.id))
        out.append(BehaviorEvent(kind, self.cam_id, t, ids, round(float(score), 3), center, source, detail))

    # -------------------------------------------------------------- mỗi người
    def _update_track(self, tr: Track, t: float, mode: str) -> None:
        cfg, st = self.cfg, tr.state
        s = tr.last
        ref_h = tr.ref_h
        ff = frame_features(s.kp, s.box, ref_h)
        posture = ff.posture
        probs = st.get("probs") if t - st.get("probs_t", -1e9) <= PROB_FRESH_S else None
        if probs is not None and mode in ("hybrid", "model"):
            thr = cfg.behavior_model_thr
            if probs["lying"] >= thr:
                posture = "lying"
            elif probs["slumped"] >= thr:
                posture = "slumped"
            elif mode == "model":
                posture = "upright" if probs["normal"] >= 0.5 else posture
        feats: deque = st.setdefault("feats", deque())
        hip = ff.hip_y if ff.hip_y is not None else ff.center[1]
        feats.append((t, posture, hip))
        while feats and t - feats[0][0] > 6.0:
            feats.popleft()
        recent = [p for tt, p, _ in feats if t - tt <= 1.0]
        hold = max(set(recent), key=recent.count) if recent else posture
        st["posture"], st["posture_hold"] = posture, hold

        # ---- chuyển động: so tư thế trung bình 0,3 s gần nhất với 0,3 s cách đây ~0,85 s (khử rung của model)
        a = tr.window(t - 1.0, t - 0.7)
        b = tr.window(t - 0.3, t)
        motion = None
        if len(a) >= 1 and len(b) >= 1:
            sh = pose_motion(visible_mean(np.stack([x.kp for x in a])), visible_mean(np.stack([x.kp for x in b])), ref_h)
            if sh is not None:
                dt = max(0.3, np.mean([x.t for x in b]) - np.mean([x.t for x in a]))
                motion = sh / dt
        if motion is not None:
            m0 = st.get("motion")
            st["motion"] = motion if m0 is None else 0.5 * m0 + 0.5 * motion
            st["motion_t"] = t
        elif t - st.get("motion_t", -1e9) > 1.5:
            st["motion"] = None  # không đo được (khuất khớp) -> không đoán là bất động
        # ---- tốc độ tay/chân (cho đánh nhau)
        if len(tr.hist) >= 2:
            p = tr.hist[-2]
            dt = s.t - p.t
            if 0 < dt <= 0.6:
                sp = [float(np.hypot(*(s.kp[j, :2] - p.kp[j, :2]))) / ref_h / dt
                      for j in LIMBS if s.kp[j, 2] >= KP_CONF and p.kp[j, 2] >= KP_CONF]
                lim: deque = st.setdefault("limb", deque(maxlen=3))
                lim.append(max(sp) if sp else 0.0)
                st["limb_speed"] = median(lim)

    def _rules_track(self, tr: Track, t: float, mode: str, out: list) -> None:
        cfg, st = self.cfg, tr.state
        ref_h = tr.ref_h
        posture, hold = st["posture"], st["posture_hold"]
        feats = st["feats"]
        probs = st.get("probs") if t - st.get("probs_t", -1e9) <= PROB_FRESH_S else None
        use_rules = mode in ("rules", "hybrid") or (mode == "model" and not st.get("_model_ok"))  # chưa có model: tạm dùng luật

        # ---------------- TÉ NGÃ
        if st.get("fall_pending") is None:
            onset_src = None
            if use_rules and posture == "lying":
                earlier = [(tt, p, hy) for tt, p, hy in feats if t - cfg.behavior_fall_window_s <= tt < t]
                ups = [(tt, hy) for tt, p, hy in earlier if p in ("upright", "sitting")]
                if ups:
                    hy_now = feats[-1][2]
                    drop = max((hy_now - hy) for _, hy in ups) / ref_h
                    if drop >= cfg.behavior_fall_drop:
                        onset_src, st["fall_drop"] = "rules", round(drop, 2)
            if probs is not None and mode in ("hybrid", "model") and probs["fall"] >= cfg.behavior_model_thr:
                st["fall_hits"] = st.get("fall_hits", 0) + (1 if st.get("fall_hit_t") != st.get("probs_t") else 0)
                st["fall_hit_t"] = st.get("probs_t")
                if st["fall_hits"] >= 2:
                    onset_src = "rules+model" if onset_src else "model"
            elif probs is not None:
                st["fall_hits"] = 0
            if onset_src:
                st["fall_pending"], st["fall_src"] = t, onset_src
        else:
            t0 = st["fall_pending"]
            if t - t0 >= cfg.behavior_fall_confirm_s:
                seg = [p for tt, p, _ in feats if tt >= t0]
                frac = sum(1 for p in seg if p == "lying") / max(1, len(seg))
                if frac >= 0.6:
                    st["fell_at"] = t
                    score = probs["fall"] if probs else min(1.0, 0.5 + st.get("fall_drop", 0.4))
                    self._fire(out, "FALL", t, [tr], score, st.get("fall_src", "rules"),
                               drop=st.get("fall_drop"))
                st["fall_pending"] = None
                st["fall_hits"] = 0

        # ---------------- NẰM XUỐNG (đột ngột / nằm dưới sàn) — báo dù vẫn cử động
        thr = cfg.behavior_still_motion
        motion = st.get("motion")
        if hold == "lying":
            st["lie_last"] = t
            if st.get("lie_since") is None:
                st["lie_since"] = t
                st["lie_sudden"] = any(p in ("upright", "sitting") and t - tt <= cfg.behavior_sudden_window_s + 1.0
                                       for tt, p, _ in feats)  # +1 s: tư thế "giữ" trễ ~0,5-1 s so với khung đầu tiên nằm
            st["lying_s"] = t - st["lie_since"]
            need = cfg.behavior_sudden_lying_s if st.get("lie_sudden") else cfg.behavior_lying_s
            if cfg.behavior_lying_s > 0 and not st.get("lie_fired") and st["lying_s"] >= need:
                st["lie_fired"] = True
                if t - st.get("fell_at", -1e9) > 15.0:  # đã báo TÉ NGÃ cho người này thì không báo trùng
                    src = "model" if (probs is not None and probs.get("lying", 0) >= cfg.behavior_model_thr) else "rules"
                    self._fire(out, "LYING_DOWN", t, [tr], 0.8 if st.get("lie_sudden") else 0.6, src,
                               lying_s=round(st["lying_s"], 1), sudden=bool(st.get("lie_sudden")),
                               moving=bool(motion is not None and motion >= thr))
        elif t - st.get("lie_last", -1e9) > 1.5:
            st["lie_since"], st["lie_fired"], st["lying_s"] = None, False, 0.0

        # ---------------- BẤT ĐỘNG (nằm / ngồi gục): chỉ đếm khi ĐO ĐƯỢC cử động và cử động nhỏ; cử động rõ là đếm lại
        if hold in DOWN:
            st["down_last"] = t
            if motion is None or motion > 1.5 * thr:
                st["still_since"] = None
            elif st.get("still_since") is None and motion < thr:
                st["still_since"] = t
        elif t - st.get("down_last", -1e9) > 2.0:
            st["still_since"] = None
            st["still_fired"] = False
        ss = st.get("still_since")
        st["still_s"] = 0.0 if ss is None else t - ss
        if ss is not None and st["still_s"] >= cfg.behavior_still_s and not st.get("still_fired"):
            st["still_fired"] = True
            kind = "LYING_STILL" if hold == "lying" else "SLUMPED"
            after_fall = t - st.get("fell_at", -1e9) <= cfg.behavior_still_s + 60
            src = "model" if (probs is not None and probs.get(hold, 0) >= cfg.behavior_model_thr) else "rules"
            self._fire(out, kind, t, [tr], 1.0, src, still_s=round(st["still_s"], 1), after_fall=after_fall)

    def _rules_fight(self, live: list[Track], t: float, mode: str, out: list) -> None:
        cfg = self.cfg
        seen = set()
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                key = (min(a.id, b.id), max(a.id, b.id))
                seen.add(key)
                ps = self.pairs.setdefault(key, {"acc": 0.0, "t": t})
                dt = min(0.5, max(0.0, t - ps["t"]))
                ps["t"] = t
                rh = (a.ref_h + b.ref_h) / 2
                ca = ((a.last.box[0] + a.last.box[2]) / 2, (a.last.box[1] + a.last.box[3]) / 2)
                cb = ((b.last.box[0] + b.last.box[2]) / 2, (b.last.box[1] + b.last.box[3]) / 2)
                close = np.hypot(ca[0] - cb[0], ca[1] - cb[1]) < cfg.behavior_fight_dist * rh
                fast = max(a.state.get("limb_speed", 0.0), b.state.get("limb_speed", 0.0)) > cfg.behavior_fight_speed
                rule_hit = close and fast and mode in ("rules", "hybrid")
                model_hit = False
                if mode in ("hybrid", "model"):
                    for tr in (a, b):
                        pr = tr.state.get("probs") if t - tr.state.get("probs_t", -1e9) <= PROB_FRESH_S else None
                        if pr is not None and pr["fight"] >= cfg.behavior_model_thr and close:
                            model_hit = True
                if mode == "model" and not (a.state.get("_model_ok") or b.state.get("_model_ok")):
                    rule_hit = close and fast  # chưa có model: tạm dùng luật
                if rule_hit or model_hit:
                    ps["acc"] += dt
                    ps["src"] = "rules+model" if rule_hit and model_hit else ("model" if model_hit else "rules")
                else:
                    ps["acc"] = max(0.0, ps["acc"] - 0.5 * dt)
                if ps["acc"] >= cfg.behavior_fight_s:
                    ps["acc"] = 0.0
                    self._fire(out, "FIGHT", t, [a, b], 1.0, ps.get("src", "rules"))
                for tr in (a, b):
                    tr.state["fight_now"] = max(tr.state.get("fight_now", 0.0), ps["acc"])
        for k in [k for k in self.pairs if k not in seen and t - self.pairs[k]["t"] > 3.0]:
            del self.pairs[k]

    # -------------------------------------------------------------- một khung
    def step(self, t: float, kps: np.ndarray, boxes: np.ndarray, confs: np.ndarray, mode: str) -> tuple[list[Track], list[BehaviorEvent]]:
        self.last_t = t
        self.tracker.update(t, kps, boxes, confs)
        live = [tr for tr in self.tracker.tracks.values() if tr.last_t == t]
        out: list[BehaviorEvent] = []
        for tr in live:
            tr.state["fight_now"] = 0.0
            self._update_track(tr, t, mode)
        for tr in live:
            self._rules_track(tr, t, mode, out)
        self._rules_fight(live, t, mode, out)
        return live, out

    def views(self, t: float) -> list[TrackView]:
        cfg = self.cfg
        res = []
        for tr in self.tracker.tracks.values():
            if t - tr.last_t > 1.0:
                continue
            st = tr.state
            still = st.get("still_s", 0.0)
            label, level = "", 0
            hold = st.get("posture_hold", "unknown")
            if st.get("fall_pending") is not None:
                label, level = "NGA?", 1
            if hold in DOWN:
                if hold == "lying":
                    lying = st.get("lying_s", 0.0)
                    label, level = ("NAM DUOI SAN", 2) if st.get("lie_fired") else (f"Nam {lying:.0f}s", 1)
                else:
                    label, level = "Guc", 1
                if still:
                    label += f" bat dong {still:.0f}/{cfg.behavior_still_s:.0f}s"
                elif hold == "lying" and st.get("motion") is not None:
                    label += " (cu dong)"
                if st.get("still_fired"):
                    label, level = ("NAM BAT DONG" if hold == "lying" else "NGOI GUC"), 2
            if t - st.get("fell_at", -1e9) <= 10:
                label, level = "TE NGA!", 2
            if st.get("fight_now", 0.0) > 0.2:
                label, level = ("DANH NHAU?", 1) if level < 2 else (label, level)
            if any(f[0] == "FIGHT" and t - f[1] <= 10 and f[4] == tr.id for f in self.fired):
                label, level = "DANH NHAU!", 2
            pr = st.get("probs") if t - st.get("probs_t", -1e9) <= PROB_FRESH_S else None
            res.append(TrackView(tr.id, [float(v) for v in tr.last.box], st.get("posture", "unknown"), label, level,
                                 round(still, 1), None if st.get("motion") is None else round(st["motion"], 3),
                                 None if pr is None else {k: round(v, 2) for k, v in pr.items()}))
        return res


# ============================================================================ nhiều camera + model
class Analyzer:
    def __init__(self, cfg, pose=None, classifier=None):
        self.cfg = cfg
        self.pose = pose  # PoseEstimator (pose.py) hoặc None khi test luật bằng keypoint có sẵn
        self.classifier = classifier  # (net, meta, device) hoặc None
        self.cams: dict[str, CameraAnalyzer] = {}

    @property
    def mode(self) -> str:
        return self.cfg.behavior_mode

    def cam(self, cam_id: str) -> CameraAnalyzer:
        if cam_id not in self.cams:
            self.cams[cam_id] = CameraAnalyzer(cam_id, self.cfg)
        return self.cams[cam_id]

    def drop_idle(self, now: float, idle_s: float = 60.0) -> None:
        for k in [k for k, c in self.cams.items() if c.last_t and now - c.last_t > idle_s]:
            del self.cams[k]

    def process(self, batch: list[tuple[str, np.ndarray, float]], draw: bool = True) -> list[CamResult]:
        """batch: [(cam_id, ảnh BGR, t)] — mỗi camera tối đa 1 ảnh. Chạy pose theo lô rồi luật từng camera."""
        if not batch:
            return []
        poses = self.pose.infer([b[1] for b in batch])
        return self.process_poses([(c, img, t, *p) for (c, img, t), p in zip(batch, poses)], draw=draw)

    def process_poses(self, items, draw: bool = True) -> list[CamResult]:
        """items: [(cam_id, ảnh|None, t, kps(n,17,3), boxes(n,4), confs(n,))]."""
        mode = self.mode
        results = []
        for cam_id, img, t, kps, boxes, confs in items:
            ca = self.cam(cam_id)
            live, events = ca.step(t, kps, boxes, confs, mode)
            results.append([ca, img, t, live, events])
        if self.classifier is not None and mode in ("hybrid", "model"):
            self._classify([(r[0], r[2], r[3]) for r in results])
        out = []
        for ca, img, t, live, events in results:
            views = ca.views(t)
            image = render(img, ca, views, self.cfg) if (draw and img is not None) else None
            out.append(CamResult(ca.cam_id, t, views, events, image))
        return out

    def _classify(self, groups) -> None:
        import torch

        from ares.behavior.model import build_window

        net, meta, device = self.classifier
        xs, owners = [], []
        for ca, t, live in groups:
            for tr in live:
                st = tr.state
                if tr.hits < 5 or t - st.get("probs_t", -1e9) < EVAL_EVERY_S:
                    continue
                others = [o for o in live if o is not tr]
                nb = None
                if others:
                    c = ((tr.last.box[0] + tr.last.box[2]) / 2, (tr.last.box[1] + tr.last.box[3]) / 2)
                    nb = min(others, key=lambda o: np.hypot((o.last.box[0] + o.last.box[2]) / 2 - c[0],
                                                            (o.last.box[1] + o.last.box[3]) / 2 - c[1]))
                h = [s for s in tr.hist if s.t >= t - meta["span"] - 0.5]
                h2 = [s for s in nb.hist if s.t >= t - meta["span"] - 0.5] if nb is not None else []
                x = build_window(np.array([s.t for s in h]), np.stack([s.kp for s in h]), t, tr.ref_h,
                                 np.array([s.t for s in h2]) if h2 else None,
                                 np.stack([s.kp for s in h2]) if h2 else None, T=meta["T"], span=meta["span"])
                xs.append(x)
                owners.append((tr, t))
        if not xs:
            return
        with torch.no_grad():
            p = torch.softmax(net(torch.from_numpy(np.stack(xs)).to(device)), dim=1).cpu().numpy()
        classes = meta["classes"]
        for (tr, t), row in zip(owners, p):
            tr.state["probs"] = {c: float(row[classes.index(c)]) if c in classes else 0.0 for c in CLASSES}
            tr.state["probs_t"] = t
            tr.state["_model_ok"] = True


# ============================================================================ vẽ
_COL = {0: (80, 200, 80), 1: (0, 170, 255), 2: (40, 40, 230)}


def render(img: np.ndarray, ca: CameraAnalyzer, views: list[TrackView], cfg) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    scale = cfg.behavior_clip_width / w if w > cfg.behavior_clip_width else 1.0
    out = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else img.copy()
    for v in views:
        tr = ca.tracker.tracks.get(v.id)
        col = _COL[v.level]
        x1, y1, x2, y2 = (int(c * scale) for c in v.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2 if v.level < 2 else 3)
        if tr is not None:
            kp = tr.last.kp
            for i, j in EDGES:
                if kp[i, 2] >= KP_CONF and kp[j, 2] >= KP_CONF:
                    cv2.line(out, (int(kp[i, 0] * scale), int(kp[i, 1] * scale)),
                             (int(kp[j, 0] * scale), int(kp[j, 1] * scale)), col, 2, cv2.LINE_AA)
        txt = f"#{v.id} {v.label}".strip()
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(out, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
