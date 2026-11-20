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


POSE_VI = {"standing": "Đứng", "walking": "Đang đi", "sitting": "Ngồi", "lying": "Nằm", "slumped": "Ngồi gục",
           "upper": "Thấy nửa người", "unknown": "Chưa rõ tư thế"}
POSE_KEYS = ("standing", "walking", "sitting", "lying", "slumped", "upper")


def no_accent(s: str) -> str:
    """Bỏ dấu tiếng Việt (cho nơi chỉ hiện được ASCII, vd chữ vẽ bằng OpenCV khi máy không có font)."""
    import unicodedata

    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def summarize(views: list) -> tuple[int, str]:
    """Trạng thái chung của 1 khung hình -> (mức 0 an toàn | 1 chú ý | 2 báo động, câu tiếng Việt)."""
    if not views:
        return 0, "Không có người trong khung hình"
    level = max(v.level for v in views)
    if level == 0:
        cnt = {k: sum(1 for v in views if v.pose == k) for k in POSE_KEYS}
        parts = [f"{n} {POSE_VI[k].lower()}" for k, n in cnt.items() if n]
        return 0, f"AN TOÀN · {len(views)} người" + (": " + ", ".join(parts) if parts else "")
    who = [f"{v.note or v.text} (người #{v.id})" for v in sorted(views, key=lambda v: -v.level) if v.level == level]
    return level, ("BÁO ĐỘNG: " if level == 2 else "CHÚ Ý: ") + " · ".join(who[:3])


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
    ppe: dict | None = None  # đồ bảo hộ: {"missing": [...], "forbidden": [...], "ok": [...], "pending": [...], "all_ok": bool}
    ppe_text: str = ""  # "Thiếu kính, chụp tai" / "Đeo găng tay (cấm)"
    pose: str = "unknown"  # tư thế ổn định để hiển thị: standing | walking | sitting | lying | slumped | unknown
    note: str = ""  # câu cho thanh trạng thái khi cần chú ý (hành vi và/hoặc đồ bảo hộ)
    text: str = ""  # nhãn tiếng Việt vẽ lên hình: "Đứng", "Ngồi", "Nằm 3 giây (cử động)", "TÉ NGÃ!"...
    conf: float = 0.0  # độ chắc của tư thế 0..1 (độ đồng thuận 1 giây gần nhất × độ tin cậy các khớp)


@dataclass
class CamResult:
    cam_id: str
    t: float
    tracks: list[TrackView]
    events: list[BehaviorEvent]
    image: np.ndarray | None  # ảnh đã vẽ (thu nhỏ về clip_width) — None nếu không yêu cầu vẽ
    alert: tuple | None = None  # (mức, câu) của AI bổ sung cả camera (cháy/khói) đang hiệu lực


class CameraAnalyzer:
    """Trạng thái của MỘT camera."""

    def __init__(self, cam_id: str, cfg):
        self.cam_id, self.cfg = cam_id, cfg
        self.tracker = Tracker(gate=bool(getattr(cfg, "behavior_track_gate", True)),
                               warmup_s=float(getattr(cfg, "behavior_track_warmup_s", 2.0)),
                               edge=float(getattr(cfg, "behavior_track_edge", 0.06)),
                               confirm_s=float(getattr(cfg, "behavior_track_confirm_s", 4.0)),
                               recover_s=float(getattr(cfg, "behavior_track_recover_s", 6.0)))
        self.pairs: dict[tuple[int, int], dict] = {}
        self.fired: deque = deque(maxlen=200)  # (kind, t, center, ref_h, track_id)
        self.last_t = 0.0
        self.ppe_t = -1e18  # lần gần nhất chạy AI bảo hộ cho camera này
        self.ppe_dets: list = []  # các món đồ bảo hộ thấy ở lần đó (để vẽ)
        self.ppe_on = False
        from ares.hazards.rules import CamHazards

        self.hz = CamHazards(cfg)  # AI bổ sung: cháy/khói, hút thuốc

    # -------------------------------------------------------------- đồ bảo hộ
    def ppe_step(self, t: float, live: list[Track], dets: list, rules: dict, scale: float, out: list) -> None:
        """Ghép kết quả YOLO đồ bảo hộ với từng người đang thấy, cập nhật bộ đếm, sinh sự kiện "PPE"."""
        from ares.ppe import describe
        from ares.ppe.rules import assess, update_track

        self.ppe_t, self.ppe_dets, self.ppe_on = t, dets, True
        items = rules["required"] + rules["forbidden"]
        trs = [tr for tr in live if tr.hits >= 3]
        sts = assess([(tr.last.kp, tr.last.box) for tr in trs], dets, items, scale)
        for tr, st in zip(trs, sts):
            v = update_track(tr.state, t, st, rules, self.cfg)
            if v:
                # lần đầu của 1 người: qua bộ lọc chung (chống báo trùng khi bám vết đổi mã người tại cùng chỗ);
                # các lần sau do bộ đếm đồ bảo hộ tự quyết (đeo lại rồi tháo, hoặc nhắc lại sau ppe_remind_s)
                self._fire(out, "PPE", t, [tr], 0.8, "ppe", force=v["again"], missing=v["missing"], forbidden=v["forbidden"],
                           text=describe(v["missing"], v["forbidden"]))

    # -------------------------------------------------------------- AI bổ sung (cháy/khói, hút thuốc)
    def hazard_step(self, t: float, live: list[Track], found: dict, out: list) -> None:
        def fire_cb(what, best, dets):
            box = best.box if best is not None else (0, 0, 0, 0)
            out.append(BehaviorEvent("FIRE", self.cam_id, t, [], round(float(best.conf if best else 0.5), 3),
                                     ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), "model",
                                     {"what": what, "text": "Thấy lửa" if what == "lửa" else "Thấy khói",
                                      "boxes": [[d.cls, *[round(v, 1) for v in d.box], round(d.conf, 3)] for d in dets][:20]}))

        def person_cb(tr, box, cig):
            self._fire(out, "SMOKING", t, [tr], round(float(box.conf if box else 0.5), 3), "model", text="Đang hút thuốc",
                       boxes=[[d.cls, *[round(v, 1) for v in d.box], round(d.conf, 3)] for d in cig][:20])

        self.hz.step(t, found, live, fire_cb, person_cb)

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

    def _fire(self, out: list, kind: str, t: float, trs: list[Track], score: float, source: str, force: bool = False,
              **detail) -> None:
        cs = [((tr.last.box[0] + tr.last.box[2]) / 2, (tr.last.box[1] + tr.last.box[3]) / 2) for tr in trs]
        center = (float(np.mean([c[0] for c in cs])), float(np.mean([c[1] for c in cs])))
        rh = float(np.mean([tr.ref_h for tr in trs]))
        ids = [tr.id for tr in trs]
        if not force and not self._cooldown_ok(kind, t, ids, center, rh):
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
        legs: deque = st.setdefault("legs", deque(maxlen=10))  # có thấy hông + gối không (để phân biệt đứng / ngồi)
        legs.append(ff.sitting is not None)
        vis = s.kp[:, 2][s.kp[:, 2] >= KP_CONF]
        agree = recent.count(hold) / len(recent) if recent else 0.0
        st["posture_conf"] = float(agree * (vis.mean() if len(vis) else float(s.conf)))

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
        # ---- đang đi hay đứng yên: tốc độ tâm khung người (theo chiều cao người / giây), có trễ để nhãn không nhấp nháy
        if a and b:
            ca_ = np.mean([[(x.box[0] + x.box[2]) / 2, (x.box[1] + x.box[3]) / 2] for x in a], axis=0)
            cb_ = np.mean([[(x.box[0] + x.box[2]) / 2, (x.box[1] + x.box[3]) / 2] for x in b], axis=0)
            dtc = max(0.3, np.mean([x.t for x in b]) - np.mean([x.t for x in a]))
            v = float(np.hypot(*(cb_ - ca_))) / max(ref_h, 1.0) / dtc
            st["speed"] = v if st.get("speed") is None else 0.5 * st["speed"] + 0.5 * v
            st["walking"] = st["speed"] > (0.18 if st.get("walking") else 0.3)
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
    def step(self, t: float, kps: np.ndarray, boxes: np.ndarray, confs: np.ndarray, mode: str,
             frame_wh: tuple[int, int] | None = None) -> tuple[list[Track], list[BehaviorEvent]]:
        self.last_t = t
        self.tracker.update(t, kps, boxes, confs, frame_wh)
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
            hold = st.get("posture_hold", "unknown")
            legs = st.get("legs") or ()
            if hold in ("upright", "unknown") and legs and sum(legs) * 2 < len(legs):
                pose = "upper"  # chỉ thấy nửa người trên (vd webcam để gần): không đoán đứng hay ngồi
            else:
                pose = {"upright": "walking" if st.get("walking") else "standing"}.get(hold, hold)
            text, level = POSE_VI[pose], 0
            if st.get("fall_pending") is not None:
                text, level = "Mất thăng bằng?", 1
            if hold in DOWN:
                if hold == "lying":
                    lying = st.get("lying_s", 0.0)
                    text, level = ("NẰM DƯỚI SÀN", 2) if st.get("lie_fired") else (f"Nằm {lying:.0f} giây" if lying >= 1 else "Vừa nằm xuống", 1)
                else:
                    text, level = "Ngồi gục", 1
                if still:
                    text += f", bất động {still:.0f}/{cfg.behavior_still_s:.0f} giây"
                elif hold == "lying" and st.get("motion") is not None:
                    text += " (còn cử động)"
                if st.get("still_fired"):
                    text, level = ("NẰM BẤT ĐỘNG" if hold == "lying" else "NGỒI GỤC BẤT ĐỘNG"), 2
            if t - st.get("fell_at", -1e9) <= 10:
                text, level = "TÉ NGÃ!", 2
            if st.get("fight_now", 0.0) > 0.2:
                text, level = ("Xô đẩy?", 1) if level < 2 else (text, level)
            if any(f[0] == "FIGHT" and t - f[1] <= 10 and f[4] == tr.id for f in self.fired):
                text, level = "ĐÁNH NHAU!", 2
            pr = st.get("probs") if t - st.get("probs_t", -1e9) <= PROB_FRESH_S else None
            if t - st.get("smoking_t", -1e9) <= 5.0:  # AI bổ sung: đang hút thuốc
                text, level = (text if level else "Hút thuốc"), max(level, 1)
            ppe, ppe_text, note = None, "", (text if level else "")
            if self.ppe_on and t - st.get("ppe_t", -1e9) <= 5.0:
                from ares.ppe.rules import short_text

                ppe = st.get("ppe")
                ppe_text = short_text(ppe)
                if ppe_text:
                    note = f"{note} · {ppe_text}" if note else ppe_text
                    level = max(level, 1)
            res.append(TrackView(tr.id, [float(v) for v in tr.last.box], st.get("posture", "unknown"),
                                 no_accent(text) if level else "", level,
                                 round(still, 1), None if st.get("motion") is None else round(st["motion"], 3),
                                 None if pr is None else {k: round(v, 2) for k, v in pr.items()},
                                 pose=pose, text=text, conf=round(st.get("posture_conf", 0.0), 2),
                                 ppe=ppe, ppe_text=ppe_text, note=note))
        return res


# ============================================================================ nhiều camera + model
class Analyzer:
    def __init__(self, cfg, pose=None, classifier=None, ppe=None, hazards=None):
        from ares.ppe import clean_rules

        self.cfg = cfg
        self.hazards = hazards or {}  # {mục: HazardDetector} — AI bổ sung (ares/hazards)
        self.pose = pose  # PoseEstimator (pose.py) hoặc None khi test luật bằng keypoint có sẵn
        self.classifier = classifier  # (net, meta, device) hoặc None
        self.ppe = ppe  # PPEDetector (ares/ppe/detector.py) hoặc None = chưa cài model đồ bảo hộ
        self.cams: dict[str, CameraAnalyzer] = {}
        # luật đồ bảo hộ: mặc định của máy (ARES_PPE_REQUIRED / _FORBIDDEN) + riêng từng camera (server gửi về theo địa điểm)
        req = clean_rules(cfg.ppe_required)
        self.ppe_default = clean_rules({"required": req["required"],
                                        "forbidden": req["forbidden"] + clean_rules({"forbidden": [
                                            x.strip() for x in cfg.ppe_forbidden.split(",")]})["forbidden"]})
        self.ppe_rules: dict[str, dict] = {}

    def rules_for(self, cam_id: str) -> dict:
        r = self.ppe_rules.get(cam_id)  # luật riêng của camera (theo địa điểm trên server); trống thì dùng mặc định của máy
        return r if r and any(r.values()) else self.ppe_default

    def _ppe_scale(self, img) -> float:
        return min(1.0, float(self.cfg.ppe_imgsz) / max(img.shape[:2]))

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
        ppe = None
        if self.ppe is not None:  # chỉ camera có luật đồ bảo hộ, có người, và đã tới lượt (ppe_fps) mới chạy YOLO đồ bảo hộ
            due = [(c, img) for (c, img, t), p in zip(batch, poses)
                   if len(p[1]) and any(self.rules_for(c).values()) and t - self.cam(c).ppe_t >= 1.0 / self.cfg.ppe_fps - 1e-3]
            if due:
                ppe = dict(zip([c for c, _ in due], self.ppe.infer([img for _, img in due])))
        hz = None
        if self.hazards:  # AI bổ sung: mỗi camera ~hazard_fps lần/giây
            due = [(c, img) for c, img, t in batch if t - self.cam(c).hz.t >= 1.0 / self.cfg.hazard_fps - 1e-3]
            if due:
                hz = {c: {} for c, _ in due}
                for name, det in self.hazards.items():
                    if det.name == "smoking" and not any(len(p[1]) for (c, _, _), p in zip(batch, poses) if c in hz):
                        continue  # không có người thì khỏi tìm điếu thuốc
                    for (c, _), dets in zip(due, det.infer([img for _, img in due])):
                        hz[c][name] = dets
        return self.process_poses([(c, img, t, *p) for (c, img, t), p in zip(batch, poses)], draw=draw, ppe=ppe, hz=hz)

    def process_poses(self, items, draw: bool = True, ppe: dict | None = None, hz: dict | None = None) -> list[CamResult]:
        """items: [(cam_id, ảnh|None, t, kps(n,17,3), boxes(n,4), confs(n,))]. ppe: {cam_id: [PPEDet]} nếu khung này có
        chạy AI bảo hộ (test có thể bơm thẳng)."""
        mode = self.mode
        results = []
        for cam_id, img, t, kps, boxes, confs in items:
            ca = self.cam(cam_id)
            live, events = ca.step(t, kps, boxes, confs, mode,
                                   None if img is None else (img.shape[1], img.shape[0]))
            rules = self.rules_for(cam_id)
            if ppe is not None and cam_id in ppe and any(rules.values()):
                ca.ppe_step(t, live, ppe[cam_id], rules, 1.0 if img is None else self._ppe_scale(img), events)
            elif not any(rules.values()):
                ca.ppe_on = False
            if hz is not None and cam_id in hz:
                ca.hazard_step(t, live, hz[cam_id], events)
            results.append([ca, img, t, live, events])
        if self.classifier is not None and mode in ("hybrid", "model"):
            self._classify([(r[0], r[2], r[3]) for r in results])
        out = []
        for ca, img, t, live, events in results:
            views = ca.views(t)
            image = render(img, ca, views, self.cfg) if (draw and img is not None) else None
            al = ca.hz.alert
            out.append(CamResult(ca.cam_id, t, views, events, image, (al[1], al[2]) if al and t <= al[0] else None))
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
_COL = {0: (80, 200, 80), 1: (0, 170, 255), 2: (40, 40, 230)}  # BGR: xanh an toàn · cam chú ý · đỏ báo động
# màu khung xương theo bộ phận (BGR): đầu xanh lá · tay xanh dương · thân tím hồng · chân cam
_PART_COL = {"head": (90, 220, 90), "arm": (255, 170, 60), "torso": (220, 110, 230), "leg": (40, 160, 255)}


def _edge_part(i: int, j: int) -> str:
    if i <= 4 and j <= 4:
        return "head"
    if {i, j} <= {5, 6, 11, 12} or (i == 0 and j in (5, 6)):
        return "torso"
    return "arm" if max(i, j) <= 10 else "leg"


_FONT_FILES = ("segoeuib.ttf", "arialbd.ttf", "tahomabd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
               "/Library/Fonts/Arial Bold.ttf", "arial.ttf")
_fonts: dict = {}
_patches: dict = {}


def _font(size: int):
    """Font có dấu tiếng Việt (PIL). Không có font/PIL -> None, khi đó vẽ chữ không dấu bằng OpenCV."""
    if size not in _fonts:
        f = None
        try:
            from PIL import ImageFont

            for name in _FONT_FILES:
                try:
                    f = ImageFont.truetype(name, size)
                    break
                except OSError:
                    continue
        except ImportError:
            pass
        _fonts[size] = f
    return _fonts[size]


def text_patch(text: str, size: int, fg: tuple, bg: tuple, pad: int = 4) -> np.ndarray:
    """Ảnh BGR nhỏ chứa chữ (có dấu) trên nền màu — có bộ nhớ đệm, vẽ 1 lần dùng nhiều khung."""
    key = (text, size, fg, bg, pad)
    hit = _patches.get(key)
    if hit is not None:
        return hit
    font = _font(size)
    if font is not None:
        from PIL import Image, ImageDraw

        l, t_, r, b = font.getbbox(text)
        asc, desc = font.getmetrics()
        im = Image.new("RGB", (r - l + 2 * pad, asc + desc + pad), bg[::-1])
        ImageDraw.Draw(im).text((pad - l, pad // 2), text, font=font, fill=fg[::-1])
        patch = np.asarray(im)[:, :, ::-1].copy()
    else:
        import cv2

        txt, sc = no_accent(text), size / 26.0
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
        patch = np.full((th + base + pad, tw + 2 * pad, 3), bg, np.uint8)
        cv2.putText(patch, txt, (pad, th + pad // 2), cv2.FONT_HERSHEY_SIMPLEX, sc, fg, 1, cv2.LINE_AA)
    if len(_patches) > 600:
        _patches.clear()
    _patches[key] = patch
    return patch


def _blit(out: np.ndarray, patch: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    H, W = out.shape[:2]
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    ph, pw = min(patch.shape[0], H - y), min(patch.shape[1], W - x)
    if ph <= 0 or pw <= 0:
        return
    roi = out[y:y + ph, x:x + pw]
    roi[:] = patch[:ph, :pw] if alpha >= 1.0 else (alpha * patch[:ph, :pw] + (1 - alpha) * roi).astype(np.uint8)


def render(img: np.ndarray, ca: CameraAnalyzer, views: list[TrackView], cfg) -> np.ndarray:
    """Vẽ lên hình: khung người theo mức (xanh/cam/đỏ), khung xương màu theo bộ phận, nhãn "ID 1 · NGỒI 92%",
    và thanh trạng thái phía trên (AN TOÀN / CHÚ Ý / BÁO ĐỘNG). Báo động: viền đỏ quanh cả khung hình."""
    import cv2

    h, w = img.shape[:2]
    scale = cfg.behavior_clip_width / w if w > cfg.behavior_clip_width else 1.0
    out = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else img.copy()
    H, W = out.shape[:2]
    fs = max(11, min(15, W // 54))  # cỡ chữ theo chiều rộng hình (nhỏ gọn để nhiều người không che nhau)
    banner = bool(getattr(cfg, "behavior_overlay_status", True))
    top = (fs + 12) if banner else 0
    for v in sorted(views, key=lambda v: v.level):  # người đang báo động vẽ sau cùng (nằm trên)
        tr = ca.tracker.tracks.get(v.id)
        col = _COL[v.level]
        x1, y1, x2, y2 = (int(c * scale) for c in v.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 1 if v.level < 2 else 2)
        if tr is not None:
            kp = tr.last.kp
            for i, j in EDGES:
                if kp[i, 2] >= KP_CONF and kp[j, 2] >= KP_CONF:
                    cv2.line(out, (int(kp[i, 0] * scale), int(kp[i, 1] * scale)),
                             (int(kp[j, 0] * scale), int(kp[j, 1] * scale)), _PART_COL[_edge_part(i, j)], 1, cv2.LINE_AA)
            for k in range(5, 17):
                if kp[k, 2] >= KP_CONF:
                    cv2.circle(out, (int(kp[k, 0] * scale), int(kp[k, 1] * scale)), 2, (255, 255, 255), -1, cv2.LINE_AA)
        txt = f"ID {v.id} · {(v.text or POSE_VI.get(v.pose, '')).upper()}"
        if v.level == 0 and v.conf > 0:
            txt += f" {round(v.conf * 100)}%"
        patch = text_patch(txt, fs, (255, 255, 255), col)
        lines = 2 if v.ppe is not None else 1  # nhãn nằm phía trên khung người (không che mặt)
        ly = max(top, y1 - lines * patch.shape[0])
        _blit(out, patch, x1, ly)
        if v.ppe is not None:  # dòng thứ hai: đồ bảo hộ (cam = vi phạm · xanh = đủ · xám = không thấy rõ nên không kết luận)
            if v.ppe_text:
                p2 = text_patch(v.ppe_text.upper(), fs, (255, 255, 255), _COL[1])
            elif v.ppe.get("all_ok"):
                p2 = text_patch("ĐỦ BẢO HỘ", fs, (255, 255, 255), _COL[0])
            elif v.ppe.get("pending"):
                p2 = text_patch("ĐANG KIỂM ĐỒ BẢO HỘ…", fs, (255, 255, 255), (120, 120, 120))
            else:
                p2 = text_patch("CHƯA THẤY RÕ ĐỒ BẢO HỘ", fs, (255, 255, 255), (120, 120, 120))
            _blit(out, p2, x1, ly + patch.shape[0])
    if ca.ppe_on and getattr(cfg, "ppe_draw", True) and ca.ppe_dets and ca.last_t - ca.ppe_t <= 1.5:
        for d in ca.ppe_dets:  # các món đồ bảo hộ AI nhìn thấy: hộp mảnh màu xanh ngọc
            if d.worn:
                bx1, by1, bx2, by2 = (int(c * scale) for c in d.box)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), (220, 220, 60), 1)
    if ca.hz.dets and ca.last_t - ca.hz.t <= 1.5:  # AI bổ sung: hộp mảnh đỏ (lửa/khói) / cam (điếu thuốc)
        for dets in ca.hz.dets.values():
            for d in dets:
                bx1, by1, bx2, by2 = (int(c * scale) for c in d.box)
                col = (40, 40, 230) if d.cls in ("fire", "smoke") else (0, 140, 255)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), col, 1)
                p = text_patch(f"{d.label} {round(d.conf * 100)}%", max(10, fs - 3), (255, 255, 255), col, pad=2)
                _blit(out, p, bx1, max(top, by1 - p.shape[0]))
    if banner:
        level, text = summarize(views)
        al = ca.hz.alert
        if al and ca.last_t <= al[0] and al[1] >= level:
            level, text = al[1], al[2]
        strip = out[:top]
        strip[:] = (strip * 0.35).astype(np.uint8)  # dải tối mờ phía trên
        head = ("TRẠNG THÁI: " if level == 0 else "") + text
        _blit(out, text_patch(head, fs, _COL[level] if level else (140, 235, 140), (0, 0, 0), pad=6), 4, 2, alpha=0.999)
        if level == 2:
            cv2.rectangle(out, (1, 1), (W - 2, H - 2), _COL[2], 3)
    return out
