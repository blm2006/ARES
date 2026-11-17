"""Dịch vụ hành vi chạy trong ARES: nhận khung hình từ các camera, phân tích ở luồng riêng, lưu sự kiện + clip.

Gắn vào ARES (ares/runtime.py):
  * VisionTracker (camera điều khiển "main") và CameraHub (camera phụ) gọi `offer(cam_id, ảnh, t)` sau mỗi khung —
    chỉ giữ tham chiếu ảnh MỚI NHẤT của mỗi camera, không copy, không chặn luồng vision.
  * Luồng `ares-behavior` chạy tối đa behavior_fps lần/giây: gom ảnh mới của mọi camera thành 1 lô -> YOLO-pose ->
    luật/model -> sự kiện. Chậm (CPU yếu) thì tự bỏ khung cũ, không dồn hàng.
  * Sự kiện: ghi DB + thư mục bằng chứng, đẩy vào nhật ký sự kiện của ARES (emit), hiện ở trang /behavior.
Không đọc/ghi bất cứ thứ gì của bộ điều khiển đèn/điều hòa.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from ares.behavior import KIND_SEVERITY, KIND_VI
from ares.behavior.analyzer import Analyzer, BehaviorEvent, CameraAnalyzer
from ares.behavior.store import ClipRecorder, EventStore, event_dir_name
from ares.state import Event

log = logging.getLogger(__name__)


def resolve_path(p: str, base: Path) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else base / q


class BehaviorMonitor:
    def __init__(self, cfg, clock, emit: Callable[[Event], None] | None = None,
                 name_of: Callable[[str], str] | None = None, analyzer: Analyzer | None = None):
        self.cfg, self.clock = cfg, clock
        self.emit = emit or (lambda e: None)
        self.name_of = name_of or (lambda cid: cid)
        self.root = Path(cfg.behavior_dir)
        self.store = EventStore(self.root)
        self.clips = ClipRecorder(cfg, self.store)
        self.analyzer = analyzer  # test có thể bơm Analyzer giả; bình thường nạp ở luồng nền
        self._latest: dict[str, tuple[np.ndarray, float]] = {}
        self._done_t: dict[str, float] = {}
        self._overlay: dict[str, tuple[bytes, float, list]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self.loading = analyzer is None
        self.error = ""
        self.classifier_info: dict | None = None
        self.ppe_info: dict = {"ready": False, "error": "", "hint": ""}
        self.stats = {"loop_fps": 0.0, "infer_ms": 0.0, "frames": 0, "events": 0}
        allow = cfg.behavior_cameras.strip()
        self._allow = None if allow.lower() in ("", "all", "*") else {x.strip() for x in allow.split(",") if x.strip()}
        self._last_house = 0.0

    # ------------------------------------------------------------------ vào
    def wants(self, cam_id: str) -> bool:
        return self._allow is None or cam_id in self._allow

    def offer(self, cam_id: str, image: np.ndarray | None, t: float) -> None:
        """Gọi từ luồng vision/hub. O(1), không copy. Ảnh không được bị sửa tại chỗ sau khi gọi."""
        if image is None or not self.wants(cam_id):
            return
        with self._lock:
            self._latest[cam_id] = (image, t)

    # ------------------------------------------------------------------ vòng đời
    def start(self) -> None:
        self._th = threading.Thread(target=self._run, name="ares-behavior", daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=8)
        self.clips.close()
        self.store.close()

    def _load(self) -> None:
        from ares.behavior.analyzer import Analyzer
        from ares.behavior.pose import PoseEstimator, resolve_device
        from ares.config import REPO_ROOT

        cfg = self.cfg
        device = resolve_device(cfg.behavior_device or cfg.device)
        pose = PoseEstimator(resolve_path(cfg.behavior_weights, REPO_ROOT), device, cfg.behavior_imgsz,
                             cfg.behavior_conf, cfg.half)
        clf = None
        cpath = resolve_path(cfg.behavior_classifier, REPO_ROOT) if cfg.behavior_classifier else self.root / "models" / "current.pt"
        if cfg.behavior_mode != "rules" and cpath.exists():
            from ares.behavior.model import load_classifier
            try:
                net, meta = load_classifier(cpath, device)
                clf = (net, meta, device)
                self.classifier_info = {"path": str(cpath), "classes": meta["classes"], "metrics": meta.get("metrics", {}),
                                        "trained_at": meta.get("trained_at")}
            except Exception as e:  # noqa: BLE001
                log.exception("Không nạp được model hành vi %s", cpath)
                self.emit(Event(self.clock.time(), "BEHAVIOR_MODEL_FAILED", "warn",
                                f"Không nạp được model hành vi ({e}) — chỉ dùng luật."))
        elif cfg.behavior_mode == "model":
            self.emit(Event(self.clock.time(), "BEHAVIOR_NO_MODEL", "warn",
                            "ARES_BEHAVIOR_MODE=model nhưng chưa có model đã train — tạm dùng luật. "
                            "Train bằng: python -m ares.behavior train"))
        from ares.ppe.load import load_detector, load_rules

        ppe, self.ppe_info = load_detector(cfg)
        self.analyzer = Analyzer(cfg, pose, clf, ppe)
        self.analyzer.ppe_rules = load_rules(cfg)  # luật theo camera lần trước server gửi (dùng được cả khi mất mạng)
        if ppe is not None:
            self.emit(Event(self.clock.time(), "PPE_READY", "info",
                            f"AI bảo hộ sẵn sàng: {self.ppe_info['weights']} trên {ppe.device}, nhận ra: "
                            + ", ".join(self.ppe_info["items"]) + "."))
        mode = cfg.behavior_mode if clf is not None or cfg.behavior_mode == "rules" else "rules (chưa có model)"
        self.emit(Event(self.clock.time(), "BEHAVIOR_READY", "info",
                        f"Nhận diện hành vi sẵn sàng: pose {Path(pose.weights).name} trên {device}, chế độ {mode}."))

    def _run(self) -> None:
        if self.analyzer is None:
            try:
                self._load()
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                log.exception("Không khởi động được nhận diện hành vi")
                self.emit(Event(self.clock.time(), "BEHAVIOR_FAILED", "error",
                                f"Không khởi động được nhận diện hành vi: {self.error}"))
                return
            finally:
                self.loading = False
        ema = 0.0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                if self.step():
                    dt = time.monotonic() - t0
                    ema = dt if ema == 0 else 0.8 * ema + 0.2 * dt
                    self.stats["loop_fps"] = round(1.0 / ema, 2) if ema > 0 else 0.0
                self._housekeep()
            except Exception as e:  # noqa: BLE001 - một vòng lỗi không được giết luồng
                self.error = f"{type(e).__name__}: {e}"
                log.exception("Vòng nhận diện hành vi lỗi")
            wait = 1.0 / self.cfg.behavior_fps - (time.monotonic() - t0)
            self._stop.wait(max(0.005, wait))

    # ------------------------------------------------------------------ một vòng
    def step(self) -> bool:
        with self._lock:
            batch = [(c, img, t) for c, (img, t) in self._latest.items() if t > self._done_t.get(c, -1e18)]
        if not batch or self.analyzer is None:
            return False
        t0 = time.monotonic()
        results = self.analyzer.process(batch, draw=True)
        self.stats["infer_ms"] = round((time.monotonic() - t0) * 1000, 1)
        self.stats["frames"] += len(batch)
        import cv2

        for r in results:
            self._done_t[r.cam_id] = r.t
            if r.image is not None:
                ok, buf = cv2.imencode(".jpg", r.image, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok:
                    self._overlay[r.cam_id] = (buf.tobytes(), r.t, r.tracks)
                self.clips.push(r.cam_id, r.t, r.image)
            for ev in r.events:
                self._handle(ev, r.image, self.analyzer.cams.get(r.cam_id))
        return True

    def _handle(self, ev: BehaviorEvent, image: np.ndarray | None, ca: CameraAnalyzer | None) -> None:
        cfg = self.cfg
        eid = time.strftime("%Y%m%d-%H%M%S", time.localtime(ev.t)) + "-" + uuid.uuid4().hex[:6]
        rel = f"events/{event_dir_name(ev.t, cfg.tz)}/{eid}"
        d = self.root / rel
        d.mkdir(parents=True, exist_ok=True)
        severity = KIND_SEVERITY.get(ev.kind, "warn")
        cam_name = self.name_of(ev.cam_id)
        msg = f"{KIND_VI.get(ev.kind, ev.kind)} — camera '{cam_name}'"
        if ev.kind in ("LYING_STILL", "SLUMPED"):
            msg += f", bất động {ev.detail.get('still_s', cfg.behavior_still_s):.0f} giây"
            if ev.detail.get("after_fall"):
                msg += " sau cú ngã"
                severity = "error"
        if ev.kind == "LYING_DOWN":
            msg += (f", nằm xuống đột ngột ({ev.detail.get('lying_s', 0):.0f} giây sau khi đang đứng/ngồi)"
                    if ev.detail.get("sudden") else f", nằm liên tục {ev.detail.get('lying_s', 0):.0f} giây")
            msg += ", vẫn còn cử động" if ev.detail.get("moving") else ", ít cử động"
        if ev.kind == "PPE":
            msg += f": {ev.detail.get('text', '')}"
        if ev.kind == "FIGHT":
            msg += f", người #{ev.track_ids[0]} và #{ev.track_ids[1]}"
        else:
            msg += f", người #{ev.track_ids[0]}"
        when = self.clock.now()
        meta = {"id": eid, "t": ev.t, "ts": when.isoformat(timespec="seconds"), "cam_id": ev.cam_id, "cam_name": cam_name,
                "kind": ev.kind, "label": KIND_VI.get(ev.kind, ev.kind), "severity": severity, "message": msg,
                "track_ids": ev.track_ids, "score": ev.score, "source": ev.source, "detail": ev.detail,
                "center": ev.center, "config": {"still_s": cfg.behavior_still_s, "mode": cfg.behavior_mode}}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.add({"id": eid, "t": ev.t, "ts": meta["ts"], "cam_id": ev.cam_id, "cam_name": cam_name,
                        "kind": ev.kind, "label": meta["label"], "severity": severity, "message": msg,
                        "track_ids": json.dumps(ev.track_ids), "score": ev.score, "source": ev.source,
                        "status": "new", "verdict": "", "dir": rel})
        self.clips.start(eid, ev.cam_id, ev.t, d, image, self._kp_payload(ev, ca))
        self.stats["events"] += 1
        self.emit(Event(self.clock.time(), f"BEHAVIOR_{ev.kind}", severity, msg,
                        {"event_id": eid, "cam_id": ev.cam_id, "kind": ev.kind}))

    @staticmethod
    def _kp_payload(ev: BehaviorEvent, ca: CameraAnalyzer | None) -> dict:
        """Chuỗi khung xương ~20 giây gần nhất của các người liên quan (để train lại sau khi duyệt)."""
        if ca is None:
            return {}
        out = {}
        for k, tid in enumerate(ev.track_ids):
            tr = ca.tracker.tracks.get(tid)
            if tr is None or not tr.hist:
                continue
            h = [s for s in tr.hist if s.t >= ev.t - 20]
            out[f"t{k}"] = np.array([s.t for s in h], np.float64)
            out[f"kp{k}"] = np.stack([s.kp for s in h]).astype(np.float32)
            out[f"box{k}"] = np.stack([s.box for s in h]).astype(np.float32)
        out["t_event"] = np.array([ev.t], np.float64)
        return out

    def _housekeep(self) -> None:
        now = self.clock.time()
        if now - self._last_house < 60:
            return
        self._last_house = now
        if self.analyzer is not None:
            self.analyzer.drop_idle(now)
        with self._lock:
            for c in [c for c, (_, t) in self._latest.items() if now - t > 120]:
                del self._latest[c]
                self._overlay.pop(c, None)
        n = self.store.cleanup(self.cfg.behavior_keep_days)
        if n:
            log.info("Đã xoá %d sự kiện hành vi cũ hơn %d ngày (chưa duyệt)", n, self.cfg.behavior_keep_days)

    # ------------------------------------------------------------------ cho dashboard
    def frame(self, cam_id: str) -> tuple[bytes | None, float]:
        o = self._overlay.get(cam_id)
        return (o[0], o[1]) if o else (None, 0.0)

    def set_ppe_rules(self, rules: dict) -> bool:
        """Luật đồ bảo hộ theo camera từ server ({cam_id: {"required": [...], "forbidden": [...]}}). True nếu có đổi."""
        from ares.ppe import clean_rules
        from ares.ppe.load import save_rules

        new = {str(k): clean_rules(v) for k, v in (rules or {}).items()}
        if self.analyzer is None or new == self.analyzer.ppe_rules:
            return False
        self.analyzer.ppe_rules = new
        try:
            save_rules(self.cfg, new)
        except OSError:
            log.warning("Không lưu được luật đồ bảo hộ")
        return True

    def ppe_status(self) -> dict:
        a = self.analyzer
        return {**self.ppe_info, "default": a.ppe_default if a else None,
                "cameras": {c: r for c, r in (a.ppe_rules if a else {}).items() if any(r.values())}}

    def status(self) -> dict:
        now = self.clock.time()
        cams = []
        for cid, (jpg, t, views) in list(self._overlay.items()):
            cams.append({"id": cid, "name": self.name_of(cid), "age_s": round(now - t, 1), "n_people": len(views),
                         "tracks": [{"id": v.id, "posture": v.posture, "label": (v.text or v.label) + (" · " + v.ppe_text if v.ppe_text else ""),
                                     "level": v.level, "pose": v.pose,
                                     "conf": v.conf,
                                     "still_s": v.still_s, "motion": v.motion, "probs": v.probs} for v in views]})
        cfg = self.cfg
        return {"enabled": True, "loading": self.loading, "error": self.error, "mode": cfg.behavior_mode,
                "classifier": self.classifier_info, "stats": dict(self.stats), "cameras": cams, "ppe": self.ppe_status(),
                "counts": self.store.counts(),
                "config": {"still_s": cfg.behavior_still_s, "fps": cfg.behavior_fps, "cameras": cfg.behavior_cameras,
                           "clip_pre_s": cfg.behavior_clip_pre_s, "clip_post_s": cfg.behavior_clip_post_s,
                           "weights": cfg.behavior_weights}}
