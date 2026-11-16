"""Pipeline mỗi khung: phát hiện → (bù camera) → tracking → động học → đèn → gán người-xe → model tầng 2
→ đọc biển số → đếm/đo tốc độ → 16 rule → vẽ → bằng chứng.

Ba luồng như spec: (1) đọc video (VideoSource), (2) vòng xử lý này, (3) ghi file (EvidenceManager + AsyncVideoWriter).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

import cv2
import numpy as np

from traffic_safety_ai.analytics.counting import LineCounter, SpeedSampler, SpeedTrapMeter
from traffic_safety_ai.core.association import RiderAssociator
from traffic_safety_ai.core.attributes import (CargoEstimator, EgoPhoneDetector, HelmetEstimator, LaneSegEstimator,
                                               OrientationEstimator, PhoneEstimator, POVLightEstimator,
                                               TurnSignalTracker)
from traffic_safety_ai.core.config import AppPaths, ModelsConfig, SiteConfig
from traffic_safety_ai.core.ego_motion import CameraMotionMonitor, EgoMotionEstimator
from traffic_safety_ai.core.homography import CalibrationError, GroundCalibration
from traffic_safety_ai.core.kinematics import KinematicsConfig, KinematicsEstimator
from traffic_safety_ai.core.regions import make_tiles, merge_tiled, scene_roi, shift
from traffic_safety_ai.core.tracker import ByteTracker, TrackerConfig
from traffic_safety_ai.core.traffic_light import LightStateTracker
from traffic_safety_ai.core.types import TRACKED_CLASSES, Detection, FrameContext, ViolationEvent
from traffic_safety_ai.evidence.recorder import EvidenceManager
from traffic_safety_ai.evidence.store import Store
from traffic_safety_ai.lpr.reader import PlateReader
from traffic_safety_ai.rules.base_rule import BaseRule, default_corridor
from traffic_safety_ai.rules.catalog import LegalCatalog
from traffic_safety_ai.rules.registry import RuleStatus, build_rules
from traffic_safety_ai.utils.logger import EventLog
from traffic_safety_ai.utils.visualizer import Visualizer

log = logging.getLogger(__name__)
HAZARD_POINT_RULES = {"RB-07", "RB-08", "HZ-02", "HZ-03", "HZ-04", "RB-11"}


class Perception(Protocol):
    def detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]: ...

    def attributes(self, ctx: FrameContext, aux_dets: list[Detection]) -> None: ...

    def capabilities(self) -> set[str]: ...


class YoloPerception:
    """Nhận thức bằng model thật (ModelHub)."""

    def __init__(self, hub: Any, models_cfg: ModelsConfig, scene: SiteConfig):
        self.hub = hub
        self.cfg = models_cfg
        self.scene = scene
        base = hub.get("base")
        if base is None:
            raise RuntimeError("Không nạp được model nền: " + hub.status.get("base", "?"))
        self.base = base
        ch = scene.camera.channel
        self.helmet = HelmetEstimator(hub.get("helmet"), models_cfg.helmet.every_n,
                                      max_hz=models_cfg.helmet.max_hz) if hub.get("helmet") else None
        # model nền chạy trên ẢNH CẮT người lái để tìm điện thoại — chỉ khi model nhận được kích thước ảnh tuỳ ý
        # (ONNX cố định 960×960 chạy bằng CPU thì quá chậm cho việc này)
        base_crops_ok = getattr(base, "task", "") == "detect" and not getattr(base, "fixed_size", False)
        phone_model = hub.get("phone") or (base if base.has_class("phone") and base_crops_ok else None)
        self.phone = PhoneEstimator(hub.get("pose"), phone_model, models_cfg.pose.every_n,
                                    max_hz=models_cfg.pose.max_hz) if ch == "cctv" else None
        self.cargo = CargoEstimator(hub.get("cargo"), use_bgsub=(ch == "cctv")) if ch == "cctv" else None
        self.blink = TurnSignalTracker() if ch == "pov" else None
        self.orient = OrientationEstimator(hub.get("orientation")) if ch == "pov" and hub.get("orientation") else None
        self.lane = LaneSegEstimator(hub.get("lane_seg"), models_cfg.lane_seg.every_n) if ch == "pov" and hub.get("lane_seg") else None
        mount = scene.pov.phone_mount_roi
        self.ego_phone = EgoPhoneDetector(phone_model, 2, mount) if ch == "pov" and phone_model else None
        self.pov_light = POVLightEstimator() if ch == "pov" else None
        self.last_ms = 0.0
        self._regions_for: tuple[int, int] | None = None
        self.roi: tuple[int, int, int, int] | None = None
        self.tiles: list[tuple[int, int, int, int]] = []

    def capabilities(self) -> set[str]:
        return {k for k, (ok, _) in self.hub.capabilities().items() if ok}

    def _plan_regions(self, w: int, h: int) -> None:
        spec = self.cfg.base
        self.roi = scene_roi(self.scene, w, h, spec=spec.roi)
        self.tiles = []
        if spec.tiles:
            cols, rows = (list(spec.tiles) + [1, 1])[:2]
            if cols * rows > 1:
                self.tiles = make_tiles(self.roi or (0, 0, w, h), cols, rows, spec.tile_overlap)
        self._regions_for = (w, h)
        if self.roi is not None or self.tiles:
            log.info("Model nền chạy trong vùng %s%s", self.roi or "cả khung",
                     f" + {len(self.tiles)} ô" if self.tiles else "")

    def detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        if self._regions_for != (w, h):
            self._plan_regions(w, h)
        if self.roi is None and not self.tiles:
            out = self.base.predict([frame])[0].boxes
        else:
            x1, y1, x2, y2 = self.roi or (0, 0, w, h)
            out = shift(self.base.predict([frame[y1:y2, x1:x2]])[0].boxes, x1, y1, w, h)
            if self.tiles:
                crops = [frame[ty1:ty2, tx1:tx2] for tx1, ty1, tx2, ty2 in self.tiles]
                for (tx1, ty1, _, _), o in zip(self.tiles, self.base.predict(crops)):
                    out.extend(shift(o.boxes, tx1, ty1, w, h))
                out = merge_tiled(out)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out

    def attributes(self, ctx: FrameContext, aux_dets: list[Detection]) -> None:
        for est in (self.helmet, self.phone, self.cargo, self.blink, self.orient, self.lane, self.ego_phone):
            if est is not None:
                est.update(ctx)
        if self.pov_light is not None:
            self.pov_light.update(ctx, [d for d in aux_dets if d.cls == "traffic_light"])


@dataclass
class FrameResult:
    ctx: FrameContext
    events: list[ViolationEvent]
    annotated: np.ndarray | None
    vids: list[str] = field(default_factory=list)


class AsyncVideoWriter:
    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        self.q: queue.Queue = queue.Queue(maxsize=64)
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.size = size
        self.t = threading.Thread(target=self._run, name="video-writer", daemon=True)
        self.t.start()

    def _run(self) -> None:
        while True:
            f = self.q.get()
            if f is None:
                break
            if (f.shape[1], f.shape[0]) != self.size:
                f = cv2.resize(f, self.size)
            self.w.write(f)
        self.w.release()

    def write(self, frame: np.ndarray) -> None:
        self.q.put(frame)

    def close(self) -> None:
        self.q.put(None)
        self.t.join(timeout=30)


class Pipeline:
    def __init__(self, scene: SiteConfig, perception: Perception, paths: AppPaths, store: Store,
                 plate_reader: PlateReader | None, source: str, fps: float, start_time: datetime,
                 catalog: LegalCatalog | None = None, only_rules: set[str] | None = None, async_io: bool = True,
                 draw_every: int = 1, extra_capabilities: set[str] | None = None, run_note: str = "",
                 models_desc: dict[str, Any] | None = None, tracker_cfg: dict[str, Any] | None = None):
        self.scene = scene
        self.channel = scene.camera.channel
        self.perception = perception
        self.paths = paths.ensure()
        self.store = store
        self.plates = plate_reader
        self.source = source
        self.fps = max(1.0, float(fps))
        self.start_time = start_time
        self.catalog = catalog or LegalCatalog()
        self.calib: GroundCalibration | None = None
        self.calib_note = "không hiệu chuẩn"
        if scene.calibration.enabled:
            try:
                self.calib = GroundCalibration(scene.calibration.image_points, scene.calibration.world_points,
                                               scene.calibration.max_reproj_error_m)
                self.calib_note = self.calib.summary()
                if not self.calib.ok:
                    log.warning(self.calib_note)
            except CalibrationError as e:
                self.calib_note = f"hiệu chuẩn lỗi: {e}"
                log.error(self.calib_note)
        caps = set(perception.capabilities()) | set(extra_capabilities or ())
        if self.calib is None:
            caps.discard("calibration")
        self.rules, self.rule_report = build_rules(scene, self.catalog, caps, only_rules)
        if self.calib is None and scene.calibration.enabled:
            # cấu hình có khai báo hiệu chuẩn nhưng điểm hỏng → tắt các rule cần đo mét
            for r in list(self.rules):
                need_m = (r.code == "RB-08" and self.channel == "cctv") or (
                    r.code == "RB-10" and not scene.lines.speed_traps)
                if need_m:
                    self.rules.remove(r)
                    self._mark_off(r.code, "hiệu chuẩn mặt đường không hợp lệ")
        tcfg = TrackerConfig()
        overrides = dict(tracker_cfg or getattr(getattr(perception, "cfg", None), "tracker", None) or {})
        for k, v in overrides.items():
            if hasattr(tcfg, k):
                setattr(tcfg, k, type(getattr(tcfg, k))(v))
            else:
                log.warning("Bỏ qua tham số tracker không hợp lệ: %s", k)
        self.tracker = ByteTracker(tcfg, fps=self.fps)
        self.kin = KinematicsEstimator(KinematicsConfig(scale=(self.channel == "pov")))
        self.assoc = RiderAssociator()
        self.ego = EgoMotionEstimator(scene.pov.horizon_y_ratio, scene.pov.camera_height_m, scene.pov.hfov_deg) \
            if self.channel == "pov" else None
        self.lights = {lt.id: LightStateTracker(lt.id, lt.roi, lt.layout) for lt in scene.traffic_lights} \
            if self.channel == "cctv" else {}
        self.cam_monitor = CameraMotionMonitor() if self.channel == "cctv" else None
        self._cam_warned_at = -1e9
        self.cam_moving_frames = 0
        self.counter = LineCounter(scene)
        self.trap = SpeedTrapMeter(scene)
        self.sampler = SpeedSampler()
        self.viz = Visualizer()
        self.draw_every = max(1, draw_every)
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.run_dir = self.paths.runs / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = EventLog(self.run_dir)
        self.evidence = EvidenceManager(
            self.paths, store, scene, self.run_id, source, self.fps,
            plate_provider=(self.plates.result if self.plates else None),
            plate_image_provider=(self.plates.plate_image if self.plates else None),
            async_io=async_io, catalog_warning=self.catalog.warning)
        self.recent_events: deque[ViolationEvent] = deque(maxlen=8)
        self.violators: set[int] = set()
        self._confirmed_ids: set[int] = set()
        self._plates_to_forget: set[int] = set()
        self.frames = 0
        self.t_last_flush = 0.0
        self.proc_fps = 0.0
        self._t_wall = time.perf_counter()
        self.latest_jpeg: bytes | None = None
        self.preview = False            # True: cập nhật ảnh xem trước mỗi draw_every khung (dashboard trực tiếp)
        self.status: dict[str, Any] = {}
        self._bg_saved = False
        self.on_event: Callable[[ViolationEvent, str], None] | None = None
        store.start_run({
            "id": self.run_id, "source": source, "camera_id": scene.camera.id, "camera_name": scene.camera.name,
            "location": scene.camera.location, "channel": self.channel,
            "started_at": datetime.now().isoformat(timespec="seconds"), "frames": 0, "fps": self.fps,
            "config_hash": scene.config_hash(), "models": models_desc or {},
            "rules": [{"code": s.code, "enabled": s.enabled, "reason": s.reason} for s in self.rule_report],
            "notes": f"{self.calib_note}. {run_note}".strip(),
        })
        self.run_note = f"{self.calib_note}. {run_note}".strip()

    def _mark_off(self, code: str, reason: str) -> None:
        for s in self.rule_report:
            if s.code == code:
                s.enabled, s.reason = False, reason

    # ------------------------------------------------------------------ một khung
    def process(self, frame_idx: int, t: float, frame: np.ndarray, render: bool = False) -> FrameResult:
        wall = self.start_time + timedelta(seconds=t)
        if not self._bg_saved:
            cv2.imwrite(str(self.run_dir / "background.jpg"), frame)
            self._bg_saved = True
        dets = self.perception.detect(frame, frame_idx, t)
        tracked = [d for d in dets if d.cls in TRACKED_CLASSES]
        aux = [d for d in dets if d.cls not in TRACKED_CLASSES]
        ego = None
        if self.ego is not None:
            ego = self.ego.update(frame, t, [d.xyxy for d in tracked])
        warp = None if ego is None else ego.warp
        cam_moving = False
        cam_warp = None
        if self.cam_monitor is not None:
            cm = self.cam_monitor.update(frame, t, [d.xyxy for d in tracked])
            cam_moving = cm.recently_moved
            self.cam_moving_frames += int(cam_moving)
            if cm.warp is not None:
                warp = cam_warp = cm.warp      # bù chuyển động camera cho tracker (giữ ổn định ID) và bộ đếm
            if cm.moving_now and t - self._cam_warned_at > 30.0:
                self._cam_warned_at = t
                log.warning("Camera đang rung/xoay (dịch %.1f px, tỉ lệ %.3f) — tạm ngưng các rule dựa trên hướng/"
                            "tốc độ. Nếu xảy ra thường xuyên: gia cố giá đỡ camera.", cm.shift_px, cm.scale)
        to_world = self.calib.point_to_world if self.calib is not None else None
        tracks = self.tracker.update(tracked, frame_idx, t, warp=warp, to_world=to_world)
        # "mới" = lần đầu xuất hiện trong danh sách đã xác nhận (track cần 2 khung mới được xác nhận nên so với
        # tập ID của tracker ở khung trước sẽ luôn rỗng)
        new_ids = {tr.id for tr in tracks if tr.id not in self._confirmed_ids}
        self._confirmed_ids |= new_ids
        for tr in tracks:
            self.kin.update(tr)
        light_states = {lid: lt.update(frame, t) for lid, lt in self.lights.items()}
        persons = [tr for tr in tracks if tr.cls == "person"]
        cycles = [tr for tr in tracks if tr.cls in ("motorcycle", "bicycle")]
        riders, rider_of = self.assoc.update(persons, cycles)
        ctx = FrameContext(
            frame_idx=frame_idx, t=t, wall_time=wall, frame=frame, channel=self.channel, camera_id=self.scene.camera.id,
            tracks=tracks, all_tracks=dict(self.tracker.tracks), scene=self.scene, fps=self.fps, calib=self.calib,
            ego=ego, lights=light_states, riders=riders, rider_of=rider_of, new_track_ids=new_ids,
        )
        ctx.aux["light_trackers"] = self.lights
        ctx.aux["aux_dets"] = aux
        ctx.aux["camera_moving"] = cam_moving
        ctx.aux["camera_warp"] = cam_warp
        if self.channel == "pov":
            w, h = ctx.frame_size
            ctx.aux["corridor"] = default_corridor(w, h, self.scene.pov.ego_corridor, self.scene.pov) \
                if not self.scene.zones.ego_corridor else [tuple(p) for p in self.scene.zones.ego_corridor[0].polygon]
        self.perception.attributes(ctx, aux)

        # biển số: ưu tiên xe đang bị nghi vi phạm
        suspects: set[int] = set(self.violators)
        for r in self.rules:
            suspects |= r.live_suspects
        if self.plates is not None:
            self.plates.process(frame, tracks, frame_idx, t, priority_ids=suspects)

        self.trap.update(ctx)
        self.counter.update(ctx)
        self.sampler.update(ctx)

        events: list[ViolationEvent] = []
        overlays = []
        for r in self.rules:
            try:
                evs = r.evaluate(ctx)
            except Exception:
                log.exception("Rule %s lỗi ở khung %d", r.code, frame_idx)
                continue
            events.extend(evs)
            overlays.extend(r.live_overlays)

        annotated = None
        plate_labels = {}
        if self.plates is not None:
            for tid in suspects:
                res = self.plates.result(tid)
                if res is not None:
                    plate_labels[tid] = res.display
        if events or render or (self.preview and frame_idx % self.draw_every == 0):
            # băng tên vi phạm chỉ hiện ~5 giây rồi tắt (trước đây nằm lì trên video đến hết)
            fresh = [e for e in self.recent_events if 0 <= t - e.t <= 5.0]
            annotated = self.viz.draw_frame(frame, ctx, overlays, fresh + events,
                                            self.proc_fps, suspects, plate_labels)
        vids = []
        for ev in events:
            ev_img = self.viz.draw_frame(frame, ctx, ev.overlays, [ev], None, set(ev.track_ids), plate_labels)
            vid = self.evidence.add_event(ev, ctx, ev_img)
            vids.append(vid)
            self.event_log.write(ev, vid)
            self.recent_events.append(ev)
            if ev.primary_track_id is not None:
                self.violators.add(ev.primary_track_id)
            if ev.rule_code in HAZARD_POINT_RULES:
                tr = ctx.by_id(ev.track_ids[0]) if ev.track_ids else None
                pt = tr.bottom if tr is not None else (None if ev.bbox is None else
                                                       ((ev.bbox[0] + ev.bbox[2]) / 2, ev.bbox[3]))
                if pt is not None:
                    self.store.add_hazard_point(self.run_id, self.scene.camera.id, ev.rule_code,
                                                ev.wall_time.isoformat(timespec="seconds"), float(pt[0]), float(pt[1]))
            if self.on_event:
                self.on_event(ev, vid)
            log.info("%s %s | %s | track %s | %s", vid, ev.rule_code, ev.rule_name, ev.primary_track_id,
                     ev.severity)

        self.evidence.on_frame(ctx)
        self._cleanup(ctx)
        if self._plates_to_forget and self.plates is not None:
            waiting = {c.event.primary_track_id for c in self.evidence.pending}
            for tid in [i for i in self._plates_to_forget if i not in waiting]:
                self.plates.forget(tid)             # hồ sơ đã chốt xong → không giữ ảnh/chuỗi đọc biển trong RAM nữa
                self._plates_to_forget.discard(tid)
        self.frames += 1
        now = time.perf_counter()
        dt = now - self._t_wall
        self._t_wall = now
        if dt > 0:
            inst = 1.0 / dt
            self.proc_fps = inst if self.proc_fps == 0 else 0.9 * self.proc_fps + 0.1 * inst
        if t - self.t_last_flush >= 60.0:
            self.flush_counts()
            self.t_last_flush = t
        if annotated is not None:
            small = annotated if annotated.shape[1] <= 1280 else cv2.resize(
                annotated, (1280, int(annotated.shape[0] * 1280 / annotated.shape[1])))
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.latest_jpeg = buf.tobytes()
        self.status = {"run_id": self.run_id, "frame": frame_idx, "t": round(t, 2), "wall_time": wall.isoformat(timespec="seconds"),
                       "fps": round(self.proc_fps, 1), "tracks": len(tracks), "events": len(self.recent_events),
                       "lights": {k: v.state for k, v in light_states.items()}, "camera_moving": cam_moving}
        return FrameResult(ctx, events, annotated, vids)

    def _cleanup(self, ctx: FrameContext) -> None:
        removed = self.tracker.pop_removed()
        if not removed:
            return
        ids = [tr.id for tr in removed]
        for r in self.rules:
            r.forget_tracks(ids)
        for tr in removed:
            self.counter.forget(tr.id)
            self.trap.forget(tr.id)
            self._confirmed_ids.discard(tr.id)
            self.evidence.track_ended(tr.id)
            s = self.sampler.finish(tr.id)
            if s is not None:
                self.store.add_speed(self.run_id, self.scene.camera.id, tr.id, s[0],
                                     (self.start_time + timedelta(seconds=tr.last_t)).isoformat(timespec="seconds"),
                                     s[1], "homography")
            trap = tr.attrs.get("trap_speed")
            if trap is not None:
                self.store.add_speed(self.run_id, self.scene.camera.id, tr.id, tr.cls,
                                     (self.start_time + timedelta(seconds=tr.last_t)).isoformat(timespec="seconds"),
                                     trap[0], "vach_do")
            # xe đã rời khung: bỏ khỏi danh sách "đang nghi vấn" (nếu không, mỗi khung lại hợp nhất biển số cho MỌI xe
            # từng vi phạm từ đầu buổi → chậm dần khi chạy RTSP nhiều giờ)
            self.violators.discard(tr.id)
            if self.plates is not None:
                pending = any(c.event.primary_track_id == tr.id for c in self.evidence.pending)
                if pending:
                    self._plates_to_forget.add(tr.id)   # hồ sơ còn chờ chốt → giữ biển số tới lúc ghi xong
                else:
                    self.plates.forget(tr.id)           # không giữ biển số của xe đã đi khỏi (kể cả xe không vi phạm)

    def flush_counts(self) -> None:
        rows = self.counter.flush_rows()
        if rows:
            self.store.add_counts(self.run_id, self.scene.camera.id, rows)

    def finish(self) -> None:
        self.counter.finalize()
        self.flush_counts()
        self.evidence.close()
        note = self.run_note
        if self.cam_monitor is not None and self.frames:
            frac = self.cam_moving_frames / self.frames
            note += f"; camera rung/xoay {frac * 100:.0f}% thời gian ({self.cam_monitor.n_moves} lần)"
            if frac > 0.2:
                log.warning("Camera chuyển động %.0f%% thời gian — các rule hướng/tốc độ bị tạm ngưng phần lớn thời "
                            "gian. Camera CCTV cần cố định chắc chắn.", frac * 100)
        self.store.end_run(self.run_id, self.frames, self.proc_fps, note)

    def rules_by_code(self) -> dict[str, BaseRule]:
        return {r.code: r for r in self.rules}

    def report_rows(self) -> list[RuleStatus]:
        return self.rule_report
