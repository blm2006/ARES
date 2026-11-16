"""Thu bằng chứng cho mỗi sự kiện: ảnh trước/tại/sau, ảnh xe, ảnh biển, clip, ảnh ghép, phiếu HTML, JSON + SHA-256.

Luồng xử lý:
  * Vòng đệm giữ ~pre_roll_s giây khung hình gần nhất (JPEG đã thu nhỏ) → clip có cả đoạn TRƯỚC sự kiện.
  * Khi rule phát sự kiện: tạo hồ sơ "chờ" + ghi ngay một dòng CSDL (biển số chưa có).
  * Hồ sơ được chốt khi đủ post_roll_s VÀ (xe đã ra khỏi khung, HOẶC biển số đã đọc chắc chắn, HOẶC hết
    max_plate_wait_s) → lấy kết quả biển số tốt nhất tại thời điểm đó.
  * Ghi file chạy ở luồng nền (không chặn vòng nhận diện).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from traffic_safety_ai.core.config import AppPaths, SiteConfig
from traffic_safety_ai.core.types import VEHICLE_TYPE_VI, FrameContext, ViolationEvent
from traffic_safety_ai.evidence.privacy import apply_privacy_mask, blur_bystanders
from traffic_safety_ai.evidence.report import compose_sheet, record_html, sha256_file
from traffic_safety_ai.evidence.store import Store
from traffic_safety_ai.lpr.fusion import FusedPlate
from traffic_safety_ai.utils.geometry import crop as crop_box
from traffic_safety_ai.utils.text import TextRenderer, default_renderer

log = logging.getLogger(__name__)
RED = (53, 57, 229)
CODECS = [("avc1", ".mp4", True), ("VP80", ".webm", True), ("mp4v", ".mp4", False)]


@dataclass
class Case:
    vid: str
    event: ViolationEvent
    record: dict[str, Any]
    folder: Path
    snapshot: np.ndarray
    annotated: np.ndarray
    pre: list[tuple[int, float, bytes]]
    post: list[tuple[int, float, bytes]] = field(default_factory=list)
    boxes: dict[int, np.ndarray] = field(default_factory=dict)       # frame_idx → hộp xe chính (toạ độ gốc)
    others: list[dict[str, Any]] = field(default_factory=list)
    involved: list[dict[str, Any]] = field(default_factory=list)
    track_gone: bool = False
    vehicle_crop: np.ndarray | None = None


class FrameRing:
    def __init__(self, seconds: float, fps: float, max_width: int = 1280, quality: int = 85,
                 clip_fps: float | None = None, privacy_polygons: list | None = None):
        self.fps = max(1.0, fps)
        want = min(self.fps, clip_fps or min(15.0, self.fps))
        self.step = max(1, int(round(self.fps / want)))
        # tốc độ khung THỰC của clip sau khi lấy 1 trên `step` khung (25 fps, muốn 10 → step 2 → 12,5 fps). Ghi clip
        # với đúng con số này, nếu không clip phát nhanh/chậm hơn thực tế và thời lượng lệch.
        self.clip_fps = self.fps / self.step
        self.buf: deque[tuple[int, float, bytes]] = deque(maxlen=max(2, int(seconds * self.clip_fps) + 2))
        self.max_width = max_width
        self.quality = quality
        self.scale = 1.0
        self.privacy_polygons = privacy_polygons or []

    def encode(self, frame: np.ndarray) -> bytes:
        # vùng riêng tư cố định (cửa sổ nhà dân...) phải bị che ở MỌI sản phẩm: cả ảnh trước/sau và clip
        frame = apply_privacy_mask(frame, self.privacy_polygons)
        h, w = frame.shape[:2]
        self.scale = min(1.0, self.max_width / float(w))
        img = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_AREA) \
            if self.scale < 1.0 else frame
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else b""

    def push(self, frame_idx: int, t: float, frame: np.ndarray) -> tuple[int, float, bytes] | None:
        if frame_idx % self.step:
            return None
        item = (frame_idx, t, self.encode(frame))
        self.buf.append(item)
        return item


class EvidenceManager:
    def __init__(self, paths: AppPaths, store: Store, scene: SiteConfig, run_id: str, source: str, fps: float,
                 plate_provider: Callable[[int], FusedPlate | None] | None = None,
                 plate_image_provider: Callable[[int], np.ndarray | None] | None = None,
                 text: TextRenderer | None = None, async_io: bool = True, catalog_warning: str = ""):
        self.paths = paths.ensure()
        self.store = store
        self.scene = scene
        self.run_id = run_id
        self.source = source
        self.ev_cfg = scene.evidence
        self.ring = FrameRing(self.ev_cfg.pre_roll_s, fps, self.ev_cfg.clip_max_width, 85, self.ev_cfg.clip_fps,
                              [z.polygon for z in scene.zones.privacy_mask])
        self.plate_provider = plate_provider or (lambda tid: None)
        self.plate_image_provider = plate_image_provider or (lambda tid: None)
        self.text = text or default_renderer()
        self.catalog_warning = catalog_warning
        self.pending: list[Case] = []
        self.done_ids: list[str] = []
        self.async_io = async_io
        self._q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        if async_io:
            self._worker = threading.Thread(target=self._run, name="evidence-writer", daemon=True)
            self._worker.start()

    # ------------------------------------------------------------------ vòng khung
    def on_frame(self, ctx: FrameContext) -> None:
        item = self.ring.push(ctx.frame_idx, ctx.t, ctx.frame)
        live = {t.id: t for t in ctx.tracks}
        for c in list(self.pending):
            if item is not None and ctx.t - c.event.t <= self.ev_cfg.post_roll_s + 1e-6:
                c.post.append(item)
            pid = c.event.primary_track_id
            if pid is not None:
                tr = live.get(pid)
                if tr is not None:
                    c.boxes[ctx.frame_idx] = tr.xyxy.copy()
                elif ctx.all_tracks.get(pid) is None:
                    c.track_gone = True
            if self._due(c, ctx.t):
                self.pending.remove(c)
                self._finalize(c)

    def track_ended(self, track_id: int) -> None:
        for c in self.pending:
            if c.event.primary_track_id == track_id:
                c.track_gone = True

    def _due(self, c: Case, t: float) -> bool:
        age = t - c.event.t
        if age < self.ev_cfg.post_roll_s:
            return False
        if age >= max(self.ev_cfg.post_roll_s, self.ev_cfg.max_plate_wait_s):
            return True
        pid = c.event.primary_track_id
        if pid is None or c.track_gone:
            return True
        res = self.plate_provider(pid)
        return res is not None and res.valid and res.confidence >= 0.85 and res.n_agree >= 4

    # ------------------------------------------------------------------ sự kiện
    def add_event(self, ev: ViolationEvent, ctx: FrameContext, annotated: np.ndarray) -> str:
        vid = self.store.next_violation_id(ev.wall_time, ctx.camera_id)
        day = ev.wall_time.strftime("%Y-%m-%d")
        folder = self.paths.evidence / day / vid
        folder.mkdir(parents=True, exist_ok=True)
        rec = self._base_record(vid, ev, folder)
        self.store.insert_violation(rec)
        involved_ids = set(ev.track_ids)
        others, involved = [], []
        for t in ctx.tracks:
            d = {"id": t.id, "cls": t.cls, "xyxy": t.xyxy.tolist(), "plate_box": t.attrs.get("plate_box")}
            (involved if t.id in involved_ids else others).append(d)
        boxes = {}
        vcrop = None
        masked = apply_privacy_mask(ctx.frame, self.ring.privacy_polygons)   # ảnh cắt xe cũng không lộ vùng riêng tư
        if ev.primary_track_id is not None:
            tr = ctx.by_id(ev.primary_track_id)
            if tr is not None:
                for p in tr.history:
                    boxes[p.frame_idx] = p.xyxy.copy()
                vcrop, _ = crop_box(masked, tr.xyxy, pad=0.12)
                vcrop = vcrop.copy()
        elif ev.bbox is not None:
            vcrop, _ = crop_box(masked, ev.bbox, pad=0.15)
            vcrop = vcrop.copy()
        case = Case(vid, ev, rec, folder, ctx.frame.copy(), annotated, list(self.ring.buf), boxes=boxes,
                    others=others, involved=involved, vehicle_crop=vcrop)
        self.pending.append(case)
        return vid

    def _base_record(self, vid: str, ev: ViolationEvent, folder: Path) -> dict[str, Any]:
        cam = self.scene.camera
        legal = ev.legal or {}
        return {
            "id": vid, "rule_code": ev.rule_code, "rule_name": ev.rule_name, "variant": ev.variant,
            "severity": ev.severity, "channel": ev.channel, "camera_id": cam.id, "camera_name": cam.name,
            "location": cam.location, "occurred_at": ev.wall_time.isoformat(timespec="seconds"),
            "frame_idx": ev.frame_idx, "video_t": round(ev.t, 3), "vehicle_cls": ev.vehicle_cls,
            "vehicle_label": VEHICLE_TYPE_VI.get(ev.vehicle_cls or "", "—" if ev.vehicle_cls is None else ev.vehicle_cls),
            "behavior": legal.get("behavior") or ev.rule_name, "legal_ref": legal.get("ref"),
            "fine": legal.get("fine"), "points": legal.get("points"), "legal_verified": int(bool(legal.get("verified"))),
            "ai_confidence": round(float(ev.confidence), 3), "measurements": ev.measurements, "notes": ev.notes,
            "status": "cho_xac_minh", "evidence_dir": str(folder), "files": {}, "run_id": self.run_id,
            "source": self.source,
        }

    # ------------------------------------------------------------------ chốt hồ sơ
    def flush(self) -> None:
        for c in list(self.pending):
            self.pending.remove(c)
            self._finalize(c)
        if self.async_io and self._worker is not None:
            self._q.join()

    def close(self) -> None:
        self.flush()
        if self.async_io and self._worker is not None:
            self._q.put(None)
            self._worker.join(timeout=10)
            self._worker = None

    def _finalize(self, c: Case) -> None:
        pid = c.event.primary_track_id
        plate: FusedPlate | None = self.plate_provider(pid) if pid is not None else None
        plate_img = None
        if plate is not None and plate.crop is not None:
            plate_img = plate.crop
        elif pid is not None:
            plate_img = self.plate_image_provider(pid)
        job = (c, plate, None if plate_img is None else plate_img.copy())
        if self.async_io:
            self._q.put(job)
        else:
            self._write(*job)

    def _run(self) -> None:
        while True:
            job = self._q.get()
            try:
                if job is None:
                    return
                self._write(*job)
            except Exception:  # pragma: no cover - ghi log, không làm sập luồng
                log.exception("Lỗi ghi bằng chứng")
            finally:
                self._q.task_done()

    def _frame_near(self, frames: list[tuple[int, float, bytes]], t: float) -> tuple[np.ndarray | None, int | None]:
        if not frames:
            return None, None
        fi, _, jpg = min(frames, key=lambda x: abs(x[1] - t))
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        return img, fi

    def _draw_box(self, img: np.ndarray | None, c: Case, fi: int | None) -> np.ndarray | None:
        if img is None or fi is None:
            return img
        b = c.boxes.get(fi)
        if b is None and c.boxes:
            near = min(c.boxes, key=lambda k: abs(k - fi))
            if abs(near - fi) <= 3:
                b = c.boxes[near]
        if b is not None:
            s = img.shape[1] / float(c.snapshot.shape[1])
            x1, y1, x2, y2 = (int(v * s) for v in b)
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 3)
        return img

    def _write(self, c: Case, plate: FusedPlate | None, plate_img: np.ndarray | None) -> None:
        q = [cv2.IMWRITE_JPEG_QUALITY, int(self.ev_cfg.jpeg_quality)]
        f = c.folder
        mask = [z.polygon for z in self.scene.zones.privacy_mask]
        snapshot = apply_privacy_mask(c.snapshot, mask)
        cv2.imwrite(str(f / "snapshot_raw.jpg"), snapshot, [cv2.IMWRITE_JPEG_QUALITY, 95])
        snap_sha = sha256_file(f / "snapshot_raw.jpg")
        overview = apply_privacy_mask(c.annotated, mask)
        cv2.imwrite(str(f / "overview.jpg"), overview, q)
        before, fb = self._frame_near(c.pre, c.event.t - 1.0)
        after, fa = self._frame_near(c.post, c.event.t + 1.0)
        before = self._draw_box(before, c, fb)
        after = self._draw_box(after, c, fa)
        if before is not None:
            cv2.imwrite(str(f / "before.jpg"), before, q)
        if after is not None:
            cv2.imwrite(str(f / "after.jpg"), after, q)
        if c.vehicle_crop is not None and c.vehicle_crop.size:
            cv2.imwrite(str(f / "vehicle.jpg"), c.vehicle_crop, q)
        if plate_img is not None and plate_img.size:
            cv2.imwrite(str(f / "plate.jpg"), plate_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rec = dict(c.record)
        if plate is not None:
            rec.update({"plate_text": plate.text, "plate_display": plate.display, "plate_conf": plate.confidence,
                        "plate_color": plate.color, "plate_color_label": plate.color_label,
                        "plate_valid": int(plate.valid), "province": plate.province, "plate_reads": plate.n_reads,
                        "plate_candidates": [list(x) for x in plate.candidates]})
        rec["detect_unit"] = self.scene.camera.detect_unit
        rec["handling_note"] = self.scene.camera.handling_note
        sheet = compose_sheet(rec, overview, before, after, c.vehicle_crop, plate_img, snap_sha, self.text)
        cv2.imwrite(str(f / "composite.jpg"), sheet, q)
        if self.scene.privacy.blur_bystanders_in_export:
            pub_over = blur_bystanders(overview, c.others, self.scene.privacy.blur_all_faces_in_export, c.involved)
            pub = compose_sheet(rec, pub_over, before, after, c.vehicle_crop, plate_img, snap_sha, self.text)
            cv2.imwrite(str(f / "composite_public.jpg"), pub, q)
        if self.ev_cfg.save_clip:
            self._write_clip(c, f)
        files = {}
        for p in sorted(f.iterdir()):
            if p.is_file() and p.name not in ("record.json", "record.html"):
                files[p.name] = {"sha256": sha256_file(p), "size": p.stat().st_size}
        rec["files"] = files
        db_fields = {k: v for k, v in rec.items() if k in (
            "plate_text", "plate_display", "plate_conf", "plate_color", "plate_color_label", "plate_valid", "province",
            "plate_reads", "plate_candidates", "files")}
        self.store.update_violation(c.vid, **db_fields)
        (f / "record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (f / "record.html").write_text(record_html(rec, f, self.catalog_warning), encoding="utf-8")
        self.done_ids.append(c.vid)

    def _write_clip(self, c: Case, folder: Path) -> None:
        frames = c.pre + [x for x in c.post if x[0] > (c.pre[-1][0] if c.pre else -1)]
        if len(frames) < 2:
            return
        first = cv2.imdecode(np.frombuffer(frames[0][2], np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            return
        h, w = first.shape[:2]
        fps = self.ring.clip_fps
        for fourcc, ext, playable in CODECS:
            path = folder / f"clip{ext}"
            try:
                wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            except Exception:
                continue
            if not wr.isOpened():
                wr.release()
                if path.exists():
                    path.unlink(missing_ok=True)
                continue
            for fi, _, jpg in frames:
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = self._draw_box(img, c, fi)
                if abs(fi - c.event.frame_idx) <= self.ring.step:
                    cv2.rectangle(img, (0, 0), (w - 1, h - 1), RED, 6)
                wr.write(img)
            wr.release()
            if path.exists() and path.stat().st_size > 2048:     # bộ mã hoá hỏng đôi khi chỉ ghi phần đầu tệp
                (folder / "clip_info.json").write_text(json.dumps({"codec": fourcc, "browser_playable": playable,
                                                                   "fps": fps, "frames": len(frames)}),
                                                       encoding="utf-8")
                return
            path.unlink(missing_ok=True)
