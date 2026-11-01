"""Nhiều camera cho dashboard kiểu CCTV: dò camera, đếm người từng camera, dựng ảnh có khung người.

Phân vai:
  * camera ĐIỀU KHIỂN ("main"): đã do VisionTracker sở hữu (đếm theo vùng -> đèn/điều hòa). Hub KHÔNG mở lại thiết bị
    (camera USB không mở được hai lần) mà chỉ phản chiếu ảnh + số người từ SharedState.
  * camera PHỤ: do hub mở, đếm người tổng (không chia vùng, không điều khiển thiết bị), cũng góp ảnh cho học chủ động
    và mẫu cho PCA. Cắm thêm camera USB là tự hiện thêm (dò chỉ số 0..N-1), hoặc khai báo RTSP trong ares_config/cameras.json.

Một luồng suy luận duy nhất chạy vòng tròn qua các camera phụ (một model, không nhân bản VRAM). 16 camera trên CPU
chậm hơn 1 giây/vòng; có GPU thì nhanh hơn nhiều. Camera mất tín hiệu không làm chết vòng: chỉ đánh dấu offline.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ares.config import AresSettings
from ares.detector import Det
from ares.state import Event, SharedState

log = logging.getLogger(__name__)

MAIN_ID = "main"
_ID_BAD = re.compile(r"[^A-Za-z0-9]+")


def sanitize_id(raw: str, fallback: str) -> str:
    s = _ID_BAD.sub("", str(raw))[:10]
    return s or fallback


def mask_source(src: str) -> str:
    """Không bao giờ lộ mật khẩu RTSP ra dashboard."""
    return re.sub(r"(://)[^/@]*@", r"\1***@", str(src))


def grid_dims(n: int) -> tuple[int, int]:
    """(cột, hàng) cho n camera: 1 -> 1x1 toàn màn hình, 2 -> 2x1, 4 -> 2x2, 6 -> 3x2, 9 -> 3x3, 16 -> 4x4."""
    if n <= 1:
        return 1, 1
    cols = math.ceil(math.sqrt(n))
    return cols, math.ceil(n / cols)


@dataclass
class CameraSpec:
    id: str
    name: str
    source: str
    kind: str = "usb"  # usb | rtsp | file | sim

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "source": mask_source(self.source), "kind": self.kind}


@dataclass
class CamStatus:
    online: bool = False
    n_people: int = 0
    n_boxes: int = 0
    fps: float = 0.0
    t: float = 0.0  # lần cuối có kết quả đếm
    frames: int = 0
    lost_ticks: int = 0
    lost_s: float = 0.0
    error: str = ""
    jpeg: bytes = b""


@dataclass
class _Cam:
    spec: CameraSpec
    source: object
    detector: object | None = None
    st: CamStatus = field(default_factory=CamStatus)
    last_infer: float = 0.0


def guess_kind(source: str) -> str:
    s = str(source).strip().lower()
    if s.isdigit():
        return "usb"
    if s.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://")):
        return "rtsp"
    return "file"


def load_extra_specs(cfg: AresSettings) -> tuple[list[CameraSpec], bool]:
    """Đọc ares_config/cameras.json -> (camera phụ khai báo tay, có tự dò USB không)."""
    auto = cfg.camera_auto_detect
    path = cfg.cameras_file
    specs: list[CameraSpec] = []
    if path and Path(path).exists():
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("Không đọc được %s: %s — bỏ qua", path, e)
            return [], auto
        auto = bool(raw.get("auto_usb", auto))
        used = {MAIN_ID}
        for i, c in enumerate(raw.get("cameras", []), start=1):
            src = str(c.get("source", "")).strip()
            if not src:
                continue
            cid = sanitize_id(c.get("id", ""), f"cam{i}")
            while cid in used:
                cid = f"{cid[:8]}{i}"
            used.add(cid)
            specs.append(CameraSpec(cid, str(c.get("name") or f"Camera {i}"), src, guess_kind(src)))
    return specs, auto


def probe_usb(max_n: int, skip: set[int]) -> list[int]:
    """Chỉ số camera USB đang trả được khung hình. Mở rồi nhả ngay."""
    import cv2

    found: list[int] = []
    for i in range(max_n):
        if i in skip:
            continue
        cap = None
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append(i)
        except Exception:
            log.debug("Dò camera %d lỗi", i, exc_info=True)
        finally:
            if cap is not None:
                cap.release()
    return found


def render_boxes(image: np.ndarray, dets: list[Det], cfg: AresSettings) -> bytes:
    import cv2

    h, w = image.shape[:2]
    scale = cfg.hub_jpeg_width / w if w > cfg.hub_jpeg_width else 1.0
    img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else image.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v * scale) for v in (d.x1, d.y1, d.x2, d.y2))
        if d.conf < cfg.count_conf_min:
            col = (160, 160, 160)
        elif cfg.al_conf_low <= d.conf <= cfg.al_conf_high:
            col = (0, 170, 255)
        else:
            col = (80, 200, 80)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return buf.tobytes() if ok else b""


class CameraHub:
    def __init__(self, cfg: AresSettings, clock, state: SharedState, *, control_name: str = "",
                 source_factory: Callable[[CameraSpec], object] | None = None,
                 detector: object | None = None, detector_for: Callable[[CameraSpec], object] | None = None,
                 registry=None, detector_factory: Callable | None = None, al=None, pca=None,
                 emit: Callable[[Event], None] | None = None):
        self.cfg, self.clock, self.state = cfg, clock, state
        self.control_name = control_name or cfg.camera_name or f"Phòng {cfg.room_id}"
        self.source_factory = source_factory or self._default_source
        self.detector, self.detector_for = detector, detector_for
        self.registry, self.detector_factory = registry, detector_factory
        self.al, self.pca = al, pca
        self.emit = emit if emit is not None else (lambda e: None)
        self._cams: dict[str, _Cam] = {}
        self._lock = threading.RLock()
        self._bad_versions: set[str] = set()
        self._scan_thread: threading.Thread | None = None
        self.period_fn: Callable[[], float] = lambda: self.cfg.hub_interval_s  # runtime gắn FpsGovernor.hub_period
        self.scanning = False
        self.last_scan: dict | None = None

    # ------------------------------------------------------------------ mở nguồn
    def _default_source(self, spec: CameraSpec):
        from ares.vision_tracker import CvFrameSource

        return CvFrameSource(spec.id, spec.source, self.cfg.camera_loop, self.cfg.hub_interval_s)

    # ------------------------------------------------------------------ quản lý danh sách camera phụ
    def add(self, spec: CameraSpec) -> bool:
        with self._lock:
            if spec.id in self._cams or spec.id == MAIN_ID:
                return False
            try:
                src = self.source_factory(spec)
            except Exception as e:  # noqa: BLE001
                log.warning("Không mở được camera %s (%s): %s", spec.id, mask_source(spec.source), e)
                self.emit(Event(self.clock.time(), "CAMERA_OPEN_FAILED", "warn",
                                f"Không mở được camera '{spec.name}' ({mask_source(spec.source)}): {e}"))
                return False
            det = self.detector_for(spec) if self.detector_for else None
            self._cams[spec.id] = _Cam(spec, src, det)
        self.emit(Event(self.clock.time(), "CAMERA_ADDED", "info", f"Đã thêm camera '{spec.name}' ({spec.kind})."))
        return True

    def remove(self, cam_id: str) -> None:
        with self._lock:
            c = self._cams.pop(cam_id, None)
        if c is not None:
            try:
                c.source.close()
            except Exception:
                pass

    def specs(self) -> list[CameraSpec]:
        with self._lock:
            return [c.spec for c in self._cams.values()]

    def rescan(self, background: bool = True) -> None:
        """Nạp lại cameras.json + dò camera USB mới. Dò camera có thể mất vài giây nên mặc định chạy nền."""
        if self.scanning:
            return
        if not background:
            self._scan()
            return
        self._scan_thread = threading.Thread(target=self._scan, name="ares-camscan", daemon=True)
        self._scan_thread.start()

    def _scan(self) -> None:
        self.scanning = True
        added: list[str] = []
        try:
            extra, auto = load_extra_specs(self.cfg)
            for sp in extra:
                if sp.id not in {c.spec.id for c in self._cams.values()} and self.add(sp):
                    added.append(sp.id)
            if auto and self.cfg.camera_probe_max > 0:
                skip = {int(self.cfg.camera_source)} if str(self.cfg.camera_source).strip().isdigit() else set()
                for c in list(self._cams.values()):
                    if c.spec.kind == "usb" and c.spec.source.isdigit():
                        skip.add(int(c.spec.source))
                for i in probe_usb(self.cfg.camera_probe_max, skip):
                    sp = CameraSpec(sanitize_id(f"usb{i}", f"usb{i}"), f"Camera USB {i}", str(i), "usb")
                    if self.add(sp):
                        added.append(sp.id)
            self.last_scan = {"added": added, "at": self.clock.now().isoformat(timespec="seconds")}
        except Exception:
            log.exception("Dò camera lỗi")
        finally:
            self.scanning = False

    # ------------------------------------------------------------------ nạp model mới (camera dùng chung model thật)
    def _maybe_reload_model(self) -> None:
        if self.detector is None and self._cams and self.detector_for is None \
                and self.registry is not None and self.detector_factory is not None:
            ver, path = self.registry.production()  # nạp chậm: chỉ khi thật sự có camera phụ
            try:
                self.detector = self.detector_factory(path, ver)
            except Exception:  # noqa: BLE001
                log.exception("Hub không nạp được model %s", ver)
                self.emit(Event(self.clock.time(), "MODEL_LOAD_FAILED", "error", f"Camera phụ không nạp được model {ver}."))
            return
        if self.registry is None or self.detector_factory is None or self.detector is None:
            return
        ver, path = self.registry.production()
        cur = getattr(self.detector, "version", "base")
        if ver == cur or ver in self._bad_versions:
            return
        try:
            self.detector = self.detector_factory(path, ver)
        except Exception:  # noqa: BLE001 - VisionTracker đã báo lỗi nạp model; ở đây chỉ không thử lại
            self._bad_versions.add(ver)
            log.warning("Hub không nạp được model %s", ver, exc_info=True)

    # ------------------------------------------------------------------ một vòng đếm
    def tick(self) -> int:
        """Đọc + đếm người ở mọi camera phụ một lượt. Trả số camera đã xử lý."""
        self._maybe_reload_model()
        with self._lock:
            cams = list(self._cams.values())
        done = 0
        for cam in cams:
            try:
                done += 1 if self._tick_one(cam) else 0
            except Exception as e:  # noqa: BLE001 - một camera lỗi không được làm hỏng cả vòng
                cam.st.error = f"{type(e).__name__}: {e}"
                log.exception("Camera %s lỗi", cam.spec.id)
        return done

    def _tick_one(self, cam: _Cam) -> bool:
        cfg, st = self.cfg, cam.st
        img = cam.source.read()
        t = self.clock.time()
        if img is None:
            st.lost_ticks += 1
            st.lost_s += max(0.0, float(self.period_fn()))
            if st.online and st.lost_s >= cfg.frame_stale_s:
                st.online = False
                self.emit(Event(t, "CAMERA_LOST", "warn", f"Camera '{cam.spec.name}' mất tín hiệu."))
            return False
        if not st.online and st.frames > 0:
            self.emit(Event(t, "CAMERA_RESTORED", "info", f"Camera '{cam.spec.name}' đã có hình trở lại."))
        st.online, st.lost_ticks, st.lost_s, st.error = True, 0, 0.0, ""
        det = cam.detector or self.detector
        if det is None:
            st.error = "Chưa có model nhận diện"
            return False
        dets, _ms = det.detect(img)
        counted = [d for d in dets if d.cls == cfg.person_class_id and d.conf >= cfg.count_conf_min]
        if st.t:
            dt = t - st.t
            if dt > 0:
                inst = 1.0 / dt
                st.fps = inst if st.fps == 0 else 0.7 * st.fps + 0.3 * inst
        st.t, st.n_people, st.n_boxes = t, len(counted), len(dets)
        st.frames += 1
        try:
            st.jpeg = render_boxes(img, dets, cfg)
        except Exception:
            log.debug("Không dựng được ảnh camera %s", cam.spec.id, exc_info=True)
        if self.al is not None:
            self.al.offer(img, dets, self.clock.now(), camera_id=cam.spec.id)
        if self.pca is not None:
            self.pca.offer_corpus(cam.spec.id, img)
        return True

    # ------------------------------------------------------------------ cho dashboard
    def _main_entry(self) -> dict:
        v = self.state.get_vision()
        jpg, ft = self.state.get_frame()
        now = self.clock.time()
        online = bool(v is not None and now - v.t <= max(self.cfg.frame_stale_s, 3 * self.cfg.vision_interval_s))
        return {"id": MAIN_ID, "name": self.control_name, "control": True, "kind": guess_kind(self.cfg.camera_source),
                "source": mask_source(self.cfg.camera_source), "online": online,
                "n_people": v.n_total if v is not None else 0, "n_boxes": v.n_detections if v is not None else 0,
                "zones": list(v.n_zones) if v is not None else [], "fps": round(v.fps, 2) if v is not None else 0.0,
                "age_s": round(now - ft, 1) if jpg else None, "has_frame": bool(jpg), "error": ""}

    def overview(self) -> dict:
        now = self.clock.time()
        out = [self._main_entry()]
        with self._lock:
            cams = list(self._cams.values())
        for c in cams:
            st = c.st
            out.append({"id": c.spec.id, "name": c.spec.name, "control": False, "kind": c.spec.kind,
                        "source": mask_source(c.spec.source), "online": st.online, "n_people": st.n_people,
                        "n_boxes": st.n_boxes, "zones": [], "fps": round(st.fps, 2),
                        "age_s": round(now - st.t, 1) if st.t else None, "has_frame": bool(st.jpeg), "error": st.error})
        cols, rows = grid_dims(len(out))
        return {"cameras": out, "count": len(out), "online": sum(1 for c in out if c["online"]),
                "total_people": sum(c["n_people"] for c in out if c["online"]),
                "grid": {"cols": cols, "rows": rows}, "scanning": self.scanning, "last_scan": self.last_scan}

    def frame(self, cam_id: str) -> tuple[bytes | None, float]:
        if cam_id == MAIN_ID:
            return self.state.get_frame()
        with self._lock:
            c = self._cams.get(cam_id)
        if c is None:
            raise KeyError(cam_id)
        return (c.st.jpeg or None), c.st.t

    def close(self) -> None:
        with self._lock:
            ids = list(self._cams)
        for i in ids:
            self.remove(i)
