"""Thread 1: Vision Processing & Spatial Smoothing (mỗi 1-5 giây).

Khung hình -> YOLO (person) -> điểm giữa-đáy bbox -> Zone 1..N -> đẩy vào 2 bộ đệm trượt
(30 s cho đèn, 10 phút cho điều hòa) -> trao khung hình cho Active Learning (không chặn).

Không có I/O chậm ở đây: ghi ảnh/DB/MQTT đều ở luồng khác. Khi một vòng lỗi thì chỉ vòng đó bị bỏ.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from ares.active_learner import ActiveLearner
from ares.buffers import OccupancyBuffer
from ares.config import AresSettings
from ares.detector import Det, Detector
from ares.model_registry import ModelRegistry
from ares.state import Event, SharedState, VisionSnapshot
from ares.zones import ZoneMap

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    exhausted: bool

    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class CvFrameSource:
    """Camera USB/RTSP, file video hoặc thư mục video — dùng lại reader của hệ thống cũ (realtime/camera.py).

    Camera thật: chờ khung hình MỚI (tối đa `wait_s`), nên vision chạy đúng tốc độ camera khi FPS đích cao hơn
    ("full FPS") và không bao giờ đếm lại cùng một khung hình hai lần.
    FILE video: tua theo thời gian thật đã trôi qua kể từ lần đọc trước (nhịp vision có thể đổi khi train), nếu
    không video 30 fps sẽ chạy chậm hơn đồng hồ hệ thống.
    """

    def __init__(self, cam_id: str, source: str, loop: bool, interval_s: float, wait_s: float = 0.5):
        try:
            from realtime.camera import PlaylistVideoReader, VideoFileReader, build_reader
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Cần chạy từ thư mục gốc dự án yolo-al (để import realtime.camera)") from e
        self.interval_s = interval_s
        self.wait_s = wait_s
        self._last_read: float | None = None
        self.reader = build_reader(cam_id, source, 4, loop=loop, fps_limit=None)
        self.is_file = isinstance(self.reader, (VideoFileReader, PlaylistVideoReader))
        self.reader.start()

    @property
    def exhausted(self) -> bool:
        return bool(getattr(self.reader, "exhausted", False))

    def _video_fps(self) -> float:
        cap = getattr(self.reader, "_cap", None)
        try:
            fps = float(cap.get(5)) if cap is not None else 0.0  # cv2.CAP_PROP_FPS
        except Exception:
            fps = 0.0
        return fps if 1.0 <= fps <= 240.0 else 25.0

    def read(self) -> np.ndarray | None:
        if self.is_file:
            now = time.monotonic()
            elapsed = self.interval_s if self._last_read is None else now - self._last_read
            self._last_read = now
            fps = self._video_fps()
            skip = min(int(fps * 30), max(0, round(fps * elapsed) - 1))
            for _ in range(skip):
                if self.reader.next() is None:
                    break
            f = self.reader.next()
        else:
            f = self.reader.next(timeout_s=self.wait_s)
        return None if f is None else f.image

    def close(self) -> None:
        self.reader.stop()


# ---------------------------------------------------------------------------
# Dựng ảnh minh hoạ cho dashboard
# ---------------------------------------------------------------------------
def render_overlay(image: np.ndarray, dets: list[Det], zonemap: ZoneMap, zone_counts: list[int],
                   cfg: AresSettings) -> bytes:
    import cv2

    h, w = image.shape[:2]
    scale = cfg.overlay_width / w if w > cfg.overlay_width else 1.0
    img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else image.copy()
    zonemap.draw(img, zone_counts)
    for d in dets:
        x1, y1, x2, y2 = (int(v * scale) for v in (d.x1, d.y1, d.x2, d.y2))
        if d.conf < cfg.count_conf_min:
            col = (160, 160, 160)  # không được đếm
        elif cfg.al_conf_low <= d.conf <= cfg.al_conf_high:
            col = (0, 170, 255)  # dải bất định -> ứng viên học chủ động (cam)
        else:
            col = (80, 200, 80)  # chắc chắn (xanh)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.circle(img, ((x1 + x2) // 2, y2), 4, (0, 0, 255), -1)  # điểm giữa-đáy dùng để xếp vùng
        cv2.putText(img, f"{d.conf:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return buf.tobytes() if ok else b""


class VisionTracker:
    def __init__(self, cfg: AresSettings, clock, zonemap: ZoneMap, source: FrameSource, detector: Detector,
                 light_buf: OccupancyBuffer, hvac_buf: OccupancyBuffer, state: SharedState,
                 al: ActiveLearner | None = None, registry: ModelRegistry | None = None,
                 detector_factory: Callable[[Path, str], Detector] | None = None,
                 emit: Callable[[Event], None] | None = None, compute_brightness: bool = True,
                 pca=None, cam_id: str = "main", period_fn: Callable[[], float] | None = None,
                 async_model_load: bool = False):
        self.cfg, self.clock, self.zonemap, self.source, self.detector = cfg, clock, zonemap, source, detector
        self.light_buf, self.hvac_buf, self.state, self.al = light_buf, hvac_buf, state, al
        self.registry, self.detector_factory = registry, detector_factory
        self.emit = emit if emit is not None else (lambda e: None)
        self.compute_brightness = compute_brightness
        self.pca, self.cam_id = pca, cam_id
        self.frames = 0
        self.no_frame_ticks = 0
        self.no_frame_s = 0.0
        # chu kỳ hiện tại giữa 2 nhịp (đổi theo FpsGovernor khi train); dùng để tính "mất camera bao lâu"
        self.period_fn = period_fn or (lambda: cfg.vision_interval_s)
        # Hoán đổi model không ngắt: nạp model mới ở luồng nền, nhịp vision vẫn chạy model cũ tới khi nạp xong.
        self.async_model_load = async_model_load
        self._swap_lock = threading.Lock()
        self._loading_ver: str | None = None
        self._ready: tuple | None = None
        self._fps = 0.0
        self._last_t: float | None = None
        self._masks: dict[tuple[int, int], list[np.ndarray]] = {}
        self._bad_versions: set[str] = set()
        self._warned_stale = False
        self._last_overlay = -1e18
        state.model_version = getattr(detector, "version", "base")

    # ---------------------------------------------------------------- nhịp
    def tick(self) -> bool:
        self._maybe_reload_model()
        img = self.source.read()
        t = self.clock.time()
        if img is None:
            self.no_frame_ticks += 1
            self.no_frame_s += max(0.0, float(self.period_fn()))
            if not self._warned_stale and self.no_frame_s >= self.cfg.frame_stale_s:
                self._warned_stale = True
                self.emit(Event(t, "VISION_STALE", "warn",
                                f"Không đọc được khung hình trong {self.cfg.frame_stale_s:.0f} giây — điều khiển chuyển sang chế độ dự phòng."))
            return False
        if self._warned_stale:
            self._warned_stale = False
            self.emit(Event(t, "VISION_RESTORED", "info", "Camera đã có hình trở lại."))
        self.no_frame_ticks = 0
        self.no_frame_s = 0.0

        dets, ms = self.detector.detect(img)
        counted = [d for d in dets if d.cls == self.cfg.person_class_id and d.conf >= self.cfg.count_conf_min]
        h, w = img.shape[:2]
        zc = self.zonemap.count(counted, w, h)

        t = self.clock.time()
        self.light_buf.push(t, zc.total, zc.zones)
        self.hvac_buf.push(t, zc.total, zc.zones)

        if self._last_t is not None and t > self._last_t:
            inst = 1.0 / (t - self._last_t)
            self._fps = inst if self._fps == 0 else 0.8 * self._fps + 0.2 * inst
        self._last_t = t
        self.frames += 1

        snap = VisionSnapshot(
            t=t, n_total=zc.total, n_zones=list(zc.zones), n_unzoned=zc.unzoned, n_detections=len(dets),
            inference_ms=round(ms, 1), fps=round(self._fps, 3),
            brightness=self._brightness(img) if self.compute_brightness else [],
            model_version=getattr(self.detector, "version", "base"),
        )
        ls, hs = self.light_buf.stats(t), self.hvac_buf.stats(t)
        self.state.set_vision(
            snap,
            {"mean_total": ls.mean_total, "mean_zones": ls.mean_zones, "coverage": ls.coverage},
            {"mean_total": hs.mean_total, "mean_zones": hs.mean_zones, "coverage": hs.coverage},
        )
        self.state.model_version = snap.model_version
        if t - self._last_overlay >= self.cfg.overlay_interval_s:
            self._last_overlay = t
            try:
                self.state.set_frame(render_overlay(img, dets, self.zonemap, zc.zones, self.cfg), t)
            except Exception:
                log.debug("Không dựng được ảnh overlay", exc_info=True)

        if self.al is not None:
            self.al.offer(img, dets, self.clock.now(), camera_id=self.cam_id)  # không chặn; xem ActiveLearner.offer
        if self.pca is not None:
            self.pca.offer_corpus(self.cam_id, img)  # khung mẫu cho PCA (vài ms, rất thưa)
        return True

    # ---------------------------------------------------------------- độ sáng camera (HSV-V) từng vùng
    def _brightness(self, img: np.ndarray) -> list[float | None]:
        import cv2

        h, w = img.shape[:2]
        sw = 160
        sh = max(1, int(h * sw / w))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        v = small.max(axis=2)
        masks = self._masks.get((sw, sh))
        if masks is None:
            masks = []
            for z in self.zonemap.room.zones:
                m = np.zeros((sh, sw), dtype=np.uint8)
                pts = np.array([[int(x * (sw - 1)), int(y * (sh - 1))] for x, y in z.polygon], dtype=np.int32)
                cv2.fillPoly(m, [pts], 1)
                masks.append(m.astype(bool))
            self._masks[(sw, sh)] = masks
        return [round(float(v[m].mean()), 1) if m.any() else None for m in masks]

    # ---------------------------------------------------------------- nạp model mới
    def _maybe_reload_model(self) -> None:
        if self.registry is None or self.detector_factory is None:
            return
        ver, path = self.registry.production()
        cur = getattr(self.detector, "version", "base")
        with self._swap_lock:
            ready, self._ready = self._ready, None
            loading = self._loading_ver
        if ready is not None:
            new, rver, dt = ready
            if rver == ver:  # vẫn đúng bản production (không bị rollback trong lúc nạp)
                self.detector = new
                self.emit(Event(self.clock.time(), "MODEL_SWAPPED", "info",
                                f"Đã hoán đổi model nhận diện {cur} -> {rver} (nạp nền {dt:.1f}s, không ngắt nhận diện)."))
                return
            cur = getattr(self.detector, "version", "base")
        if ver == cur or ver in self._bad_versions or ver == loading:
            return
        if self.async_model_load:
            with self._swap_lock:
                self._loading_ver = ver
            threading.Thread(target=self._load_bg, args=(path, ver, cur), name=f"ares-load-{ver}", daemon=True).start()
            return
        t0 = time.time()
        try:
            new = self.detector_factory(path, ver)
        except Exception as e:  # noqa: BLE001
            self._bad_versions.add(ver)
            log.exception("Không nạp được model %s", ver)
            self.emit(Event(self.clock.time(), "MODEL_LOAD_FAILED", "error",
                            f"Không nạp được model {ver}: {e}. Tiếp tục dùng {cur}."))
            return
        self.detector = new
        self.emit(Event(self.clock.time(), "MODEL_SWAPPED", "info",
                        f"Đã chuyển model nhận diện {cur} -> {ver} ({time.time() - t0:.1f}s)."))

    def _load_bg(self, path: Path, ver: str, cur: str) -> None:
        t0 = time.time()
        try:
            new = self.detector_factory(path, ver)
        except Exception as e:  # noqa: BLE001
            self._bad_versions.add(ver)
            log.exception("Không nạp được model %s", ver)
            self.emit(Event(self.clock.time(), "MODEL_LOAD_FAILED", "error",
                            f"Không nạp được model {ver}: {e}. Tiếp tục dùng {cur}."))
            with self._swap_lock:
                self._loading_ver = None
            return
        with self._swap_lock:
            self._ready = (new, ver, time.time() - t0)
            self._loading_ver = None

    def close(self) -> None:
        try:
            self.source.close()
        except Exception:
            pass
