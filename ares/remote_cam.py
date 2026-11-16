"""Camera của MÁY KHÁC gửi hình qua server: nguồn khung hình "ares://<token>".

Luồng: app ARES Viewer trên máy khác bật webcam → gửi ảnh JPEG lên server (/api/my/cameras/<id>/frame) →
Edge (máy chủ AI) kéo ảnh mới nhất về bằng khoá thiết bị (/api/v1/frame/<token>) → YOLO người + YOLO hành vi chạy
như mọi camera khác → hình đã vẽ + sự kiện gửi lại server → dashboard của đúng người đó.

Kéo ảnh ở luồng nền (không chặn vòng camera của CameraHub); read() chỉ trả ảnh MỚI, không có thì trả None
(quá frame_stale_s giây không có ảnh mới => camera hiện "mất tín hiệu", đúng khi người dùng tắt app).
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.error
import urllib.request

import numpy as np

log = logging.getLogger(__name__)


class AresPullSource:
    exhausted = False

    def __init__(self, cfg, spec, fps: float = 15.0):
        if not cfg.uplink_url or not cfg.uplink_key:
            raise RuntimeError("Camera từ máy khác cần ARES_UPLINK_URL + ARES_UPLINK_KEY (Edge phải nối với server)")
        token = spec.source.split("://", 1)[1].strip("/")
        if not token.replace("-", "").replace("_", "").isalnum():
            raise ValueError("token camera không hợp lệ")
        from ares.uplink import default_edge_id

        self.url = f"{cfg.uplink_url.rstrip('/')}/api/v1/frame/{token}"
        self.key = cfg.uplink_key
        self.edge_id = cfg.uplink_edge_id or default_edge_id()
        self.period = 1.0 / fps
        self._lock = threading.Lock()
        self._img: np.ndarray | None = None
        self._img_t = 0.0
        self._given_t = 0.0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, name=f"ares-remotecam-{spec.id}", daemon=True)
        self._th.start()

    def _fetch(self) -> tuple[bytes, float] | None:
        req = urllib.request.Request(self.url, headers={"Authorization": f"Bearer {self.key}", "X-Edge-Id": self.edge_id})
        with urllib.request.urlopen(req, timeout=3) as r:
            if r.status == 204:
                return None
            return r.read(), float(r.headers.get("X-Frame-T") or time.time())

    def _run(self) -> None:
        import cv2

        last_t = 0.0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                got = self._fetch()
                if got is not None and got[1] > last_t:
                    last_t = got[1]
                    img = cv2.imdecode(np.frombuffer(got[0], np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        with self._lock:
                            self._img, self._img_t = img, last_t
            except (urllib.error.URLError, OSError, ValueError) as e:
                log.debug("Kéo ảnh camera máy khác lỗi: %s", e)
                self._stop.wait(1.0)
            self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._img is None or self._img_t <= self._given_t:
                return None
            self._given_t = self._img_t
            return self._img

    def close(self) -> None:
        self._stop.set()
