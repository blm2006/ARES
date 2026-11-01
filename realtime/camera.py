"""Frame source + ring buffer với DROP POLICY TƯỜNG MINH (spec 1.2).

Buffer đầy -> drop frame CŨ NHẤT và tăng counter. Không im lặng, không block
thread đọc. Counter đi vào telemetry, không đi vào dataset pipeline.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    cam_id: str
    frame_idx: int
    ts: datetime
    image: np.ndarray  # BGR, ĐỘ PHÂN GIẢI GỐC

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


class CameraReader:
    """Đọc camera trong thread riêng, đẩy vào ring buffer.

    Thread đọc không bao giờ chờ consumer: nếu consumer chậm, frame cũ bị drop.
    Đây là lựa chọn có chủ ý — realtime ưu tiên độ trễ thấp hơn tính đầy đủ.
    """

    def __init__(
        self,
        cam_id: str,
        source: str | int,
        buffer_size: int = 8,
        reconnect_delay_s: float = 2.0,
        use_gstreamer: bool = False,
    ):
        self.cam_id = cam_id
        self.source = int(source) if str(source).isdigit() else source
        self.buffer: deque[Frame] = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.reconnect_delay_s = reconnect_delay_s
        self.use_gstreamer = use_gstreamer

        self.dropped = 0
        self.read = 0
        self._frame_idx = 0
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._on_drop = None  # callback(n)

    exhausted = False  # nguồn hữu hạn (video) set True khi đọc hết

    def set_drop_callback(self, fn) -> None:
        self._on_drop = fn

    def _open(self) -> cv2.VideoCapture:
        if self.use_gstreamer and isinstance(self.source, str):
            cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(self.source)
        # Giảm buffer nội bộ của driver để không cộng thêm độ trễ ẩn.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def start(self) -> "CameraReader":
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.cam_id}", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    log.warning("[%s] mở camera thất bại, thử lại sau %.1fs", self.cam_id, self.reconnect_delay_s)
                    time.sleep(self.reconnect_delay_s)
                    continue
                log.info("[%s] camera mở OK", self.cam_id)

            ok, img = self._cap.read()
            if not ok or img is None:
                log.warning("[%s] mất stream, reconnect", self.cam_id)
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay_s)
                continue

            self._frame_idx += 1
            self.read += 1
            frame = Frame(
                cam_id=self.cam_id,
                frame_idx=self._frame_idx,
                ts=datetime.now(timezone.utc),
                image=img,
            )
            with self._lock:
                if len(self.buffer) == self.buffer_size:
                    # DROP POLICY: bỏ frame cũ nhất, đếm, báo telemetry.
                    self.buffer.popleft()
                    self.dropped += 1
                    if self._on_drop:
                        self._on_drop(1)
                self.buffer.append(frame)

    def latest(self) -> Frame | None:
        """Lấy frame mới nhất; các frame cũ hơn trong buffer bị bỏ (và tính là drop)."""
        with self._lock:
            if not self.buffer:
                return None
            n_skipped = len(self.buffer) - 1
            if n_skipped > 0:
                self.dropped += n_skipped
                if self._on_drop:
                    self._on_drop(n_skipped)
            frame = self.buffer[-1]
            self.buffer.clear()
            return frame

    def next(self, timeout_s: float = 1.0) -> Frame | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            f = self.latest()
            if f is not None:
                return f
            time.sleep(0.002)
        return None

    @property
    def drop_rate(self) -> float:
        total = self.read
        return self.dropped / total if total else 0.0

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()


class VideoFileReader(CameraReader):
    """Đọc file video — dùng để test pipeline không cần camera thật.

    Khác CameraReader ở một điểm quan trọng: KHÔNG drop frame, vì với file thì
    drop là làm sai kết quả chứ không phải giữ realtime.
    """

    def __init__(self, cam_id: str, path: str, loop: bool = False, fps_limit: float | None = None):
        super().__init__(cam_id, path, buffer_size=1)
        self.loop = loop
        self.fps_limit = fps_limit
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(path)

    def start(self) -> "VideoFileReader":
        return self  # đọc đồng bộ

    def next(self, timeout_s: float = 1.0) -> Frame | None:
        if self.fps_limit:
            time.sleep(1.0 / self.fps_limit)
        ok, img = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self._cap.read()
            if not ok:
                self.exhausted = True
                return None
        self._frame_idx += 1
        self.read += 1
        return Frame(self.cam_id, self._frame_idx, datetime.now(timezone.utc), img)

    def latest(self) -> Frame | None:
        return self.next()


VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".mpg", ".mpeg")


def is_video_source(source: str) -> bool:
    """True nếu source là đường dẫn file video (theo đuôi)."""
    return str(source).lower().endswith(VIDEO_EXTS)


def _natural_key(name: str):
    """Sort tự nhiên: 'vid2' đứng trước 'vid10' (không phải sort chuỗi thô)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_videos_in_dir(folder: str) -> list[str]:
    """Tất cả file video trong thư mục, sắp xếp theo TÊN (thứ tự tự nhiên)."""
    try:
        names = [f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTS)]
    except OSError:
        return []
    names.sort(key=_natural_key)
    return [os.path.join(folder, f) for f in names]


class PlaylistVideoReader(CameraReader):
    """Đọc LẦN LƯỢT nhiều file video trong một thư mục, theo thứ tự tên.

    Hết video này tự chuyển sang video kế; hết cả danh sách thì dừng (hoặc quay
    lại từ đầu nếu loop=True). Không drop frame — giống VideoFileReader.
    """

    def __init__(self, cam_id: str, paths: list[str], loop: bool = False, fps_limit: float | None = None):
        if not paths:
            raise FileNotFoundError("Danh sách video rỗng")
        super().__init__(cam_id, paths[0], buffer_size=1)
        self.paths = list(paths)
        self.loop = loop
        self.fps_limit = fps_limit
        self.idx = 0
        self._cap = cv2.VideoCapture(self.paths[0])
        if not self._cap.isOpened():
            raise FileNotFoundError(self.paths[0])
        log.info("[%s] playlist %d video, bắt đầu: %s", cam_id, len(self.paths), self.paths[0])

    def start(self) -> "PlaylistVideoReader":
        return self  # đọc đồng bộ

    def _advance(self) -> bool:
        """Mở video kế tiếp. Trả False nếu đã hết (và không loop)."""
        if self._cap is not None:
            self._cap.release()
        self.idx += 1
        if self.idx >= len(self.paths):
            if not self.loop:
                return False
            self.idx = 0
        self._cap = cv2.VideoCapture(self.paths[self.idx])
        log.info("[%s] chuyển video %d/%d: %s", self.cam_id, self.idx + 1, len(self.paths), self.paths[self.idx])
        return self._cap.isOpened()

    def next(self, timeout_s: float = 1.0) -> Frame | None:
        if self.fps_limit:
            time.sleep(1.0 / self.fps_limit)
        tries = 0
        while True:
            ok, img = self._cap.read() if self._cap is not None else (False, None)
            if ok and img is not None:
                self._frame_idx += 1
                self.read += 1
                return Frame(self.cam_id, self._frame_idx, datetime.now(timezone.utc), img)
            tries += 1
            if tries > len(self.paths):   # không video nào đọc được -> dừng, tránh lặp vô hạn
                self.exhausted = True
                return None
            if not self._advance():
                self.exhausted = True
                return None

    def latest(self) -> Frame | None:
        return self.next()


def build_reader(
    cam_id: str,
    source: str,
    buffer_size: int,
    *,
    loop: bool = True,
    fps_limit: float | None = None,
) -> CameraReader:
    src = str(source)
    # Thư mục -> đọc lần lượt tất cả video bên trong, theo thứ tự tên.
    if os.path.isdir(src):
        vids = list_videos_in_dir(src)
        if not vids:
            raise FileNotFoundError(f"Không có file video nào trong thư mục: {src}")
        return PlaylistVideoReader(cam_id, vids, loop=loop, fps_limit=fps_limit)
    if is_video_source(src):
        return VideoFileReader(cam_id, src, loop=loop, fps_limit=fps_limit)
    return CameraReader(cam_id, source, buffer_size=buffer_size)
