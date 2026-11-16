"""Đọc khung hình ở luồng riêng (tránh I/O chặn vòng nhận diện).

  * File video / thư mục video: KHÔNG bỏ khung (hàng đợi chặn) → kết quả lặp lại được, tốc độ đo đúng.
    Mốc thời gian lấy từ CAP_PROP_POS_MSEC (đúng cả với video tốc độ khung thay đổi).
  * RTSP / webcam: bỏ khung CŨ nhất khi đầy (luôn xử lý khung mới nhất), tự kết nối lại khi rớt mạng.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".ts"}


@dataclass
class FramePacket:
    frame_idx: int
    t: float                 # giây tính từ đầu nguồn
    frame: np.ndarray
    source: str


def natural_key(s: str) -> list:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def list_sources(src: str) -> list[str]:
    p = Path(src)
    if p.is_dir():
        files = [str(f) for f in sorted(p.iterdir(), key=lambda f: natural_key(f.name)) if f.suffix.lower() in VIDEO_EXT]
        if not files:
            raise FileNotFoundError(f"Thư mục {src} không có video")
        return files
    return [src]


def is_live(src: str) -> bool:
    return src.isdigit() or src.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))


def guess_start_time(path: str, duration_s: float | None = None) -> tuple[datetime, str]:
    """Thời điểm bắt đầu video: theo tên file (YYYYMMDD_HHMMSS…) → hoặc giờ sửa file trừ thời lượng → hoặc bây giờ."""
    name = Path(path).name
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[ T_-]?(\d{2})[-_:h]?(\d{2})[-_:m]?(\d{2})", name)
    if m:
        try:
            return datetime(*(int(x) for x in m.groups())), "theo tên file"
        except ValueError:
            pass
    p = Path(path)
    if p.exists():
        end = datetime.fromtimestamp(p.stat().st_mtime)
        return end - timedelta(seconds=duration_s or 0.0), "ước lượng từ giờ sửa file (nên khai báo --start-time)"
    return datetime.now(), "giờ máy lúc chạy"


class VideoSource:
    def __init__(self, src: str, queue_size: int = 16, stride: int = 1, loop: bool = False,
                 max_frames: int | None = None):
        self.src = str(src)
        self.live = is_live(self.src)
        self.sources = [self.src] if self.live else list_sources(self.src)
        self.stride = max(1, stride)
        self.loop = loop
        self.max_frames = max_frames
        self.q: deque[FramePacket | None] = deque(maxlen=queue_size if self.live else 10 ** 9)
        self.queue_size = queue_size
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fps = 30.0
        self.size = (0, 0)
        self.frame_count = 0
        self.dropped = 0
        self._probe()

    def _open(self, src: str) -> cv2.VideoCapture:
        if src.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            return cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if src.isdigit():
            return cv2.VideoCapture(int(src))
        return cv2.VideoCapture(src)

    def _probe(self) -> None:
        cap = self._open(self.sources[0])
        if not cap.isOpened():
            raise OSError(f"Không mở được nguồn video: {self.sources[0]}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and 1 <= fps <= 240 else 30.0
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

    @property
    def duration_s(self) -> float | None:
        return self.frame_count / self.fps if (self.frame_count and not self.live) else None

    def start(self) -> VideoSource:
        self._thread = threading.Thread(target=self._run, name="video-reader", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _put(self, pkt: FramePacket | None) -> None:
        with self._cv:
            if not self.live:
                while len(self.q) >= self.queue_size and not self._stop.is_set():
                    self._cv.wait(0.1)
            elif len(self.q) == self.q.maxlen and pkt is not None:
                self.dropped += 1
            self.q.append(pkt)
            self._cv.notify_all()

    def _run(self) -> None:
        idx = 0
        t_offset = 0.0
        try:
            while not self._stop.is_set():
                for s in self.sources:
                    cap = self._open(s)
                    retries = 0
                    while not cap.isOpened() and self.live and retries < 30 and not self._stop.is_set():
                        time.sleep(min(10, 1 + retries))
                        cap = self._open(s)
                        retries += 1
                    if not cap.isOpened():
                        log.error("Không mở được %s", s)
                        continue
                    fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
                    t0 = time.monotonic()
                    local = 0
                    last_t = -1.0
                    while not self._stop.is_set():
                        ok, frame = cap.read()
                        if not ok:
                            if self.live:
                                log.warning("Mất tín hiệu %s — kết nối lại", s)
                                cap.release()
                                time.sleep(1.0)
                                cap = self._open(s)
                                continue
                            break
                        if local % self.stride == 0:
                            if self.live:
                                t = time.monotonic() - t0
                            else:
                                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                                t = ms / 1000.0 if ms and ms > 0 else local / (fps or 30.0)
                                if t <= last_t:
                                    t = last_t + 1.0 / (fps or 30.0)
                                last_t = t
                            self._put(FramePacket(idx, t_offset + t, frame, s))
                            idx += 1
                            if self.max_frames and idx >= self.max_frames:
                                cap.release()
                                return
                        local += 1
                    t_offset += (last_t if last_t > 0 else local / (fps or 30.0)) + 1.0 / (fps or 30.0)
                    cap.release()
                if not self.loop:
                    break
        finally:
            self._put(None)

    def read(self, timeout: float = 30.0) -> FramePacket | None:
        end = time.monotonic() + timeout
        with self._cv:
            while not self.q:
                rem = end - time.monotonic()
                if rem <= 0 or self._stop.is_set():
                    return None
                self._cv.wait(min(rem, 0.2))
            pkt = self.q.popleft()
            self._cv.notify_all()
            return pkt

    def __iter__(self):
        while True:
            pkt = self.read()
            if pkt is None:
                return
            yield pkt
