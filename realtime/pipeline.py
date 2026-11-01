"""Vòng lặp realtime cho MỘT camera. 1 process / camera.

Thứ tự: đọc frame -> infer+track -> cập nhật track history -> xuất output
        -> (không chặn) đưa frame cho collector.

Collector chỉ làm việc rẻ và ghi ra queue; mọi thứ nặng ở worker process. Nếu
collector chậm đi vì bất kỳ lý do gì, backpressure sẽ ngắt nó trước khi FPS tụt.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from collector.service import Collector
from common.config import Settings, get_settings
from common.metrics import CameraTelemetry
from common.schemas import BBox
from realtime.camera import CameraReader, build_reader
from realtime.detector import PersonDetector
from realtime.tracker import TrackHistory

log = logging.getLogger(__name__)


@dataclass
class LiveState:
    """Trạng thái để dashboard đọc (metadata only, không store frame)."""

    cam_id: str
    boxes: list[BBox] = field(default_factory=list)
    people_count: int = 0
    active_tracks: int = 0
    telemetry: dict = field(default_factory=dict)
    collector: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, boxes, people, tracks, telemetry, collector) -> None:
        with self.lock:
            self.boxes = boxes
            self.people_count = people
            self.active_tracks = tracks
            self.telemetry = telemetry
            self.collector = collector

    def read(self):
        with self.lock:
            return (
                list(self.boxes),
                self.people_count,
                self.active_tracks,
                dict(self.telemetry),
                dict(self.collector),
            )


class RealtimePipeline:
    def __init__(
        self,
        cam_id: str,
        source: str,
        settings: Settings | None = None,
        collector: Collector | None = None,
        detector: PersonDetector | None = None,
        enable_collector: bool = True,
        model_version: str = "unknown",
        role: str = "primary",
        stream_buffer = None,
        video_loop: bool = True,
        video_fps: float | None = None,
    ):
        self.s = settings or get_settings()
        self.cam_id = cam_id
        self.reader: CameraReader = build_reader(
            cam_id, source, self.s.ring_buffer_size,
            loop=video_loop, fps_limit=video_fps,
        )
        self.detector = detector or PersonDetector(settings=self.s)
        self.history = TrackHistory(self.s.track_history_len, self.s.short_track_frames)
        self.telemetry = CameraTelemetry(cam_id, model_version=model_version, role=role)
        self.reader.set_drop_callback(self.telemetry.on_frame_dropped)
        self.collector = collector if collector is not None else (Collector(self.s) if enable_collector else None)
        self.state = LiveState(cam_id)
        self.stream_buffer = stream_buffer
        self._stop = threading.Event()
        self.frames_done = 0

    # ------------------------------------------------------------------
    def run(self, max_frames: int | None = None) -> None:
        self.reader.start()
        log.info("[%s] realtime bắt đầu", self.cam_id)
        try:
            while not self._stop.is_set():
                frame = self.reader.next(timeout_s=2.0)
                if frame is None:
                    if getattr(self.reader, "exhausted", False):
                        log.info("[%s] nguồn video đã hết -> dừng.", self.cam_id)
                        break
                    continue
                self.telemetry.on_frame_read()
                self.process(frame.image, frame.ts, frame.frame_idx)
                self.frames_done += 1
                if max_frames and self.frames_done >= max_frames:
                    break
        finally:
            self.reader.stop()
            self.telemetry.maybe_flush(force=True)
            log.info("[%s] realtime dừng sau %d frame", self.cam_id, self.frames_done)

    # ------------------------------------------------------------------
    def process(self, image: np.ndarray, ts: datetime, frame_idx: int) -> list[BBox]:
        res = self.detector.track(image)
        events = self.history.add(frame_idx, res.boxes)

        n_switch = sum(1 for e in events if e.reason.value == "id_switch")
        if n_switch:
            self.telemetry.on_id_switch(n_switch)

        self.telemetry.on_frame_processed(res.inference_ms, len(res.boxes))

        collector_snapshot: dict = {}
        if self.collector is not None:
            try:
                self.collector.on_frame(
                    image=image,
                    cam_id=self.cam_id,
                    ts=ts,
                    frame_idx=frame_idx,
                    boxes=res.boxes,
                    failure_events=events,
                    history=self.history,
                )
                self.telemetry.queue_depth = self.collector.guard.state().queue_depth
                if frame_idx % 30 == 0:
                    collector_snapshot = self.collector.snapshot()
            except Exception:
                # Learning KHÔNG BAO GIỜ được làm chết realtime.
                log.exception("[%s] collector lỗi, bỏ qua frame này", self.cam_id)

        self.telemetry.maybe_flush()
        
        # Gửi frame tới stream buffer (không store trong state)
        if self.stream_buffer is not None:
            self.stream_buffer.put(image)
        
        self.state.update(
            res.boxes,
            len(res.boxes),
            self.history.active_count,
            self.telemetry.snapshot(),
            collector_snapshot or self.state.collector,
        )
        return res.boxes

    def stop(self) -> None:
        self._stop.set()

    @property
    def telemetry_data(self):
        return self.telemetry.snapshot()
