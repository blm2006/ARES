"""Dựng "thế giới" tổng hợp để test rule mà không cần detector/tracker thật."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from traffic_safety_ai.core.config import SiteConfig
from traffic_safety_ai.core.homography import GroundCalibration
from traffic_safety_ai.core.kinematics import KinematicsEstimator
from traffic_safety_ai.core.types import Detection, FrameContext, Track, TrackPoint
from traffic_safety_ai.rules.catalog import LegalCatalog

PX_PER_M = 20.0
T0 = datetime(2026, 11, 4, 6, 45, 0)


def topdown_calibration() -> GroundCalibration:
    img = [(0, 0), (10 * PX_PER_M, 0), (10 * PX_PER_M, 30 * PX_PER_M), (0, 30 * PX_PER_M)]
    wld = [(0, 0), (10, 0), (10, 30), (0, 30)]
    return GroundCalibration(img, wld)


def box_at(x_m: float, y_m: float, w_px: float, h_px: float) -> np.ndarray:
    """Hộp có điểm tiếp đất (giữa đáy) tại toạ độ mét (x, y) trong ảnh nhìn từ trên xuống."""
    gx, gy = x_m * PX_PER_M, y_m * PX_PER_M
    return np.array([gx - w_px / 2, gy - h_px, gx + w_px / 2, gy])


class World:
    def __init__(self, scene: SiteConfig | None = None, fps: float = 30.0, calib: bool = True,
                 channel: str = "cctv", size: tuple[int, int] = (1280, 720)):
        self.scene = scene or SiteConfig()
        self.scene.camera.channel = channel
        self.fps = fps
        self.calib = topdown_calibration() if calib else None
        self.tracks: dict[int, Track] = {}
        self.kin = KinematicsEstimator()
        self.frame_idx = -1
        self.frame = np.zeros((size[1], size[0], 3), np.uint8)
        self.catalog = LegalCatalog()

    @property
    def t(self) -> float:
        return self.frame_idx / self.fps

    def step(self, objects: dict[int, tuple[str, np.ndarray]], riders: dict[int, list[int]] | None = None,
             aux: dict | None = None, lights: dict | None = None, ego=None, attrs: dict[int, dict] | None = None,
             ) -> FrameContext:
        self.frame_idx += 1
        t = self.t
        present = []
        new_ids = set()
        for tid, (cls, box) in objects.items():
            d = Detection(box, 0.9, cls)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = Track(tid, d, self.frame_idx, t, history_len=300)
                tr.state = "confirmed"
                self.tracks[tid] = tr
                new_ids.add(tid)
            tr.xyxy = d.xyxy
            tr.last_frame, tr.last_t = self.frame_idx, t
            gm = self.calib.point_to_world(d.bottom) if self.calib else None
            tr.history.append(TrackPoint(self.frame_idx, t, d.xyxy.copy(), 0.9, d.bottom, gm))
            self.kin.update(tr)
            if attrs and tid in attrs:
                tr.attrs.update(attrs[tid])
            present.append(tr)
        rider_of = {p: m for m, ps in (riders or {}).items() for p in ps}
        return FrameContext(
            frame_idx=self.frame_idx, t=t, wall_time=T0 + timedelta(seconds=t), frame=self.frame,
            channel=self.scene.camera.channel, camera_id="CAM-TEST", tracks=present, all_tracks=dict(self.tracks),
            scene=self.scene, fps=self.fps, calib=self.calib, ego=ego, lights=lights or {}, aux=aux or {},
            riders=riders or {}, rider_of=rider_of, new_track_ids=new_ids,
        )


def run(rule, world: World, frames, **kw):
    """frames: hàm i → dict objects (hoặc tuple (objects, extra kwargs)). Trả danh sách sự kiện."""
    events = []
    for i in range(kw.pop("n")):
        spec = frames(i)
        if isinstance(spec, tuple):
            objs, extra = spec
        else:
            objs, extra = spec, {}
        ctx = world.step(objs, **extra)
        events.extend(rule.evaluate(ctx))
    return events
