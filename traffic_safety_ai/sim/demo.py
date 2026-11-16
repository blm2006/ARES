"""Chạy demo mô phỏng: dựng cảnh → pipeline đầy đủ → hồ sơ bằng chứng + báo cáo + video có chú thích."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from traffic_safety_ai.core.config import AppPaths
from traffic_safety_ai.core.pipeline import AsyncVideoWriter, Pipeline
from traffic_safety_ai.evidence.store import Store
from traffic_safety_ai.lpr.reader import PlateReader
from traffic_safety_ai.rules.catalog import LegalCatalog
from traffic_safety_ai.sim.scenario import CCTVSim, POVSim, SimPerception

log = logging.getLogger(__name__)


def run_demo(channel: str = "cctv", data_dir: str | Path | None = None, write_video: bool = True,
             duration: float | None = None, async_io: bool = True, make_report: bool = True,
             progress: Callable[[int, int], None] | None = None,
             start_time: datetime | None = None) -> dict[str, Any]:
    paths = AppPaths(data_dir).ensure()
    sim = CCTVSim() if channel == "cctv" else POVSim()
    if duration is not None:
        sim.duration = float(duration)
    scene = sim.scene()
    plates = PlateReader(detector=None, ocr=None)
    perception = SimPerception(sim, plates)
    store = Store(paths.db)
    start = start_time or datetime.now().replace(hour=7, minute=5, second=0, microsecond=0)
    pipe = Pipeline(scene, perception, paths, store, plates, source=f"mo_phong_{channel}", fps=sim.fps,
                    start_time=start, catalog=LegalCatalog(), async_io=async_io,
                    run_note="Dữ liệu MÔ PHỎNG để kiểm thử — không phải vi phạm thật.",
                    models_desc={"perception": "mô phỏng (SimPerception)"})
    writer = None
    video_path = None
    if write_video:
        video_path = pipe.run_dir / "video_chu_thich.mp4"
        writer = AsyncVideoWriter(str(video_path), sim.fps, (1280, 720))
    n_total = int(sim.duration * sim.fps)
    events = []
    try:
        for i, t, img, objs in sim.frames():
            perception.set_frame_objects(objs)
            res = pipe.process(i, t, img, render=write_video)
            for ev, vid in zip(res.events, res.vids):
                events.append((vid, ev))
            if writer is not None and res.annotated is not None:
                writer.write(res.annotated)
            if progress and i % 30 == 0:
                progress(i, n_total)
    finally:
        pipe.finish()
        if writer is not None:
            writer.close()
    report = None
    if make_report:
        from traffic_safety_ai.analytics.reports import build_report

        report = build_report(store, paths, camera_id=scene.camera.id, run_id=pipe.run_id,
                              speed_limit=scene.speed.limit_kmh, title_suffix=" (MÔ PHỎNG)")
    summary = {
        "run_id": pipe.run_id,
        "run_dir": str(pipe.run_dir),
        "video": str(video_path) if video_path else None,
        "report": str(report) if report else None,
        "events": [{"id": vid, "rule": ev.rule_code, "name": ev.rule_name, "severity": ev.severity,
                    "t": round(ev.t, 2), "track": ev.primary_track_id, "variant": ev.variant} for vid, ev in events],
        "by_rule": dict(Counter(ev.rule_code for _, ev in events)),
        "rules": [(s.code, s.enabled, s.reason) for s in pipe.rule_report],
        "db": str(paths.db),
    }
    store.close()
    return summary
