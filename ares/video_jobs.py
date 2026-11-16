"""Thử video từ dashboard: người dùng (máy khác) tải video lên server -> máy AI (Edge) nhận job, chạy YOLO người + hành vi,
gửi lại video đã vẽ khung xương + danh sách sự kiện.

Chạy trong luồng riêng của Edge, dùng bộ YOLO-pose RIÊNG (không đụng luồng camera trực tiếp). Mỗi lúc chỉ xử lý 1 video.
Bật mặc định khi có ARES_UPLINK_URL; tắt bằng ARES_UPLINK_VIDEO_JOBS=false.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from ares import __version__
from ares.state import Event
from ares.uplink import HttpError, _multipart

log = logging.getLogger(__name__)
MAX_UPLOAD_MP4 = 200 * 1024 * 1024


class VideoJobWorker:
    def __init__(self, cfg, uplink, emit=None, models_factory=None, analyze_fn=None):
        self.cfg, self.up = cfg, uplink
        self.emit = emit or (lambda e: None)
        self._models_factory = models_factory  # test: thay bằng model giả
        self._analyze = analyze_fn
        self._models = None
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self.current: dict | None = None
        self.done = 0
        self.last_error = ""
        self.root = Path(cfg.behavior_dir) / "video_jobs"

    # ------------------------------------------------------------------ HTTP
    def _download(self, jid: int, dest: Path) -> None:
        req = urllib.request.Request(f"{self.up.url}/api/v1/jobs/{jid}/video")
        req.add_header("Authorization", f"Bearer {self.cfg.uplink_key}")
        req.add_header("X-Edge-Id", self.up.edge_id)
        req.add_header("User-Agent", f"ARES-Edge/{__version__}")
        try:
            with urllib.request.urlopen(req, timeout=max(30.0, self.cfg.uplink_timeout_s)) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        except urllib.error.HTTPError as e:
            raise HttpError(e.code, str(e.reason)) from None

    def _progress(self, jid: int, progress: float, message: str) -> None:
        try:
            self.up._req(f"/api/v1/jobs/{jid}/progress", json.dumps({"progress": progress, "message": message}).encode())
        except HttpError as e:
            if e.code == 404:  # người dùng đã xoá video -> dừng
                raise InterruptedError("đã huỷ trên dashboard") from None
        except Exception as e:  # noqa: BLE001 — mạng chập chờn: bỏ qua lần báo này
            log.debug("Báo tiến độ lỗi: %s", e)

    # ------------------------------------------------------------------ xử lý
    def models(self):
        if self._models is None:
            if self._models_factory is not None:
                self._models = self._models_factory()
            else:
                from ares.behavior.run_video import load_models

                self._models = load_models(self.cfg)
        return self._models

    def run_once(self) -> bool:
        """Nhận + xử lý 1 job. True nếu có job."""
        try:
            r = self.up._req("/api/v1/jobs/next", b"{}")
        except HttpError as e:
            if e.code == 404:  # server cũ chưa có tính năng này
                return False
            raise
        if not r or "id" not in r:
            return False
        jid, name = int(r["id"]), r.get("name", "video")
        self.current = {"id": jid, "name": name, "progress": 0.0}
        d = self.root / str(jid)
        d.mkdir(parents=True, exist_ok=True)
        try:
            src = d / f"source{r.get('ext') or '.mp4'}"
            self._progress(jid, 0.0, "Máy AI đang tải video về")
            self._download(jid, src)
            self._progress(jid, 0.0, "Đang nạp AI")
            pose, clf, _ = self.models()
            analyze = self._analyze
            if analyze is None:
                from ares.behavior.run_video import analyze
            last = [0.0]

            def prog(p: float, n: int) -> None:
                self.current["progress"] = p
                if time.time() - last[0] >= 2.0:
                    last[0] = time.time()
                    self._progress(jid, p, f"Đang phân tích ({round(p * 100)}%, {n} khung)")

            fps = float(self.cfg.uplink_video_fps)
            res = analyze(self.cfg, pose, clf, src, d / "result.mp4", fps, frames_zip=d / "frames.zip",
                          max_w=int(self.cfg.uplink_video_width), progress=prog, should_stop=self._stop.is_set)
            res["device"] = str(getattr(pose, "device", ""))
            res["mode"] = self.cfg.behavior_mode if clf is not None or self.cfg.behavior_mode == "rules" else "rules"
            self._progress(jid, 0.99, "Đang gửi kết quả")
            files = {}
            fz = d / "frames.zip"
            if fz.exists():
                files["frames"] = ("frames.zip", fz.read_bytes(), "application/zip")
            mp = d / "result.mp4"
            if mp.exists() and 0 < mp.stat().st_size <= MAX_UPLOAD_MP4:
                files["video"] = ("result.mp4", mp.read_bytes(), "video/mp4")
            body, ctype = _multipart({"meta": json.dumps(res, ensure_ascii=False)}, files)
            self.up._req(f"/api/v1/jobs/{jid}/result", body, ctype)
            self.done += 1
            self.emit(Event(time.time(), "VIDEO_JOB_DONE", "info",
                            f"Đã phân tích video '{name}' từ dashboard: {len(res['events'])} sự kiện, "
                            f"{res['frames']} khung trong {res['processing_s']:g}s."))
        except InterruptedError as e:
            log.info("Job video %s dừng: %s", jid, e)
        except Exception as e:  # noqa: BLE001
            log.exception("Job video %s lỗi", jid)
            self.last_error = str(e)[:200]
            try:
                self.up._req(f"/api/v1/jobs/{jid}/progress", json.dumps({"error": f"Máy AI lỗi: {e}"[:300]}).encode())
            except Exception:  # noqa: BLE001
                pass
        finally:
            self.current = None
            shutil.rmtree(d, ignore_errors=True)  # kết quả đã nằm trên server; không giữ video người dùng trên máy AI
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            got = False
            if self.up.last_error == "":
                try:
                    got = self.run_once()
                except Exception as e:  # noqa: BLE001
                    log.debug("Hỏi job video lỗi: %s", e)
            self._stop.wait(0.5 if got else 5.0)

    def start(self) -> None:
        self._th = threading.Thread(target=self._run, name="ares-video-jobs", daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=5)

    def status(self) -> dict:
        return {"current": self.current, "done": self.done, "last_error": self.last_error}
