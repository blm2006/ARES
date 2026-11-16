"""Studio — "mở như app": kéo-thả video, tự vẽ vùng/vạch/đèn ngay trên trình duyệt, bấm chạy, xem kết quả.

Gắn vào dashboard sẵn có (cùng cổng, cùng cơ chế bảo vệ token + header X-TSA):

  * Video kéo-thả được chép vào  <data>/studio/videos   (tải lên dạng luồng thô, không cần python-multipart).
  * Mỗi video có MỘT file cấu hình riêng  <data>/studio/scenes/<tên video>.yaml  — vì mỗi góc quay cần vùng/vạch riêng.
  * Phân tích chạy ở luồng nền (mỗi lúc một video); tiến độ + khung hình trực tiếp lấy qua API.

Toạ độ luôn là PIXEL theo kích thước gốc của video (camera.frame_size).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import ValidationError

from traffic_safety_ai import CONFIG_DIR, PACKAGE_ROOT
from traffic_safety_ai.core.config import AppPaths, SiteConfig, load_site_config, save_site_config

log = logging.getLogger("tsa.studio")

STATIC = Path(__file__).parent / "static"
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".ts", ".mpg", ".mpeg", ".wmv", ".flv"}
MAX_UPLOAD = 20 * 1024 ** 3          # 20 GB / tệp
EDITABLE = ("zones", "lines", "traffic_lights", "calibration")


def safe_name(name: str) -> str:
    """Tên tệp an toàn: bỏ đường dẫn, bỏ dấu tiếng Việt và ký tự lạ (OpenCV trên Windows hay lỗi với tên có dấu)."""
    base = str(name).replace("\\", "/").split("/")[-1].strip()
    base = unicodedata.normalize("NFKD", base.replace("đ", "d").replace("Đ", "D"))
    base = "".join(c for c in base if not unicodedata.combining(c))
    base = re.sub(r"[^A-Za-z0-9\-_. ()\[\]]+", "_", base).strip(". ")
    if not Path(base).stem.strip("_ ") or Path(base).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, "Tên tệp không hợp lệ hoặc không phải video (mp4, mkv, avi, mov, webm...)")
    return base[-150:]


class StudioPaths:
    def __init__(self, paths: AppPaths):
        self.root = paths.root / "studio"
        self.videos = self.root / "videos"
        self.scenes = self.root / "scenes"
        self.links = self.root / "duong_dan_video.json"
        for p in (self.videos, self.scenes):
            p.mkdir(parents=True, exist_ok=True)
        # thư mục video mẫu cạnh gói (vd C:\yolo-al\samples) — chỉ đọc
        self.extra = [d for d in (PACKAGE_ROOT.parent / "samples", PACKAGE_ROOT / "samples") if d.is_dir()]

    # -------------------------------------------------------------- video
    def _linked(self) -> dict[str, str]:
        try:
            data = json.loads(self.links.read_text(encoding="utf-8"))
            return {k: v for k, v in data.items() if isinstance(v, str)}
        except (OSError, ValueError):
            return {}

    def add_link(self, path: Path) -> str:
        name = safe_name(path.name)
        data = self._linked()
        data[name] = str(path)
        self.links.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return name

    def remove(self, name: str) -> None:
        data = self._linked()
        if name in data:
            data.pop(name)
            self.links.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        f = self.videos / name
        if f.is_file():
            f.unlink()

    def all_videos(self) -> dict[str, tuple[Path, str]]:
        out: dict[str, tuple[Path, str]] = {}
        for d in self.extra:
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in VIDEO_EXT:
                    out[f.name] = (f, "mẫu")
        for name, p in self._linked().items():
            if Path(p).is_file():
                out[name] = (Path(p), "liên kết")
        for f in sorted(self.videos.iterdir()):
            if f.is_file() and f.suffix.lower() in VIDEO_EXT:
                out[f.name] = (f, "đã thả vào")
        return out

    def video(self, name: str) -> Path:
        hit = self.all_videos().get(name)
        if hit is None:
            raise HTTPException(404, f"Không thấy video '{name}'")
        return hit[0]

    def scene_file(self, name: str) -> Path:
        return self.scenes / (Path(safe_name(name)).stem + ".yaml")


def probe(path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise HTTPException(415, f"Không mở được video '{path.name}' (tệp hỏng, thiếu codec, hoặc đường dẫn có "
                                     "ký tự đặc biệt — thử kéo-thả tệp vào Studio)")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        fps = float(fps) if 1 <= fps <= 240 else 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(fps, 2), "frames": n, "duration_s": round(n / fps, 1) if n else None}
    finally:
        cap.release()


def grab_frame(path: Path, pos: float) -> bytes:
    """Khung hình tại vị trí pos (0..1) của video → JPEG kích thước gốc."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise HTTPException(415, f"Không mở được video '{path.name}'")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame = None
        if n > 1 and pos > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(max(pos, 0.0), 0.98) * (n - 1)))
            ok, frame = cap.read()
            if not ok:
                frame = None
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if frame is None:
            ok, frame = cap.read()
            if not ok:
                raise HTTPException(415, "Không đọc được khung hình nào từ video")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise HTTPException(500, "Không nén được khung hình")
        return buf.tobytes()
    finally:
        cap.release()


def vi_errors(e: ValidationError) -> str:
    rows = []
    for err in e.errors()[:8]:
        loc = " → ".join(str(x) for x in err.get("loc", ()))
        rows.append(f"{loc}: {err.get('msg', '')}")
    return "Cấu hình chưa hợp lệ — " + "; ".join(rows)


def default_scene(name: str, size: tuple[int, int]) -> SiteConfig:
    """Cấu hình khởi đầu cho video mới: kế thừa THAM SỐ (rule, tốc độ, riêng tư) của file mẫu, KHÔNG kế thừa hình vẽ."""
    base = CONFIG_DIR / "cctv_config.yaml"
    cfg = load_site_config(base) if base.exists() else SiteConfig()
    data = cfg.model_dump(mode="json")
    for k in EDITABLE:
        data.pop(k, None)
    out = SiteConfig.model_validate(data)
    stem = Path(name).stem
    out.camera.id = re.sub(r"[^A-Za-z0-9]+", "-", unicodedata.normalize("NFKD", stem).encode("ascii", "ignore")
                           .decode()).strip("-").upper()[:24] or "CAM-01"
    out.camera.name = f"Video {stem}"
    out.camera.location = "Chưa khai báo địa điểm"
    out.camera.channel = "cctv"
    out.camera.frame_size = (int(size[0]), int(size[1]))
    return out


def rule_table(scene: SiteConfig) -> list[dict[str, Any]]:
    """Tình huống nào chạy được XÉT RIÊNG HÌNH VẼ (giả định đủ model; thiếu model sẽ báo sau khi bấm chạy)."""
    from traffic_safety_ai.rules.catalog import LegalCatalog
    from traffic_safety_ai.rules.registry import ALL_RULES, build_rules

    from_scene = {"calibration", "speed_trap", "bgsub"}
    caps = {c for cls in ALL_RULES for req in cls.requirements for c in req.any_of} - from_scene
    caps.add("traffic_light_det")
    _, report = build_rules(scene, LegalCatalog(), caps)
    ch = scene.camera.channel
    return [{"code": r.code, "name": r.name, "enabled": r.enabled, "reason": r.reason}
            for r in report if ch in r.channels]


# ====================================================================== chạy nền
class Job:
    """Một lần phân tích ở luồng nền. Trạng thái: nap_model → dang_chay → xong | da_dung | loi."""

    def __init__(self, video: str, source: Path, scene_file: Path, opts: dict[str, Any]):
        self.video = video
        self.source = source
        self.scene_file = scene_file
        self.opts = opts
        self.state = "nap_model"
        self.message = "Đang nạp model..."
        self.frame = 0
        self.total = 0
        self.events = 0
        self.proc_fps = 0.0
        self.run_id: str | None = None
        self.report_url: str | None = None
        self.video_out: Path | None = None
        self.started = time.time()
        self.ended: float | None = None
        self.rules: list[dict[str, Any]] = []
        self.stop_flag = threading.Event()
        self.thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self.state in ("nap_model", "dang_chay")

    def public(self) -> dict[str, Any]:
        el = (self.ended or time.time()) - self.started
        pct = round(100.0 * self.frame / self.total, 1) if self.total else None
        return {"video": self.video, "state": self.state, "message": self.message, "frame": self.frame,
                "total": self.total, "percent": pct, "events": self.events, "proc_fps": round(self.proc_fps, 1),
                "elapsed_s": round(el, 1), "run_id": self.run_id, "report_url": self.report_url,
                "has_video": bool(self.video_out and self.video_out.exists() and not self.active),
                "rules": self.rules, "active": self.active}


class JobRunner:
    def __init__(self, paths: AppPaths, store: Any, live: Any):
        self.paths = paths
        self.store = store
        self.live = live
        self.lock = threading.Lock()
        self.job: Job | None = None

    def start(self, job: Job) -> None:
        with self.lock:
            if self.job is not None and self.job.active:
                raise HTTPException(409, f"Đang phân tích '{self.job.video}' — chờ xong hoặc bấm Dừng trước.")
            self.job = job
            job.thread = threading.Thread(target=self._work, args=(job,), name="studio-job", daemon=True)
            job.thread.start()

    def stop(self) -> None:
        j = self.job
        if j is not None and j.active:
            j.stop_flag.set()
            j.message = "Đang dừng..."

    # ------------------------------------------------------------------
    def _work(self, job: Job) -> None:
        try:
            self._run(job)
        except Exception as e:  # noqa: BLE001 — báo lỗi lên giao diện thay vì chết im lặng
            log.exception("Studio: lỗi khi phân tích %s", job.video)
            job.state = "loi"
            job.message = f"Lỗi: {e}"
        finally:
            job.ended = time.time()
            if self.live is not None:
                self.live.pipeline = None

    def _run(self, job: Job) -> None:
        from traffic_safety_ai.core.config import load_models_config
        from traffic_safety_ai.core.detector import ModelHub
        from traffic_safety_ai.core.pipeline import AsyncVideoWriter, Pipeline, YoloPerception
        from traffic_safety_ai.core.video_source import VideoSource, guess_start_time
        from traffic_safety_ai.lpr.reader import build_plate_reader
        from traffic_safety_ai.rules.catalog import LegalCatalog
        from traffic_safety_ai.rules.registry import needed_models

        o = job.opts
        stride = max(1, int(o.get("stride") or 1))
        lpr = bool(o.get("lpr", True))
        scene = load_site_config(job.scene_file)
        mcfg = load_models_config(os.environ.get("TSA_MODELS") or None)   # mặc định configs/models.yaml
        if o.get("device"):
            mcfg.device = str(o["device"])
        src = VideoSource(str(job.source), stride=stride)
        W, H = src.size
        scene = scene.scaled_to(W, H)
        job.total = int(src.frame_count or 0)
        start, how = guess_start_time(src.sources[0], src.duration_s)
        if o.get("start_time"):
            from traffic_safety_ai.cli import _parse_time

            t = _parse_time(str(o["start_time"]))
            if t is not None:
                start, how = t, "do người dùng nhập"
        hub = ModelHub(mcfg)
        hub.load(needed_models(scene, lpr=lpr))
        perception = YoloPerception(hub, mcfg, scene)
        plates, lpr_desc = build_plate_reader(hub, mcfg) if lpr else (None, "tắt")
        models_desc = dict(hub.status)
        models_desc["lpr"] = lpr_desc
        fps_eff = src.fps / stride
        pipe = Pipeline(scene, perception, self.paths, self.store, plates, source=str(job.source), fps=fps_eff,
                        start_time=start, catalog=LegalCatalog(), draw_every=max(1, int(o.get("draw_every") or 2)),
                        models_desc=models_desc, run_note=f"Studio; cấu hình {job.scene_file.name}; giờ bắt đầu {how}")
        pipe.preview = True
        pipe.viz.labels = "day_du" if o.get("full_labels") else "gon"
        job.run_id = pipe.run_id
        job.rules = [{"code": s.code, "name": s.name, "enabled": s.enabled, "reason": s.reason}
                     for s in pipe.rule_report if scene.camera.channel in s.channels]
        if self.live is not None:
            self.live.pipeline = pipe
        writer = None
        if o.get("save_video", True):
            job.video_out = pipe.run_dir / "video_chu_thich.mp4"
            writer = AsyncVideoWriter(str(job.video_out), fps_eff, (W, H))
        job.state = "dang_chay"
        job.message = "Đang phân tích..."
        src.start()
        try:
            for pkt in src:
                if job.stop_flag.is_set():
                    break
                res = pipe.process(pkt.frame_idx, pkt.t, pkt.frame, render=writer is not None)
                job.events += len(res.events)
                job.frame = pkt.frame_idx + 1
                job.proc_fps = pipe.proc_fps
                if writer is not None and res.annotated is not None:
                    writer.write(res.annotated)
        finally:
            src.stop()
            pipe.finish()
            if writer is not None:
                writer.close()
        try:
            from traffic_safety_ai.analytics.reports import build_report

            rep = build_report(self.store, self.paths, camera_id=scene.camera.id, run_id=pipe.run_id,
                               speed_limit=scene.speed.limit_kmh)
            job.report_url = f"/reports/{rep.parent.name}/{rep.name}"
        except Exception:  # noqa: BLE001
            log.exception("Studio: không lập được báo cáo")
        if job.stop_flag.is_set():
            job.state, job.message = "da_dung", f"Đã dừng ở khung {job.frame} — {job.events} sự kiện."
        else:
            job.state, job.message = "xong", f"Xong {job.frame} khung — {job.events} sự kiện."


# ====================================================================== API
def attach_studio(app: FastAPI, paths: AppPaths, store: Any, live: Any, authorized: Any = None) -> JobRunner:
    sp = StudioPaths(paths)
    runner = JobRunner(paths, store, live)
    app.state.studio = runner

    @app.get("/studio", response_class=HTMLResponse)
    def studio_page(req: Request) -> Response:
        if authorized is not None and not authorized(req):
            return HTMLResponse("<meta charset='utf-8'><h3>Cần mã truy cập</h3><p>Mở <code>/?token=MÃ</code> trước.</p>",
                                status_code=401)
        return HTMLResponse((STATIC / "studio.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/api/studio/videos")
    def videos() -> dict[str, Any]:
        rows = []
        for name, (p, kind) in sp.all_videos().items():
            try:
                size = p.stat().st_size
            except OSError:
                continue
            rows.append({"name": name, "kind": kind, "size_mb": round(size / 1e6, 1),
                         "has_scene": sp.scene_file(name).exists()})
        rows.sort(key=lambda r: (r["kind"] == "mẫu", r["name"].lower()))
        return {"videos": rows, "folder": str(sp.videos), "scenes_folder": str(sp.scenes)}

    @app.put("/api/studio/upload")
    async def upload(req: Request, name: str) -> dict[str, Any]:
        fname = safe_name(name)
        dst = sp.videos / fname
        tmp = dst.with_name(dst.name + ".dang_tai")
        total = 0
        try:
            with tmp.open("wb") as f:
                async for chunk in req.stream():
                    total += len(chunk)
                    if total > MAX_UPLOAD:
                        raise HTTPException(413, "Tệp quá lớn (giới hạn 20 GB) — dùng ô 'thêm theo đường dẫn' thay vì kéo-thả")
                    f.write(chunk)
            if total == 0:
                raise HTTPException(400, "Tệp rỗng")
            tmp.replace(dst)
        finally:
            if tmp.exists():
                tmp.unlink()
        try:
            info = probe(dst)
        except HTTPException:
            dst.unlink(missing_ok=True)
            raise
        return {"name": fname, "size_mb": round(total / 1e6, 1), **info}

    @app.post("/api/studio/link")
    def link(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        raw = str(payload.get("path") or "").strip().strip('"')
        p = Path(raw)
        if not raw or not p.is_file() or p.suffix.lower() not in VIDEO_EXT:
            raise HTTPException(400, "Đường dẫn không tồn tại hoặc không phải tệp video")
        probe(p)
        return {"name": sp.add_link(p.resolve())}

    @app.delete("/api/studio/videos/{name}")
    def remove(name: str) -> dict[str, Any]:
        if runner.job is not None and runner.job.active and runner.job.video == name:
            raise HTTPException(409, "Video đang được phân tích — dừng trước khi gỡ")
        sp.remove(safe_name(name))
        return {"ok": True}

    @app.get("/api/studio/info")
    def info(video: str) -> dict[str, Any]:
        return probe(sp.video(video))

    @app.get("/api/studio/frame")
    def frame(video: str, pos: float = 0.0) -> Response:
        return Response(grab_frame(sp.video(video), pos), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    def _load(video: str) -> tuple[SiteConfig, bool, dict[str, Any]]:
        src = sp.video(video)
        meta = probe(src)
        f = sp.scene_file(video)
        if f.exists():
            cfg = load_site_config(f).scaled_to(meta["width"], meta["height"])
            cfg.camera.frame_size = (meta["width"], meta["height"])
            return cfg, True, meta
        return default_scene(video, (meta["width"], meta["height"])), False, meta

    @app.get("/api/studio/scene")
    def get_scene(video: str) -> dict[str, Any]:
        cfg, saved, meta = _load(video)
        return {"scene": cfg.model_dump(mode="json"), "saved": saved, "video": meta,
                "file": str(sp.scene_file(video)), "rules": rule_table(cfg)}

    def _merge(video: str, payload: dict[str, Any]) -> SiteConfig:
        cfg, _, meta = _load(video)
        data = cfg.model_dump(mode="json")
        incoming = payload.get("scene") or {}
        for k in EDITABLE:
            if k in incoming:
                data[k] = incoming[k]
        cam = incoming.get("camera") or {}
        for k in ("name", "location", "id"):
            if isinstance(cam.get(k), str) and cam[k].strip():
                data["camera"][k] = cam[k].strip()
        sp_in = incoming.get("speed") or {}
        if sp_in.get("limit_kmh") is not None:
            data["speed"]["limit_kmh"] = sp_in["limit_kmh"]
        data["camera"]["frame_size"] = [meta["width"], meta["height"]]
        try:
            return SiteConfig.model_validate(data)
        except ValidationError as e:
            raise HTTPException(422, vi_errors(e)) from e

    @app.post("/api/studio/check")
    def check_scene(video: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Kiểm tra hình vẽ hiện tại (chưa lưu) → bảng tình huống nào chạy được."""
        return {"rules": rule_table(_merge(video, payload))}

    @app.put("/api/studio/scene")
    def put_scene(video: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        cfg = _merge(video, payload)
        f = sp.scene_file(video)
        save_site_config(cfg, f)
        return {"ok": True, "file": str(f), "rules": rule_table(cfg)}

    @app.post("/api/studio/run")
    def run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        video = str(payload.get("video") or "")
        src = sp.video(video)
        f = sp.scene_file(video)
        if not f.exists():
            raise HTTPException(400, "Chưa lưu hình vẽ cho video này — bấm 'Lưu' trước khi chạy")
        job = Job(video, src, f, dict(payload.get("options") or {}))
        runner.start(job)
        return job.public()

    @app.get("/api/studio/job")
    def job_status() -> dict[str, Any]:
        return {"job": runner.job.public() if runner.job is not None else None}

    @app.post("/api/studio/stop")
    def stop() -> dict[str, Any]:
        runner.stop()
        return {"ok": True}

    @app.get("/api/studio/result.mp4")
    def result_video() -> Response:
        j = runner.job
        if j is None or j.video_out is None or not j.video_out.exists() or j.active:
            raise HTTPException(404, "Chưa có video chú thích")
        return FileResponse(j.video_out, media_type="video/mp4", filename=f"{Path(j.video).stem}_chu_thich.mp4")

    return runner
