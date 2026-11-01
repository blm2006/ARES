"""FastAPI cho dashboard ARES. Một file HTML tĩnh (không phụ thuộc CDN — chạy được trong mạng LAN không Internet).

Bảo mật (dashboard có ảnh người trong phòng và nút đổi model):
  * Mặc định chỉ nghe 127.0.0.1. Mở ra mạng bắt buộc đặt ARES_DASHBOARD_TOKEN (kiểm tra ở ares.runtime).
  * Có token: gửi qua header X-ARES-Token, hoặc ?token=... lần đầu (server đặt cookie HttpOnly).
  * Mọi yêu cầu ghi (POST) phải kèm header X-ARES: 1 — trình duyệt không tự gửi header tuỳ biến giữa các
    origin, nên trang web lạ không thể ra lệnh duyệt/đổi model thay người dùng (CSRF).
"""

from __future__ import annotations

import functools
import hmac
import re
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from ares.state import Event

if TYPE_CHECKING:  # pragma: no cover
    from ares.runtime import AresRuntime

STATIC = Path(__file__).parent / "static"
ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,50}$")
CAM_RE = re.compile(r"^[A-Za-z0-9]{1,16}$")
VER_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,32}$")
NO_STORE = {"Cache-Control": "no-store"}


class BoxIn(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    cls: int = Field(0, alias="class")

    model_config = {"populate_by_name": True}


class EditIn(BaseModel):
    boxes: list[BoxIn] = Field(default_factory=list, max_length=300)


class RejectIn(BaseModel):
    reason: str = Field("", max_length=100)


class TuningIn(BaseModel):
    values: dict[str, float | int] = Field(default_factory=dict, max_length=64)


def _err(code: int, msg: str) -> JSONResponse:
    return JSONResponse({"detail": msg}, status_code=code)


def create_app(rt: "AresRuntime") -> FastAPI:
    cfg = rt.cfg
    token = cfg.dashboard_token
    app = FastAPI(title="ARES Dashboard", docs_url=None, redoc_url=None, openapi_url=None)

    # ------------------------------------------------------------------ bảo vệ
    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        supplied = None
        if token and path != "/healthz":
            supplied = (request.headers.get("x-ares-token") or request.query_params.get("token")
                        or request.cookies.get("ares_token") or "")
            if not hmac.compare_digest(supplied.encode(), token.encode()):
                return _err(401, "Cần token. Mở dashboard dạng http://<máy>:%d/?token=<ARES_DASHBOARD_TOKEN>" % cfg.dashboard_port)
        if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("x-ares") != "1":
            return _err(403, "Thiếu header X-ARES: 1 (chống giả mạo yêu cầu từ trang web khác).")
        resp = await call_next(request)
        if token and request.query_params.get("token") and supplied is not None:
            resp.set_cookie("ares_token", token, httponly=True, samesite="strict", max_age=30 * 86400)
        return resp

    def guard_errors(fn):
        """Đổi lỗi nghiệp vụ thành mã HTTP có thông báo tiếng Việt."""
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except KeyError as e:
                return _err(404, f"Không tìm thấy: {e.args[0] if e.args else ''}")
            except FileNotFoundError as e:
                return _err(410, f"File không còn trên đĩa: {e}")
            except ValueError as e:
                return _err(409, str(e))
        return wrapper

    # ------------------------------------------------------------------ trang & trạng thái
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html; charset=utf-8", headers=NO_STORE)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/api/status")
    def status():
        s = rt.status()
        s["tz"] = cfg.timezone
        s["version"] = __import__("ares").__version__
        return JSONResponse(s, headers=NO_STORE)

    @app.get("/api/frame.jpg")
    def frame():
        jpg, _ = rt.state.get_frame()
        if not jpg:
            return _err(404, "Chưa có khung hình")
        return Response(jpg, media_type="image/jpeg", headers=NO_STORE)

    @app.get("/api/history")
    def history(hours: float = 6.0, bucket: int = 300):
        hours = min(max(hours, 0.25), 24 * 14)
        bucket = min(max(bucket, 30), 3600)
        end = rt.clock.naive_now()
        rows = rt.db.telemetry_range(end - timedelta(hours=hours), end, bucket)
        return JSONResponse({"hours": hours, "bucket_s": bucket, "rows": rows}, headers=NO_STORE)

    @app.get("/api/events")
    def events(limit: int = 100, kind: str | None = None):
        limit = min(max(limit, 1), 500)
        kinds = [k for k in (kind or "").split(",") if k] or None
        return JSONResponse({"events": rt.db.recent_events(limit, kinds)}, headers=NO_STORE)

    @app.get("/api/habits")
    def habits():
        return JSONResponse({"slot_min": cfg.habit_slot_min, "min_people": cfg.habit_min_people,
                             "p_threshold": cfg.precool_prob, "rows": rt.db.habit_matrix()}, headers=NO_STORE)

    # ------------------------------------------------------------------ học chủ động
    @app.get("/api/al/stats")
    def al_stats():
        return JSONResponse(rt.al.stats(), headers=NO_STORE)

    @app.get("/api/al/items")
    def al_items(status: str = "PENDING", limit: int = 30, offset: int = 0):
        status = status.upper()
        if status not in ("CANDIDATE", "PENDING", "APPROVED", "REJECTED", "TRAINED", "DUPLICATE", "ALL"):
            return _err(400, "status phải là CANDIDATE | PENDING | APPROVED | REJECTED | TRAINED | DUPLICATE | ALL")
        items = rt.al.list_items(status, min(max(limit, 1), 100), max(offset, 0))
        return JSONResponse({"items": items, "counts": rt.al.counts()}, headers=NO_STORE)

    @app.get("/api/al/items/{image_id}")
    @guard_errors
    def al_item(image_id: str):
        if not ID_RE.match(image_id):
            return _err(400, "image_id không hợp lệ")
        return JSONResponse(rt.al.get_item(image_id), headers=NO_STORE)

    @app.get("/api/al/items/{image_id}/image")
    @guard_errors
    def al_image(image_id: str):
        if not ID_RE.match(image_id):
            return _err(400, "image_id không hợp lệ")
        return FileResponse(rt.al.image_path(image_id), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/al/items/{image_id}/approve")
    @guard_errors
    def al_approve(image_id: str):
        if not ID_RE.match(image_id):
            return _err(400, "image_id không hợp lệ")
        return rt.al.approve(image_id)

    @app.post("/api/al/items/{image_id}/edit")
    @guard_errors
    def al_edit(image_id: str, body: EditIn):
        if not ID_RE.match(image_id):
            return _err(400, "image_id không hợp lệ")
        return rt.al.edit(image_id, [{"bbox": b.bbox, "class": b.cls} for b in body.boxes])

    @app.post("/api/al/items/{image_id}/reject")
    @guard_errors
    def al_reject(image_id: str, body: RejectIn):
        if not ID_RE.match(image_id):
            return _err(400, "image_id không hợp lệ")
        return rt.al.reject(image_id, body.reason)

    @app.post("/api/al/items/{image_id}/restore")
    @guard_errors
    def al_restore(image_id: str):
        if not ID_RE.match(image_id):
            return _err(400, "image_id không hợp lệ")
        return rt.al.restore(image_id)

    # ------------------------------------------------------------------ camera (lưới kiểu CCTV)
    @app.get("/api/cameras")
    def cameras():
        return JSONResponse(rt.hub.overview(), headers=NO_STORE)

    @app.get("/api/cameras/{cam_id}/frame.jpg")
    @guard_errors
    def camera_frame(cam_id: str):
        if not CAM_RE.match(cam_id):
            return _err(400, "Mã camera không hợp lệ")
        jpg, _ = rt.hub.frame(cam_id)
        if not jpg:
            return _err(404, "Camera chưa có khung hình")
        return Response(jpg, media_type="image/jpeg", headers=NO_STORE)

    @app.post("/api/cameras/rescan")
    def cameras_rescan():
        rt.hub.rescan()
        return {"scanning": True}

    # ------------------------------------------------------------------ lọc PCA
    def _need_pca():
        if rt.pca is None:
            raise ValueError("Lọc PCA đang tắt (ARES_PCA_ENABLED=false).")
        return rt.pca

    @app.get("/api/pca/status")
    def pca_status():
        return JSONResponse(rt.pca.status() if rt.pca else {"enabled": False}, headers=NO_STORE)

    @app.post("/api/pca/fit")
    @guard_errors
    def pca_fit():
        pca = _need_pca()
        out = pca.fit(manual=True)
        if out is None:
            raise ValueError(f"Chưa đủ ảnh mẫu để khởi tạo PCA (cần ≥ {cfg.pca_fit_min_manual}, hiện {len(pca.corpus)}).")
        return {"fitted": True, "triage": pca.triage(force=True), "status": pca.status()}

    @app.post("/api/pca/triage")
    @guard_errors
    def pca_triage():
        pca = _need_pca()
        return {"triage": pca.triage(force=True), "status": pca.status()}

    @app.post("/api/pca/release")
    @guard_errors
    def pca_release():
        pca = _need_pca()
        return {"released": pca.release_all()}

    @app.get("/api/pca/train-preview")
    @guard_errors
    def pca_train_preview():
        pca = _need_pca()
        rep = dict(pca.select_for_training(rt.al.items_for_training(), commit=False))
        for k in ("picked_ids", "dropped_ids", "forced_edited"):
            if isinstance(rep.get(k), (list, tuple, set)):
                rep[k] = len(rep[k])
        return JSONResponse(rep, headers=NO_STORE)

    # ------------------------------------------------------------------ model
    @app.get("/api/models")
    def models():
        prod, _ = rt.registry.production()
        return JSONResponse({
            "production": prod, "running": rt.state.model_version, "versions": rt.registry.versions(),
            "retrain": {"running": rt.retrainer.running, "last": rt.retrainer.last_result,
                        "time": cfg.retrain_time, "policy": cfg.auto_promote, "enabled": cfg.retrain_enabled},
            "al_counts": rt.al.counts(),
        }, headers=NO_STORE)

    @app.post("/api/models/rollback")
    @guard_errors
    def rollback():
        cur, _ = rt.registry.production()
        prev = rt.registry.rollback()
        if prev is None:
            raise ValueError("Chưa có phiên bản nào để quay lui.")
        rt.engine.emit(Event(rt.clock.time(), "MODEL_ROLLBACK", "warn", f"Quay lui model {cur} -> {prev} (thủ công từ dashboard)."))
        return {"production": prev}

    @app.post("/api/models/{version}/promote")
    @guard_errors
    def promote(version: str):
        if not VER_RE.match(version):
            return _err(400, "Tên phiên bản không hợp lệ")
        cur, _ = rt.registry.production()
        if version == cur:
            raise ValueError("Phiên bản này đang là production.")
        rt.registry.promote(version)
        rt.engine.emit(Event(rt.clock.time(), "MODEL_PROMOTED", "info", f"Duyệt model {version} làm production (thay {cur}) từ dashboard."))
        return {"production": version}

    # ------------------------------------------------------------------ cài đặt thiết bị (delay / vùng điều chỉnh)
    @app.get("/api/tuning")
    def tuning_get():
        return JSONResponse(rt.tuning.describe(), headers=NO_STORE)

    @app.post("/api/tuning")
    @guard_errors
    def tuning_set(body: TuningIn):
        return rt.tuning.update(body.values)

    @app.post("/api/tuning/reset")
    def tuning_reset():
        return rt.tuning.reset()

    @app.post("/api/retrain")
    def retrain():
        if not cfg.retrain_enabled:
            return _err(409, "Retrain đang tắt (ARES_RETRAIN_ENABLED=false).")
        out = rt.retrain_async(force=True)
        return JSONResponse(out, status_code=202 if out.get("started") else 409)

    return app
