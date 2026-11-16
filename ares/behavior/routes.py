"""API + trang /behavior cho dashboard ARES. Gắn vào app trong ares/dashboard/app.py (cùng lớp bảo vệ token/CSRF)."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

EID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
CAM_RE = re.compile(r"^[A-Za-z0-9]{1,16}$")
NO_STORE = {"Cache-Control": "no-store"}
STATIC = Path(__file__).resolve().parent.parent / "dashboard" / "static"


class VerdictIn(BaseModel):
    verdict: str = Field("", pattern=r"^(true|false|)$")  # true = đúng sự kiện, false = báo nhầm, "" = bỏ đánh giá


def _err(code: int, msg: str) -> JSONResponse:
    return JSONResponse({"detail": msg}, status_code=code)


def register(app: FastAPI, rt) -> None:
    def mon():
        return getattr(rt, "behavior", None)

    def ev_dir(eid: str) -> tuple[dict, Path]:
        m = mon()
        row = m.store.get(eid)
        return row, (m.root / row["dir"]).resolve()

    @app.get("/behavior")
    def behavior_page():
        return FileResponse(STATIC / "behavior.html", media_type="text/html; charset=utf-8", headers=NO_STORE)

    @app.get("/api/behavior/status")
    def behavior_status():
        m = mon()
        if m is None:
            return JSONResponse({"enabled": False, "reason": "Đang tắt (ARES_BEHAVIOR_ENABLED=false) hoặc chạy mô phỏng."},
                                headers=NO_STORE)
        return JSONResponse(m.status(), headers=NO_STORE)

    @app.get("/api/behavior/events")
    def behavior_events(limit: int = 100, status: str = "", kind: str = "", cam: str = ""):
        m = mon()
        if m is None:
            return JSONResponse({"items": []}, headers=NO_STORE)
        if status not in ("", "new", "ack") or not re.match(r"^[A-Z_]{0,20}$", kind) or (cam and not CAM_RE.match(cam)):
            return _err(400, "Tham số lọc không hợp lệ")
        return JSONResponse({"items": m.store.list(max(1, min(limit, 500)), status, kind, cam)}, headers=NO_STORE)

    def _guard(eid: str):
        if mon() is None:
            return _err(409, "Nhận diện hành vi đang tắt")
        if not EID_RE.match(eid):
            return _err(400, "Mã sự kiện không hợp lệ")
        try:
            mon().store.get(eid)
        except KeyError:
            return _err(404, "Không tìm thấy sự kiện")
        return None

    @app.get("/api/behavior/events/{eid}")
    def behavior_event(eid: str):
        if (e := _guard(eid)) is not None:
            return e
        row, _ = ev_dir(eid)
        return JSONResponse(row, headers=NO_STORE)

    @app.get("/api/behavior/events/{eid}/snapshot.jpg")
    def behavior_snapshot(eid: str):
        if (e := _guard(eid)) is not None:
            return e
        _, d = ev_dir(eid)
        p = d / "snapshot.jpg"
        return FileResponse(p, media_type="image/jpeg") if p.exists() else _err(404, "Chưa có ảnh")

    @app.get("/api/behavior/events/{eid}/frames/{n}.jpg")
    def behavior_frame(eid: str, n: int):
        if (e := _guard(eid)) is not None:
            return e
        _, d = ev_dir(eid)
        p = d / "frames" / f"{max(1, min(n, 9999)):04d}.jpg"
        return FileResponse(p, media_type="image/jpeg") if p.exists() else _err(404, "Không có khung này")

    @app.get("/api/behavior/events/{eid}/clip.mp4")
    def behavior_clip(eid: str):
        if (e := _guard(eid)) is not None:
            return e
        row, d = ev_dir(eid)
        p = d / "clip.mp4"
        if not p.exists():
            return _err(404, "Clip chưa ghi xong")
        return FileResponse(p, media_type="video/mp4", filename=f"{row['kind'].lower()}_{eid}.mp4")

    @app.post("/api/behavior/events/{eid}/ack")
    def behavior_ack(eid: str):
        if (e := _guard(eid)) is not None:
            return e
        mon().store.update(eid, status="ack")
        return {"ok": True}

    @app.post("/api/behavior/events/{eid}/verdict")
    def behavior_verdict(eid: str, body: VerdictIn):
        if (e := _guard(eid)) is not None:
            return e
        mon().store.update(eid, verdict=body.verdict, status="ack")
        return {"ok": True}

    @app.post("/api/behavior/ack-all")
    def behavior_ack_all():
        m = mon()
        if m is None:
            return _err(409, "Nhận diện hành vi đang tắt")
        return {"acked": m.store.ack_all()}

    @app.get("/api/behavior/cameras/{cam_id}/frame.jpg")
    def behavior_cam_frame(cam_id: str):
        m = mon()
        if m is None or not CAM_RE.match(cam_id):
            return _err(400, "Không hợp lệ")
        jpg, _ = m.frame(cam_id)
        if not jpg:
            return _err(404, "Camera chưa có khung hình phân tích")
        return Response(jpg, media_type="image/jpeg", headers=NO_STORE)
