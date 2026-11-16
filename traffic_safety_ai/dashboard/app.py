"""Trang quản trị nội bộ (FastAPI): xem/duyệt hồ sơ, tra cứu biển số kiểu "phạt nguội", thống kê, khung hình trực tiếp.

Bảo mật mặc định:
  * Chỉ nghe 127.0.0.1. Mở ra mạng LAN (host khác) thì BẮT BUỘC có mã truy cập (token) — tự sinh nếu không khai báo.
  * Mọi thao tác ghi (duyệt/loại/sửa biển) cần header `X-TSA: 1` → chặn CSRF từ trang web lạ
    (trình duyệt không gửi được header tuỳ biến sang origin khác khi không bật CORS).
  * Tệp bằng chứng chỉ phục vụ đúng tên tệp trong thư mục của hồ sơ (không cho đi ra ngoài thư mục).
  * Mọi thao tác duyệt được ghi nhật ký (ai, lúc nào, từ IP nào).
"""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

from traffic_safety_ai import __version__
from traffic_safety_ai.analytics.counting import PCU_DEFAULT, summarize_counts, summarize_speeds
from traffic_safety_ai.core.config import AppPaths
from traffic_safety_ai.core.types import SEVERITY_VI, STATUS_VI, VEHICLE_TYPE_VI
from traffic_safety_ai.evidence.store import Store, evidence_folder
from traffic_safety_ai.lpr.vn_plate import normalize_text, parse_plate
from traffic_safety_ai.rules.catalog import LegalCatalog

STATIC = Path(__file__).parent / "static"
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


class LiveState:
    """Cầu nối với pipeline đang chạy trong cùng tiến trình (lệnh `run --dashboard`)."""

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.started = time.time()

    def status(self) -> dict[str, Any] | None:
        p = self.pipeline
        if p is None:
            return None
        st = dict(p.status or {})
        st["camera"] = p.scene.camera.name
        st["location"] = p.scene.camera.location
        st["channel"] = p.channel
        st["rules"] = [{"code": s.code, "name": s.name, "enabled": s.enabled, "reason": s.reason} for s in p.rule_report]
        return st

    def jpeg(self) -> bytes | None:
        p = self.pipeline
        return None if p is None else p.latest_jpeg


def _public(rec: dict[str, Any]) -> dict[str, Any]:
    r = {k: v for k, v in rec.items() if k not in ("evidence_dir", "plate_candidates")}
    r["severity_label"] = SEVERITY_VI.get(rec.get("severity") or "", rec.get("severity"))
    r["status_label"] = STATUS_VI.get(rec.get("status") or "", rec.get("status"))
    r["vehicle_label"] = rec.get("vehicle_label") or VEHICLE_TYPE_VI.get(rec.get("vehicle_cls") or "", "—")
    files = rec.get("files") or {}
    r["has"] = {k: (k in files) for k in ("composite.jpg", "composite_public.jpg", "overview.jpg", "plate.jpg",
                                          "vehicle.jpg", "before.jpg", "after.jpg", "clip.mp4", "clip.webm")}
    return r


def code_build() -> str:
    """Dấu vân tay của mã nguồn đang chạy — để lệnh `studio` biết máy chủ cũ có còn đúng bản mới cập nhật không."""
    import hashlib

    from traffic_safety_ai import PACKAGE_ROOT

    h = hashlib.sha1()
    for f in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "tests" in f.parts or "__pycache__" in f.parts:
            continue
        try:
            h.update(f.read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:12]


BUILD = code_build()          # tính MỘT lần lúc nạp → phản ánh mã mà tiến trình này thực sự đang chạy


def create_app(paths: AppPaths | None = None, token: str | None = None, live: LiveState | None = None,
               require_token: bool = False) -> FastAPI:
    paths = (paths or AppPaths()).ensure()
    store = Store(paths.db)
    catalog = LegalCatalog()
    live = live or LiveState()
    if require_token and not token:
        token = secrets.token_urlsafe(16)
    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        yield
        store.close()

    app = FastAPI(title="Giám sát ATGT khu vực trường học", version=__version__, docs_url=None, redoc_url=None,
                  lifespan=lifespan)
    app.state.token = token
    app.state.store = store
    app.state.live = live

    # ------------------------------------------------------------------ bảo vệ
    def _client(req: Request) -> str:
        return req.client.host if req.client else "?"

    def _authorized(req: Request) -> bool:
        if not token:
            return True
        got = req.headers.get("authorization", "").removeprefix("Bearer ").strip() or req.cookies.get("tsa_token", "")
        return bool(got) and hmac.compare_digest(got.encode("utf-8"), token.encode("utf-8"))

    @app.middleware("http")
    async def guard(req: Request, call_next):  # type: ignore[no-untyped-def]
        path = req.url.path
        if token and path in ("/", "/studio") and req.query_params.get("token"):
            if hmac.compare_digest(req.query_params["token"].encode("utf-8"), token.encode("utf-8")):
                resp = RedirectResponse(path, status_code=303)
                resp.set_cookie("tsa_token", token, httponly=True, samesite="strict", max_age=12 * 3600)
                return resp
        if path.startswith("/api/") and not _authorized(req):
            return JSONResponse({"detail": "Cần mã truy cập (mở /?token=...)"}, status_code=401)
        if req.method in ("POST", "PUT", "DELETE") and req.headers.get("x-tsa") != "1":
            return JSONResponse({"detail": "Thiếu header X-TSA (chống CSRF)"}, status_code=403)
        resp = await call_next(req)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "SAMEORIGIN"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    # ------------------------------------------------------------------ trang
    @app.get("/", response_class=HTMLResponse)
    def index(req: Request) -> Response:
        if token and not _authorized(req):
            return HTMLResponse("<meta charset='utf-8'><h3>Cần mã truy cập</h3><p>Mở đường dẫn có dạng "
                                "<code>/?token=MÃ</code> (mã in ra ở cửa sổ chạy dashboard).</p>", status_code=401)
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ studio (kéo-thả video, tự vẽ, chạy)
    from traffic_safety_ai.dashboard.studio import attach_studio

    attach_studio(app, paths, store, live, authorized=_authorized)

    # ------------------------------------------------------------------ dữ liệu
    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {"version": __version__, "build": BUILD, "pid": os.getpid(), "data_dir": str(paths.root),
                "live": live.pipeline is not None,
                "rules": [{"code": c, "name": catalog.name(c, c)} for c in sorted(catalog.codes())],
                "severity": SEVERITY_VI, "status": STATUS_VI, "catalog_warning": catalog.warning}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {"live": live.status(), "stats_today": store.stats(day=time.strftime("%Y-%m-%d")),
                "runs": store.runs(5)}

    @app.get("/api/live.jpg")
    def live_jpg() -> Response:
        b = live.jpeg()
        if b is None:
            raise HTTPException(404, "Không có luồng trực tiếp")
        return Response(b, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/violations")
    def violations(rule: str | None = None, status: str | None = None, severity: str | None = None,
                   camera: str | None = None, day: str | None = None, plate: str | None = None,
                   vehicle: str | None = None, day_from: str | None = None, day_to: str | None = None,
                   run: str | None = None, q: str | None = None,
                   page: int = 1, limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        rows, total = store.list_violations(rule=rule or None, status=status or None, severity=severity or None,
                                            camera=camera or None, day=day or None, plate=plate or None,
                                            vehicle=vehicle or None, day_from=day_from or None,
                                            day_to=day_to or None, run=run or None, q=q or None,
                                            limit=limit, offset=(max(1, page) - 1) * limit)
        return {"total": total, "page": page, "items": [_public(r) for r in rows]}

    @app.get("/api/violations-facets")
    def violation_facets(rule: str | None = None, severity: str | None = None, camera: str | None = None,
                         vehicle: str | None = None, day_from: str | None = None, day_to: str | None = None,
                         run: str | None = None, q: str | None = None) -> dict[str, Any]:
        out = store.violation_facets(rule=rule or None, severity=severity or None, camera=camera or None,
                                     vehicle=vehicle or None, day_from=day_from or None, day_to=day_to or None,
                                     run=run or None, q=q or None)
        out["vehicles"] = [{"id": v, "name": VEHICLE_TYPE_VI.get(v, v)} for v in out["vehicles"]]
        out["status"] = STATUS_VI
        return out

    @app.get("/vi-pham", response_class=HTMLResponse)
    def violations_page(req: Request) -> Response:
        if token and not _authorized(req):
            return HTMLResponse("<meta charset='utf-8'><h3>Cần mã truy cập</h3><p>Mở <code>/?token=MÃ</code> trước.</p>",
                                status_code=401)
        return HTMLResponse((STATIC / "vi_pham.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/api/violations/{vid}")
    def violation(vid: str) -> dict[str, Any]:
        rec = store.get_violation(vid)
        if rec is None:
            raise HTTPException(404, "Không có hồ sơ")
        out = _public(rec)
        out["plate_candidates"] = rec.get("plate_candidates")
        return out

    @app.get("/api/violations/{vid}/file/{name}")
    def violation_file(vid: str, name: str) -> Response:
        rec = store.get_violation(vid)
        if "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(400, "Tên tệp không hợp lệ")
        found = evidence_folder(rec, paths.evidence) if rec is not None else None
        if found is None:
            raise HTTPException(404, "Không có hồ sơ")
        folder = found.resolve()
        f = (folder / name).resolve()
        if f.parent != folder or not f.is_file():
            raise HTTPException(404, "Không có tệp")
        if name == "record.html":
            return HTMLResponse(f.read_text(encoding="utf-8"))
        return FileResponse(f)

    @app.post("/api/violations/{vid}/review")
    def review(vid: str, req: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        action = str(body.get("action", ""))
        reviewer = str(body.get("reviewer") or "").strip()[:80]
        note = str(body.get("note") or "").strip()[:500]
        plate = str(body.get("plate") or "").strip()[:20] or None
        if not reviewer:
            raise HTTPException(400, "Cần ghi tên người duyệt")
        if action not in ("approve", "reject", "reset", "edit_plate"):
            raise HTTPException(400, "Thao tác không hợp lệ")
        rec = store.get_violation(vid)
        if rec is None:
            raise HTTPException(404, "Không có hồ sơ")
        if action == "edit_plate":
            if not plate:
                raise HTTPException(400, "Thiếu biển số")
            ok = store.review(vid, {"cho_xac_minh": "reset", "da_xac_minh": "approve", "loai_bo": "reject"}
                              .get(rec.get("status") or "", "reset"), reviewer, note or "sửa biển số", plate)
        else:
            ok = store.review(vid, action, reviewer, note, plate)
        detail = json.dumps({"action": action, "plate": plate, "note": note,
                             "old_plate": rec.get("plate_display")}, ensure_ascii=False)
        store.audit(reviewer, f"review:{action}", vid, detail, _client(req))
        return {"ok": ok, "item": _public(store.get_violation(vid) or {})}

    @app.get("/api/lookup")
    def lookup(plate: str, req: Request) -> dict[str, Any]:
        q = normalize_text(plate)
        if len(q) < 4:
            raise HTTPException(400, "Nhập ít nhất 4 ký tự biển số")
        p = parse_plate(q)
        rows = store.lookup_plate(q)
        store.audit("dashboard", "lookup", q, "", _client(req))
        return {"query": q, "display": p.display if p else q, "valid": bool(p and p.valid),
                "total": len(rows), "items": [_public(r) for r in rows]}

    @app.get("/api/stats")
    def stats(day: str | None = None, camera: str | None = None) -> dict[str, Any]:
        return store.stats(day=day or None, camera=camera or None)

    @app.get("/api/traffic")
    def traffic(day: str | None = None, camera: str | None = None, limit_kmh: float | None = None) -> dict[str, Any]:
        cs = summarize_counts(store.counts(camera_id=camera or None, day=day or None), PCU_DEFAULT)
        ss = summarize_speeds(store.speeds(camera_id=camera or None, day=day or None), limit_kmh)
        return {"counts": cs, "speeds": ss}

    @app.get("/api/hazards")
    def hazards(camera: str | None = None, day: str | None = None) -> dict[str, Any]:
        return {"items": store.hazard_points(camera_id=camera or None, day=day or None)}

    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        return {"items": store.runs(100)}

    @app.get("/api/audit")
    def audit() -> dict[str, Any]:
        return {"items": store.audit_log(300)}

    @app.post("/api/report")
    def report(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        from traffic_safety_ai.analytics.reports import build_report

        out = build_report(store, paths, camera_id=body.get("camera") or None, day=body.get("day") or None)
        return {"path": str(out), "url": f"/reports/{out.parent.name}/{out.name}"}

    @app.get("/reports/{folder}/{name}")
    def report_file(folder: str, name: str, req: Request) -> Response:
        if not _authorized(req):
            raise HTTPException(401, "Cần mã truy cập")
        base = paths.reports.resolve()
        f = (base / folder / name).resolve()
        if f.parent.parent != base or not f.is_file():
            raise HTTPException(404, "Không có tệp")
        return FileResponse(f)

    @app.get("/evidence/{day}/{vid}/{name}")
    def evidence_file(day: str, vid: str, name: str, req: Request) -> Response:
        """Liên kết tương đối trong báo cáo HTML (../../evidence/<ngày>/<mã>/record.html) khi mở qua dashboard."""
        if not _authorized(req):
            raise HTTPException(401, "Cần mã truy cập")
        rec = store.get_violation(vid)
        found = evidence_folder(rec, paths.evidence) if rec is not None else None
        if found is None or found.name != vid or found.parent.name != day:
            raise HTTPException(404, "Không có hồ sơ")
        return violation_file(vid, name)

    return app


def serve(host: str = "127.0.0.1", port: int = 8200, token: str | None = None, paths: AppPaths | None = None,
          live: LiveState | None = None, block: bool = True) -> tuple[Any, str | None]:
    """Chạy dashboard. Trả (server, token). block=False → chạy ở luồng nền (dùng cùng lệnh `run`)."""
    import threading

    import uvicorn

    need = host not in LOOPBACK
    app = create_app(paths, token, live, require_token=need)
    tok = app.state.token
    cfg = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/" + (f"?token={tok}" if tok else "")
    print(f"Dashboard: {url}")
    if tok:
        print("  (mở đúng đường dẫn trên để đăng nhập; giữ kín mã truy cập)")
    if block:
        server.run()
    else:
        threading.Thread(target=server.run, name="dashboard", daemon=True).start()
    return server, tok
