"""Dashboard realtime: MJPEG stream + bbox/ID + số liệu + /metrics.

Fixed: separate frame buffer, no memory leak, add gc cleanup.
"""

from __future__ import annotations

import gc
import threading
import time
from collections import deque

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from common.config import get_settings
from realtime.pipeline import RealtimePipeline

PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<title>Realtime — person tracking</title>
<style>
:root{color-scheme:light dark}
body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;margin:0;padding:24px;background:#f6f7f9;color:#16191d}
@media(prefers-color-scheme:dark){body{background:#111316;color:#e8eaed}.card{background:#1a1d21!important;border-color:#32373d!important}}
h1{font-size:20px;margin:0 0 16px}
.grid{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
.card{background:#fff;border:1px solid #d8dde3;border-radius:10px;padding:14px 16px;min-width:220px}
img{max-width:100%;border-radius:10px;border:1px solid #d8dde3}
.k{font-size:12px;letter-spacing:.03em;text-transform:uppercase;opacity:.65}
.v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}
.warn{color:#b32424}
table{border-collapse:collapse;font-size:13px;width:100%}
td{padding:3px 8px 3px 0}
td:last-child{text-align:right;font-variant-numeric:tabular-nums}
</style></head><body>
<h1>Realtime — person detection &amp; tracking</h1>
<div class="grid">
  <div><img id="stream" src="/stream"></div>
  <div class="card"><div class="k">Người trong khung</div><div class="v" id="people">–</div>
    <div class="k" style="margin-top:12px">Active tracks</div><div class="v" id="tracks">–</div></div>
  <div class="card"><div class="k">FPS thật</div><div class="v" id="fps">–</div>
    <div class="k" style="margin-top:12px">Inference</div><div class="v" id="inf">–</div>
    <div class="k" style="margin-top:12px">Frame drop</div><div class="v" id="drop">–</div></div>
  <div class="card"><div class="k">Collector</div><table id="col"></table></div>
</div>
<script>
async function tick(){
  try{
    const r = await fetch('/api/state'); const d = await r.json();
    people.textContent = d.people_count; tracks.textContent = d.active_tracks;
    fps.textContent = (d.telemetry.fps??0).toFixed(1);
    inf.textContent = (d.telemetry.inference_ms??0).toFixed(1)+' ms';
    const dr = (d.telemetry.frame_drop_rate??0)*100;
    drop.textContent = dr.toFixed(1)+'%'; drop.className = 'v'+(dr>5?' warn':'');
    const c = d.collector||{}; const rows=[];
    rows.push(['backpressure', c.backpressure??'–']);
    rows.push(['queue depth', c.queue_depth??'–']);
    rows.push(['disk free', (c.disk_free_pct??'–')+'%']);
    for(const [k,v] of Object.entries(c.accepted||{})) rows.push(['nhận · '+k, v]);
    col.innerHTML = rows.map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  }catch(e){}
}
setInterval(tick, 1000); tick();
</script></body></html>"""


class StreamBuffer:
    """Riêng biệt: frame buffer cho streaming, không lưu vào state chính."""
    def __init__(self, maxsize: int = 2):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()
    
    def put(self, frame):
        with self.lock:
            self.buffer.append(frame)
    
    def get(self):
        with self.lock:
            if self.buffer:
                return self.buffer[-1]
        return None


def create_app(pipelines: dict[str, RealtimePipeline], stream_buffers: dict[str, StreamBuffer] | None = None) -> FastAPI:
    app = FastAPI(title="realtime-dashboard")
    default_cam = next(iter(pipelines))
    
    if stream_buffers is None:
        stream_buffers = {cam: StreamBuffer() for cam in pipelines}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/state")
    def state(cam: str | None = None):
        p = pipelines[cam or default_cam]
        boxes, people, tracks, tel, col = p.state.read()
        return JSONResponse(
            {
                "cam_id": p.cam_id,
                "people_count": people,
                "active_tracks": tracks,
                "telemetry": tel,
                "collector": col,
                "boxes": [b.model_dump() for b in boxes],
            }
        )

    @app.get("/metrics")
    def metrics():
        """Prometheus text format — cho ai muốn cắm Grafana."""
        lines = []
        for cam, p in pipelines.items():
            t = p.telemetry.snapshot()
            for k in (
                "fps",
                "inference_ms",
                "frame_drop_rate",
                "queue_depth",
                "gpu_util",
                "id_switch_per_min",
                "avg_track_length",
                "people_count",
            ):
                lines.append(f'yolo_al_{k}{{cam="{cam}",role="{t["role"]}"}} {t[k]}')
        return "\n".join(lines) + "\n"

    def gen(cam: str):
        buf = stream_buffers[cam]
        p = pipelines[cam]
        frame_count = 0
        
        while True:
            frame = buf.get()
            if frame is None:
                time.sleep(0.05)
                continue
            
            boxes, *_ = p.state.read()
            img = frame.copy()
            
            for b in boxes:
                cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 200, 120), 2)
                label = f"#{b.track_id} {b.conf:.2f}" if b.track_id is not None else f"{b.conf:.2f}"
                cv2.putText(img, label, (int(b.x1), max(14, int(b.y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 120), 1, cv2.LINE_AA)
            
            ok, buf_img = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + buf_img.tobytes() + b"\r\n"
            
            frame_count += 1
            if frame_count % 150 == 0:  # GC mỗi ~5 giây @ 30 fps
                gc.collect()
            
            time.sleep(0.02)  # 50 FPS max

    @app.get("/stream")
    def stream(cam: str | None = None):
        return StreamingResponse(
            gen(cam or default_cam), media_type="multipart/x-mixed-replace; boundary=f"
        )

    return app


def serve(pipelines: dict[str, RealtimePipeline], stream_buffers: dict[str, StreamBuffer] | None = None) -> tuple[threading.Thread, dict[str, StreamBuffer]]:
    import uvicorn

    if stream_buffers is None:
        stream_buffers = {cam: StreamBuffer() for cam in pipelines}
    
    s = get_settings()
    app = create_app(pipelines, stream_buffers)
    cfg = uvicorn.Config(app, host=s.dashboard_host, port=s.dashboard_port, log_level="warning")
    server = uvicorn.Server(cfg)
    th = threading.Thread(target=server.run, name="dashboard", daemon=True)
    th.start()
    return th, stream_buffers
