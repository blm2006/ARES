"""Gửi số liệu + sự kiện từ máy Edge lên server trung tâm (ares_server). Bật khi đặt ARES_UPLINK_URL + ARES_UPLINK_KEY.

Nguyên tắc:
  * Edge là nơi AI chạy và ra quyết định. Server chỉ để xem/tổng hợp: mất mạng thì Edge vẫn nhận diện, vẫn báo trên
    dashboard tại chỗ, vẫn chỉnh đèn/điều hòa; sự kiện nằm chờ trong DB của Edge (cột uplinked=0) và được gửi bù khi có mạng.
  * KHÔNG gửi hình trực tiếp từ camera. Chỉ gửi: số người theo camera, trạng thái AI/thiết bị (heartbeat), và sự kiện hành vi
    kèm ảnh/clip tuỳ ARES_UPLINK_MEDIA (none | snapshot | clip) — tối thiểu hoá dữ liệu cá nhân rời khỏi địa điểm.
  * Không thêm thư viện: dùng urllib của Python.
"""

from __future__ import annotations

import io
import json
import logging
import re
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

from ares import __version__
from ares.state import Event

log = logging.getLogger(__name__)
_EDGE_BAD = re.compile(r"[^A-Za-z0-9_.\-]+")
MAX_CLIP_BYTES = 20 * 1024 * 1024


def _ascii(s: str) -> str:
    """'Cổng trường' -> 'Cong truong' (mã camera chỉ dùng chữ không dấu)."""
    import unicodedata

    s = s.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def default_edge_id() -> str:
    return (_EDGE_BAD.sub("-", socket.gethostname()) or "edge")[:40]


class HttpError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(f"HTTP {code}: {msg}")
        self.code = code


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    b = "----ares" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        out.write(v.encode("utf-8"))
        out.write(b"\r\n")
    for k, (fname, data, mime) in files.items():
        out.write(f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\nContent-Type: {mime}\r\n\r\n'.encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{b}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={b}"


class Uplink:
    def __init__(self, cfg, rt, emit=None):
        self.cfg, self.rt = cfg, rt
        self.emit = emit or (lambda e: None)
        self.url = cfg.uplink_url.rstrip("/")
        self.edge_id = (_EDGE_BAD.sub("-", cfg.uplink_edge_id) if cfg.uplink_edge_id else default_edge_id())[:40]
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self.last_ok: float | None = None
        self.last_error = ""
        self.sent_events = 0
        self._fail = 0
        self._next_try = 0.0
        self._was_down = False
        self.site_name = ""
        self._results: list[dict] = []  # kết quả lệnh từ server, gửi lại ở heartbeat sau
        self._last_live = 0.0
        self.live_sent = 0
        self.video = None
        if getattr(cfg, "uplink_video_jobs", False):
            from ares.video_jobs import VideoJobWorker
            self.video = VideoJobWorker(cfg, self, emit=self.emit)

    # ------------------------------------------------------------------ HTTP
    def _req(self, path: str, data: bytes | None = None, ctype: str = "application/json") -> dict:
        req = urllib.request.Request(self.url + path, data=data, method="POST" if data is not None else "GET")
        req.add_header("Authorization", f"Bearer {self.cfg.uplink_key}")
        req.add_header("X-Edge-Id", self.edge_id)
        req.add_header("User-Agent", f"ARES-Edge/{__version__}")
        if data is not None:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.uplink_timeout_s) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail", "")
            except Exception:  # noqa: BLE001
                detail = e.reason
            raise HttpError(e.code, str(detail)) from None

    # ------------------------------------------------------------------ nội dung heartbeat
    def _ai_device(self) -> dict:
        """Thiết bị chạy AI (GPU/CPU) + lý do nếu đã phải chuyển sang CPU — để dashboard máy chủ báo rõ."""
        rt = self.rt
        det = getattr(getattr(rt, "tracker", None), "detector", None)
        beh = getattr(rt, "behavior", None)
        pose = getattr(getattr(beh, "analyzer", None), "pose", None)
        want = str(self.cfg.device or "")
        dev = str(getattr(det, "device", "") or "")
        fb = ""
        if pose is not None and getattr(pose, "fallback_reason", ""):
            fb = pose.fallback_reason
        elif want.startswith("cuda") and dev == "cpu":
            fb = "GPU không chạy được với bản PyTorch hiện tại"
        return {"device": dev, "pose_device": str(getattr(pose, "device", "") or ""), "fallback": fb}

    def heartbeat_payload(self) -> dict:
        rt, cfg = self.rt, self.cfg
        beh = getattr(rt, "behavior", None)
        ov = rt.hub.overview()
        cams = [{"id": c["id"], "name": c["name"], "online": c["online"], "n_people": c["n_people"], "fps": c["fps"],
                 "behavior": bool(beh is not None and beh.wants(c["id"])), "kind": c.get("kind", ""),
                 "source": c.get("source", ""),  # đã che mật khẩu RTSP (cameras.mask_source)
                 "removable": not c.get("control", False)} for c in ov["cameras"]]
        bstat = None
        if beh is not None:
            bstat = {"running": not beh.loading and not beh.error, "loading": beh.loading, "error": beh.error,
                     "mode": cfg.behavior_mode, "classifier": beh.classifier_info is not None,
                     "loop_fps": beh.stats.get("loop_fps"), "new_events": beh.store.counts().get("new", 0)}
        lighting = rt.state.get_lighting()
        ac = rt.state.get_ac()
        sens = rt.state.get_sensors()
        devices_on = bool(rt.state.mqtt_connected) or rt.sim
        pwm = [round(float(p), 1) for p in (lighting.pwm or [])]
        return {
            "edge": {"id": self.edge_id, "name": cfg.uplink_name or rt.hub.control_name, "version": __version__,
                     "room_id": cfg.room_id, "simulation": rt.sim,
                     "uptime_s": None if rt.started_at is None else round(rt.clock.time() - rt.started_at)},
            "cameras": cams,
            "ai": {"person_model": rt.state.model_version, "behavior": bstat, **self._ai_device()},
            "integrations": {
                "lighting": {"enabled": devices_on and bool(pwm), "pwm": pwm,
                             "avg_pwm": round(sum(pwm) / len(pwm), 1) if pwm else None},
                "ac": {"enabled": devices_on and ac.known, **ac.as_json()},
                "sensors": None if sens is None else {"temp_in": sens.temp_in, "humidity": sens.humidity,
                                                      "lux_window": sens.lux_window},
            },
            "uplink": {"pending_events": self.pending(), "media": cfg.uplink_media, "live_s": cfg.uplink_live_s,
                       "remote_cameras": cfg.uplink_remote_cameras},
            "command_results": list(self._results),
        }

    def pending(self) -> int:
        beh = getattr(self.rt, "behavior", None)
        if beh is None:
            return 0
        return beh.store.count_pending_uplink(time.time() - self.cfg.uplink_max_age_days * 86400)

    # ------------------------------------------------------------------ gửi
    def send_heartbeat(self) -> None:
        payload = self.heartbeat_payload()
        r = self._req("/api/v1/heartbeat", json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if not r.get("ok"):
            raise RuntimeError(f"server trả {r}")
        sent = {x["id"] for x in payload["command_results"]}
        self._results = [x for x in self._results if x["id"] not in sent]
        self.apply_updates(r.get("updates") or [])
        for cmd in r.get("commands") or []:
            self._results.append(self.run_command(cmd))

    # ------------------------------------------------------------------ lệnh từ dashboard server (quản lý camera)
    def run_command(self, cmd: dict) -> dict:
        from ares.cameras import CameraSpec, guess_kind, mask_source, sanitize_id

        cid = cmd.get("id")
        if not self.cfg.uplink_remote_cameras:
            return {"id": cid, "ok": False, "message": "Máy Edge đã tắt quản lý camera từ xa (ARES_UPLINK_REMOTE_CAMERAS=false)."}
        path = Path(self.cfg.cameras_file)
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:  # noqa: BLE001
            data = {}
        data.setdefault("auto_usb", self.cfg.camera_auto_detect)
        lst = data.setdefault("cameras", [])
        hub = self.rt.hub
        try:
            if cmd.get("type") == "add_camera":
                src, name = str(cmd.get("source", "")).strip(), str(cmd.get("name", "")).strip()[:60] or "Camera"
                used = {c.get("id") for c in lst} | {s.id for s in hub.specs()} | {"main"}
                base = sanitize_id(cmd.get("cam_id") or _ascii(name), "cam")
                cam_id, k = base, 2
                while cam_id in used:
                    cam_id, k = f"{base[:8]}{k}", k + 1
                spec = CameraSpec(cam_id, name, src, guess_kind(src))
                if not hub.add(spec):
                    return {"id": cid, "ok": False, "message": f"Không mở được camera {mask_source(src)} — kiểm tra địa chỉ RTSP/mạng."}
                lst.append({"id": cam_id, "name": name, "source": src})
                msg = f"Đã thêm camera '{name}' ({cam_id}). Nếu bảng camera báo 'Mất hình': kiểm tra địa chỉ RTSP, mật khẩu, mạng."
            elif cmd.get("type") == "remove_camera":
                cam_id = str(cmd.get("cam_id", ""))
                if cam_id == "main":
                    return {"id": cid, "ok": False, "message": "Không xoá được camera điều khiển (đổi ở ARES_CAMERA_SOURCE)."}
                hub.remove(cam_id)
                data["cameras"] = [c for c in lst if c.get("id") != cam_id]
                msg = f"Đã gỡ camera {cam_id}."
            else:
                return {"id": cid, "ok": False, "message": f"Lệnh không hỗ trợ: {cmd.get('type')}"}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.emit(Event(self.rt.clock.time(), "REMOTE_CAMERA", "info", msg + " (từ dashboard server)"))
            return {"id": cid, "ok": True, "message": msg, "cam_id": cam_id}
        except Exception as e:  # noqa: BLE001
            log.exception("Lệnh từ server lỗi")
            return {"id": cid, "ok": False, "message": f"Lỗi: {e}"}

    # ------------------------------------------------------------------ hình trực tiếp (tuỳ chọn)
    def _frame_for(self, cam_id: str, now: float) -> tuple[bytes | None, float]:
        """Ảnh mới nhất của 1 camera: ưu tiên ảnh AI hành vi (khung xương); không có thì ảnh hộp người (KHÔNG vẽ vùng đèn)."""
        rt = self.rt
        beh = getattr(rt, "behavior", None)
        if beh is not None:
            jpg, t = beh.frame(cam_id)
            if jpg and now - t <= 3:
                return jpg, t
        if cam_id == "main":
            raw = getattr(rt.tracker, "last_raw", None)
            if raw is not None and now - raw[2] <= 5:
                from ares.cameras import render_boxes
                return render_boxes(raw[0], raw[1], self.cfg), raw[2]
            return None, 0.0
        try:
            return rt.hub.frame(cam_id)
        except KeyError:
            return None, 0.0

    def live_frames(self) -> dict[str, bytes]:
        import cv2
        import numpy as np

        rt, out = self.rt, {}
        now = rt.clock.time()
        sent = self.__dict__.setdefault("_live_t", {})
        for c in rt.hub.overview()["cameras"][:16]:
            jpg, t = self._frame_for(c["id"], now)
            if not jpg or now - (t or 0) > 10 or sent.get(c["id"]) == t:  # không có / cũ / đã gửi khung này rồi
                continue
            sent[c["id"]] = t
            W = self.cfg.uplink_live_width
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_REDUCED_COLOR_2 if W <= 400 else cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[1] > W:
                img = cv2.resize(img, (W, int(img.shape[0] * W / img.shape[1])), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok:
                    out[c["id"]] = buf.tobytes()
            else:
                out[c["id"]] = jpg  # đã đủ nhỏ: gửi nguyên, không nén lại
        return out

    def send_live(self) -> int:
        frames = self.live_frames()
        if not frames:
            return 0
        body, ctype = _multipart({"t": str(self.rt.clock.time())},
                                 {f"cam_{cid}": (f"{cid}.jpg", data, "image/jpeg") for cid, data in frames.items()})
        self._req("/api/v1/live", body, ctype)
        self.live_sent += 1
        return len(frames)

    def _live_loop(self) -> None:
        """Luồng riêng cho hình trực tiếp: không phải chờ heartbeat/sự kiện, chu kỳ đúng ARES_UPLINK_LIVE_S."""
        while not self._stop.is_set():
            t0 = time.monotonic()
            if self.last_error == "":
                try:
                    self.send_live()
                except Exception as e:  # noqa: BLE001
                    log.debug("Gửi hình trực tiếp lỗi: %s", e)
            self._stop.wait(max(0.05, self.cfg.uplink_live_s - (time.monotonic() - t0)))

    def apply_updates(self, updates: list[dict]) -> int:
        """Người trực bấm Đã xử lý / Đúng / Báo nhầm trên server -> ghi về DB của Edge (dữ liệu train lại)."""
        beh = getattr(self.rt, "behavior", None)
        if beh is None:
            return 0
        n = 0
        for u in updates:
            try:
                beh.store.get(str(u.get("id", "")))
            except KeyError:
                continue
            kw = {"status": "ack"} if u.get("status") == "ack" else {}
            if u.get("verdict") in ("true", "false"):
                kw["verdict"] = u["verdict"]
            if kw:
                beh.store.update(str(u["id"]), **kw)
                n += 1
        return n

    def send_events(self, limit: int = 5) -> int:
        beh = getattr(self.rt, "behavior", None)
        if beh is None:
            return 0
        sent = 0
        now = time.time()
        for ev in beh.store.pending_uplink(now - self.cfg.uplink_max_age_days * 86400, limit):
            if not ev["clip_ready"] and now - ev["t"] < 90:  # đợi clip ghi xong (tối đa 90 giây)
                continue
            d = beh.root / ev["dir"]
            meta = {k: ev[k] for k in ("id", "t", "ts", "cam_id", "cam_name", "kind", "label", "severity", "message",
                                       "score", "source", "verdict")}
            mp = d / "meta.json"
            if mp.exists():
                try:
                    meta["detail"] = json.loads(mp.read_text(encoding="utf-8")).get("detail", {})
                except Exception:  # noqa: BLE001
                    pass
            files = self._media_files(d)
            body, ctype = _multipart({"meta": json.dumps(meta, ensure_ascii=False)}, files)
            try:
                self._req("/api/v1/events", body, ctype)
                beh.store.update(ev["id"], uplinked=1)
                sent += 1
                self.sent_events += 1
            except HttpError as e:
                if e.code in (400, 413, 415, 422):  # server không bao giờ nhận gói này -> bỏ, đừng gửi lặp mãi
                    beh.store.update(ev["id"], uplinked=3)
                    log.warning("Server từ chối sự kiện %s: %s", ev["id"], e)
                    continue
                raise
        return sent

    def _media_files(self, d: Path) -> dict:
        mode = self.cfg.uplink_media
        files: dict[str, tuple[str, bytes, str]] = {}
        if mode == "none":
            return files
        snap = d / "snapshot.jpg"
        if snap.exists():
            files["snapshot"] = ("snapshot.jpg", snap.read_bytes(), "image/jpeg")
        if mode == "clip":
            frames = sorted((d / "frames").glob("*.jpg"))
            if frames:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:  # jpg đã nén sẵn
                    for f in frames:
                        zf.write(f, f.name)
                if buf.tell() <= MAX_CLIP_BYTES:
                    files["frames"] = ("frames.zip", buf.getvalue(), "application/zip")
            mp4 = d / "clip.mp4"
            if mp4.exists() and mp4.stat().st_size <= MAX_CLIP_BYTES:
                files["clip"] = ("clip.mp4", mp4.read_bytes(), "video/mp4")
        return files

    # ------------------------------------------------------------------ vòng
    # ------------------------------------------------------------------ vòng
    def tick(self) -> None:
        """Một lượt: heartbeat (đúng chu kỳ) + gửi sự kiện chờ. Lỗi mạng -> lùi dần thời gian thử lại (tối đa 5 phút)."""
        now = time.time()
        if now < self._next_try:
            return
        try:
            self.send_heartbeat()
            self.send_events()
            self.last_ok, self.last_error, self._fail = now, "", 0
            self._next_try = now + self.cfg.uplink_interval_s
            if self._was_down:
                self._was_down = False
                self.emit(Event(self.rt.clock.time(), "UPLINK_UP", "info", "Đã kết nối lại server trung tâm."))
        except Exception as e:  # noqa: BLE001
            self._fail += 1
            self.last_error = str(e)[:200]
            self._next_try = now + min(300.0, 5.0 * 2 ** min(self._fail, 6))
            if not self._was_down:
                self._was_down = True
                hint = " — kiểm tra ARES_UPLINK_KEY" if isinstance(e, HttpError) and e.code == 401 else ""
                self.emit(Event(self.rt.clock.time(), "UPLINK_DOWN", "warn",
                                f"Không gửi được lên server trung tâm ({self.last_error}){hint}. Edge vẫn chạy bình thường, "
                                "sự kiện sẽ được gửi bù."))

    def _run(self) -> None:
        try:
            r = self._req("/api/v1/ping")
            self.site_name = r.get("site", "")
            self.emit(Event(self.rt.clock.time(), "UPLINK_READY", "info",
                            f"Kết nối server trung tâm {self.url} — địa điểm '{self.site_name}', máy '{self.edge_id}'."))
        except Exception as e:  # noqa: BLE001
            log.warning("Chưa kết nối được server %s: %s", self.url, e)
        while not self._stop.is_set():
            self.tick()
            # có sự kiện mới thì gửi sớm (không chờ hết chu kỳ heartbeat)
            if self.last_error == "" and self.pending():
                try:
                    self.send_events()
                except Exception:  # noqa: BLE001
                    pass
            self._stop.wait(1.0)

    def start(self) -> None:
        self._th = threading.Thread(target=self._run, name="ares-uplink", daemon=True)
        self._th.start()
        if self.cfg.uplink_live_s > 0:
            threading.Thread(target=self._live_loop, name="ares-uplink-live", daemon=True).start()
        if self.video is not None:
            self.video.start()

    def stop(self) -> None:
        self._stop.set()
        if self.video is not None:
            self.video.stop()
        if self._th is not None:
            self._th.join(timeout=5)

    def status(self) -> dict:
        return {"url": self.url, "edge_id": self.edge_id, "site": self.site_name, "last_ok": self.last_ok,
                "last_error": self.last_error, "sent_events": self.sent_events, "media": self.cfg.uplink_media,
                "live_s": self.cfg.uplink_live_s, "live_sent": self.live_sent,
                "video_jobs": self.video.status() if self.video is not None else None}
