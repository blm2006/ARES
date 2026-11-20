"""Train lại AI theo lệnh từ dashboard server (Hành vi > Train lại AI).

Dữ liệu train lấy từ đánh giá cảnh báo: người trực bấm **Đúng** / **Báo nhầm** trên dashboard -> server gửi ngược về
máy AI trong heartbeat -> lưu vào behavior.db của máy AI. Lệnh này chạy ở luồng nền trên máy AI:

  * ``behavior``: train lại model hành vi (ST-GCN-lite) = dataset công khai đã tách (nếu có) + mọi sự kiện đã đánh giá
    (Đúng -> mẫu của loại đó, Báo nhầm -> mẫu "bình thường"). Xong thì nạp model mới ngay, không cần khởi động lại.
  * ``person``: chạy vòng học chủ động của YOLO người (ảnh đã duyệt trên dashboard kỹ thuật :8100), có cổng kiểm định.

Tiến độ gửi lên server trong heartbeat (khoá ``retrain``).
"""

from __future__ import annotations

import contextlib
import io
import logging
import threading
import time

log = logging.getLogger(__name__)


class _Tail(io.TextIOBase):
    def __init__(self, keep: int = 40):
        self.lines: list[str] = []
        self.keep = keep
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.lines.append(line.rstrip()[:200])
                del self.lines[:-self.keep]
        return len(s)


class RemoteTrainer:
    def __init__(self, cfg, rt, emit=None):
        self.cfg, self.rt = cfg, rt
        self.emit = emit or (lambda e: None)
        self.state: dict = {"running": False}
        self._th: threading.Thread | None = None
        self._tail = _Tail()

    def reviewed_counts(self) -> dict:
        beh = getattr(self.rt, "behavior", None)
        if beh is None or getattr(beh, "store", None) is None:
            return {}
        try:
            rows = beh.store.reviewed()
        except Exception:  # noqa: BLE001
            return {}
        return {"true": sum(1 for r in rows if r.get("verdict") == "true"),
                "false": sum(1 for r in rows if r.get("verdict") == "false")}

    def status(self) -> dict:
        st = dict(self.state)
        if st.get("running"):
            st["log"] = self._tail.lines[-12:]
        st["reviewed"] = self.reviewed_counts()
        return st

    def start(self, target: str = "behavior", epochs: int = 40) -> tuple[bool, str]:
        if self._th is not None and self._th.is_alive():
            return False, "Đang train — đợi lần này xong."
        target = target if target in ("behavior", "person", "hazards", "all") else "behavior"
        epochs = int(max(5, min(300, epochs)))
        self._tail = _Tail()
        self.state = {"running": True, "target": target, "epochs": epochs, "started": time.time(), "message": "Đang chuẩn bị"}
        self._th = threading.Thread(target=self._run, args=(target, epochs), name="ares-remote-train", daemon=True)
        self._th.start()
        name = {"behavior": "hành vi", "person": "YOLO người", "hazards": "cháy/khói + hút thuốc", "all": "tất cả model"}[target]
        return True, f"Đã bắt đầu train lại ({name})."

    # ------------------------------------------------------------------
    def _run(self, target: str, epochs: int) -> None:
        from ares.state import Event

        out: dict = {}
        ok, msgs = True, []
        try:
            if target in ("behavior", "all"):
                self.state["message"] = "Đang train model hành vi"
                r = self._train_behavior(epochs)
                out["behavior"] = r
                ok = ok and r.get("ok", False)
                msgs.append(r.get("message", ""))
            if target in ("hazards", "all"):
                self.state["message"] = "Đang train model cháy/khói + hút thuốc"
                r = self._train_hazards(epochs)
                out["hazards"] = r
                ok = ok and r.get("ok", False)
                msgs.append(r.get("message", ""))
            if target in ("person", "all"):
                self.state["message"] = "Đang train YOLO người"
                r = self._train_person()
                out["person"] = r
                ok = ok and r.get("ok", False)
                msgs.append(r.get("message", ""))
        except Exception as e:  # noqa: BLE001
            log.exception("Train lại lỗi")
            ok, msgs = False, [f"Lỗi: {e}"]
        self.state = {"running": False, "target": target, "ok": ok, "finished": time.time(),
                      "started": self.state.get("started"), "message": " ".join(m for m in msgs if m)[:600],
                      "result": out, "log": self._tail.lines[-20:]}
        try:
            self.emit(Event(time.time(), "REMOTE_TRAIN_DONE" if ok else "REMOTE_TRAIN_FAILED", "info" if ok else "warn",
                            self.state["message"] or "Train lại xong."))
        except Exception:  # noqa: BLE001
            pass

    def _train_behavior(self, epochs: int) -> dict:
        from ares.behavior.train import load_all, train

        n_rev = self.reviewed_counts()
        with contextlib.redirect_stdout(self._tail):
            groups = load_all(self.cfg)
            if len(groups) < 2:
                return {"ok": False, "message": f"Chưa đủ dữ liệu để train model hành vi: mới có {len(groups)} clip/sự kiện có "
                        f"khung xương (đã đánh giá: {n_rev.get('true', 0)} Đúng, {n_rev.get('false', 0)} Báo nhầm). "
                        "Cần ≥ 2 — đánh giá thêm cảnh báo hoặc tải dataset công khai (python -m ares.behavior download/extract)."}
            path = train(self.cfg, epochs=epochs, promote=True)
        if path is None:
            return {"ok": False, "message": "Train model hành vi không tạo được model (xem nhật ký)."}
        import json

        meta = {}
        try:
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        reloaded = self._reload_behavior()
        m = meta.get("metrics") or {}
        return {"ok": True, "model": path.name, "macro_f1": m.get("macro_f1"), "accuracy": m.get("accuracy"),
                "n_train": meta.get("n_train"), "clips": len(meta.get("clips") or []), "reviewed": n_rev,
                "message": f"Model hành vi mới {path.name}: macro-F1 {m.get('macro_f1', 0):.2f}, "
                           f"dùng {len(meta.get('clips') or [])} clip/sự kiện ({n_rev.get('true', 0)} Đúng, "
                           f"{n_rev.get('false', 0)} Báo nhầm). " + ("Đã nạp, đang dùng." if reloaded else
                                                                     "Khởi động lại máy chủ để nạp.")}

    def _reload_behavior(self) -> bool:
        beh = getattr(self.rt, "behavior", None)
        ok = False
        if beh is not None and hasattr(beh, "reload_classifier"):
            try:
                ok = bool(beh.reload_classifier())
            except Exception:  # noqa: BLE001
                log.exception("Nạp model hành vi mới lỗi")
        vj = getattr(getattr(self.rt, "uplink", None), "video", None)
        if vj is not None:
            vj._models = None  # thử video lần sau nạp model mới
        return ok

    def _train_hazards(self, epochs: int) -> dict:
        from ares.hazards.train import train_all

        with contextlib.redirect_stdout(self._tail):
            res = train_all(self.cfg, epochs=max(10, min(epochs, 100)))
        if not res:
            return {"ok": False, "message": "Chưa cài model cháy/khói, hút thuốc (CAI_AI_BAO_HO.bat)."}
        if any(r.get("promoted") for r in res):
            beh = getattr(self.rt, "behavior", None)
            if beh is not None and hasattr(beh, "reload_hazards"):
                try:
                    beh.reload_hazards()
                except Exception:  # noqa: BLE001
                    log.exception("Nạp lại AI bổ sung lỗi")
            vj = getattr(getattr(self.rt, "uplink", None), "video", None)
            if vj is not None:
                vj._hazards = None
        return {"ok": any(r.get("ok") for r in res), "items": res, "message": " ".join(r["message"] for r in res)}

    def _train_person(self) -> dict:
        rt = self.rt
        if not getattr(self.cfg, "retrain_enabled", True) or not hasattr(rt, "retrainer"):
            return {"ok": False, "message": "Máy AI chưa bật train YOLO người (ARES_RETRAIN_ENABLED)."}
        with contextlib.redirect_stdout(self._tail):
            r = rt.retrainer.run("manual", force=True)
        r = r or {}
        st = str(r.get("status") or "")
        why = r.get("reason") or r.get("error") or r.get("message") or ""
        return {"ok": st not in ("failed", "busy", "skipped", ""), "status": st,
                "message": f"YOLO người: {st or 'xong'}{' — ' + str(why) if why else ''}."}
