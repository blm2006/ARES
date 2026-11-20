"""Xác nhận theo thời gian cho AI bổ sung — model ảnh đơn lẻ hay báo nhầm 1–2 khung (bụi, đèn vàng, bút cầm tay...),
nên chỉ báo khi thấy ở NHIỀU lần kiểm liên tiếp gần nhất.

* Cháy/khói (cả camera): lửa thấy ≥ fire_hits trong fire_window lần kiểm gần nhất; khói (dễ nhầm bụi gỗ / hơi nước)
  cần ≥ smoke_hits. Cùng camera không báo lại trong hazard_cooldown_s.
* Hút thuốc (theo người): điếu thuốc có tâm cách miệng (mũi) hoặc 1 cổ tay < 0,45 × chiều cao đầu-hông của người đó;
  thấy ≥ smoke_person_hits trong 6 lần kiểm gần nhất -> báo người đó.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ares.behavior.skeleton import KP_CONF

NOSE, LW, RW = 0, 9, 10


def near_person(box, kp: np.ndarray, ref_h: float, k: float = 0.45) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for j in (NOSE, LW, RW):
        if kp[j, 2] >= KP_CONF and np.hypot(kp[j, 0] - cx, kp[j, 1] - cy) <= k * max(ref_h, 1.0):
            return True
    return False


class CamHazards:
    """Trạng thái AI bổ sung của MỘT camera."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.t = -1e18  # lần kiểm gần nhất
        self.dets: dict[str, list] = {}  # mục -> hộp thấy ở lần kiểm đó (để vẽ + lưu train)
        self.hist: deque = deque(maxlen=12)  # (t, có lửa, có khói)
        self.fired_t = {"fire": -1e18}
        self.alert: tuple[float, int, str] | None = None  # (đến lúc, mức, câu) cho thanh trạng thái

    def step(self, t: float, found: dict[str, list], live: list, fire_cb, person_cb) -> None:
        cfg = self.cfg
        self.t, self.dets = t, found
        if "fire" in found:
            d = found["fire"]
            has_f = any(x.cls == "fire" for x in d)
            has_s = any(x.cls == "smoke" for x in d)
            self.hist.append((t, has_f, has_s))
            win = list(self.hist)[-int(cfg.hazard_fire_window):]
            nf, ns = sum(1 for h in win if h[1]), sum(1 for h in win if h[2])
            what = "lửa" if nf >= cfg.hazard_fire_hits else ("khói" if ns >= cfg.hazard_smoke_hits else "")
            if what:
                self.alert = (t + 3.0, 2, f"BÁO ĐỘNG: {'CHÁY (thấy lửa)' if what == 'lửa' else 'KHÓI'}")
                if t - self.fired_t["fire"] >= cfg.hazard_cooldown_s:
                    self.fired_t["fire"] = t
                    best = max(d, key=lambda x: x.conf) if d else None
                    fire_cb(what, best, d)
        if "smoking" in found:
            cig = found["smoking"]
            for tr in live:
                q: deque = tr.state.setdefault("smoke_q", deque(maxlen=6))
                hit = [c for c in cig if near_person(c.box, tr.last.kp, tr.ref_h)]
                q.append(bool(hit))
                if hit:
                    tr.state["smoke_box"] = hit[0]
                if sum(q) >= cfg.hazard_smoke_person_hits:
                    tr.state["smoking_t"] = t
                    person_cb(tr, hit[0] if hit else tr.state.get("smoke_box"), cig)
