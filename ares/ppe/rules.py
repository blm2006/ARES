"""Ghép đồ bảo hộ với từng người và quyết định ai đang thiếu / đang đeo đồ bị cấm. Không có luồng, không đọc đồng hồ.

Mỗi lần có kết quả YOLO đồ bảo hộ cho 1 camera:
  1. Từ khung xương của từng người dựng các vùng: đầu, mặt, thân, hai bàn tay, hai bàn chân.
  2. Mỗi hộp đồ bảo hộ được gán cho người có vùng tương ứng chứa tâm hộp (gần nhất nếu nhiều người).
  3. Với mỗi món cần xét -> "worn" (thấy món đó) · "missing" (vùng đó THẤY RÕ, đủ to, mà không có) · "unknown"
     (quay lưng, bị che, ở quá xa) — không thấy rõ thì KHÔNG kết luận.
  4. Bỏ phiếu trong 4 giây gần nhất (model hay sót vật nhỏ nên chỉ cần 1/3 số phiếu "thấy" là coi như có đeo),
     rồi cộng dồn thời gian thiếu. Thiếu liên tục >= ppe_missing_s giây -> sự kiện "PPE".
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ares.behavior.skeleton import KP_CONF
from ares.ppe import ITEMS
from ares.ppe.detector import PPEDet

VOTE_S = 4.0  # cửa sổ bỏ phiếu
# kích thước đầu tối thiểu (pixel, sau khi quy về cỡ ảnh đưa vào model) để dám kết luận "thiếu"
MIN_HEAD_PX = {"goggles": 22.0, "mask": 20.0, "ear": 18.0, "helmet": 12.0, "gloves": 16.0, "boots": 14.0,
               "vest": 10.0, "coat": 10.0}


def _pt(kp: np.ndarray, i: int):
    return (float(kp[i, 0]), float(kp[i, 1])) if kp[i, 2] >= KP_CONF else None


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def regions(kp: np.ndarray, box) -> dict:
    """Các vùng cơ thể của 1 người. -> {"hs": cỡ đầu px, "head": hộp|None, "face": bool, "ears": bool,
    "torso": hộp|None, "hands": [(x, y)], "feet": [(x, y)]}"""
    P = [_pt(kp, i) for i in range(17)]
    nose, leye, reye, lear, rear, lsho, rsho, lelb, relb, lwri, rwri, lhip, rhip = P[:13]
    lank, rank = P[15], P[16]
    sho = None if not (lsho and rsho) else ((lsho[0] + rsho[0]) / 2, (lsho[1] + rsho[1]) / 2)
    hip = None if not (lhip or rhip) else tuple(np.mean([p for p in (lhip, rhip) if p], axis=0))
    est = []
    if lear and rear:
        est.append(_dist(lear, rear))
    if leye and reye:
        est.append(2.2 * _dist(leye, reye))
    if lsho and rsho:
        est.append(0.45 * _dist(lsho, rsho))
    if sho and hip:
        est.append(0.30 * _dist(sho, hip))
    if not est:
        est.append(0.3 * float(box[2] - box[0]))
    hs = max(est)
    head_pts = [p for p in (nose, leye, reye, lear, rear) if p]
    head = None
    if len(head_pts) >= 2:
        cx, cy = np.mean(head_pts, axis=0)
        head = (cx - hs, cy - 1.3 * hs, cx + hs, cy + hs)
    elif sho:  # nhìn từ sau lưng: ước lượng đầu ngay trên vai (đủ để xét nón / chụp tai)
        head = (sho[0] - hs, sho[1] - 2.1 * hs, sho[0] + hs, sho[1] - 0.1 * hs)
    torso = None
    if sho:
        w = max(_dist(lsho, rsho), 1.2 * hs)
        bottom = hip[1] if hip else sho[1] + 2.6 * hs
        torso = (sho[0] - 0.75 * w, sho[1] - 0.3 * hs, sho[0] + 0.75 * w, bottom + 0.4 * hs)
    hands = []
    for wri, elb in ((lwri, lelb), (rwri, relb)):
        if wri:
            hands.append((wri[0] + 0.35 * (wri[0] - elb[0]), wri[1] + 0.35 * (wri[1] - elb[1])) if elb else wri)
    feet = [(a[0], a[1] + 0.3 * hs) for a in (lank, rank) if a]
    return {"hs": hs, "head": head, "face": bool(nose and (leye or reye)), "ears": bool(lear or rear),
            "seen_head": len(head_pts) >= 2, "torso": torso, "hands": hands, "feet": feet}


def _in_box(c, b, margin: float = 0.0) -> bool:
    return b is not None and b[0] - margin <= c[0] <= b[2] + margin and b[1] - margin <= c[1] <= b[3] + margin


def _match(item: str, c, R: dict) -> float | None:
    """Tâm hộp `c` có thuộc vùng của món `item` trên người này không -> khoảng cách chuẩn hoá (nhỏ = khớp hơn) hoặc None."""
    hs = R["hs"]
    if item in ("goggles", "mask", "ear", "helmet"):
        if not _in_box(c, R["head"], 0.3 * hs):
            return None
        h = R["head"]
        return _dist(c, ((h[0] + h[2]) / 2, (h[1] + h[3]) / 2)) / hs
    if item in ("vest", "coat"):
        if not _in_box(c, R["torso"]):
            return None
        t = R["torso"]
        return _dist(c, ((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)) / hs
    pts = R["hands"] if item == "gloves" else R["feet"]
    d = min((_dist(c, p) for p in pts), default=None)
    return None if d is None or d > 1.1 * hs else d / hs


def _visible(item: str, R: dict, scale: float) -> bool:
    """Vùng của món này có thấy đủ rõ để kết luận "thiếu" không."""
    if R["hs"] * scale < MIN_HEAD_PX[item]:
        return False
    if item in ("goggles", "mask"):
        return R["face"]  # phải thấy mặt (mũi + mắt)
    if item == "ear":
        return R["ears"]
    if item == "helmet":
        return R["head"] is not None
    if item in ("vest", "coat"):
        return R["torso"] is not None
    return bool(R["hands"]) if item == "gloves" else bool(R["feet"])


def assess(people: list[tuple[np.ndarray, np.ndarray]], dets: list[PPEDet], items: list[str], scale: float = 1.0) -> list[dict]:
    """people: [(kp(17,3), box(4))]. -> mỗi người {món: "worn" | "missing" | "unknown"} cho các món trong `items`."""
    Rs = [regions(kp, box) for kp, box in people]
    worn = [set() for _ in people]
    neg = [set() for _ in people]
    for d in dets:
        if d.item not in items:
            continue
        c = ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2)
        best, bi = None, -1
        for i, R in enumerate(Rs):
            m = _match(d.item, c, R)
            if m is not None and (best is None or m < best):
                best, bi = m, i
        if bi >= 0:
            (worn if d.worn else neg)[bi].add(d.item)
    out = []
    for i, R in enumerate(Rs):
        st = {}
        for it in items:
            if it in worn[i]:
                st[it] = "worn"
            elif it in neg[i] or _visible(it, R, scale):
                st[it] = "missing"
            else:
                st[it] = "unknown"
        out.append(st)
    return out


def update_track(state: dict, t: float, status: dict, rules: dict, cfg) -> dict | None:
    """Cập nhật bộ đếm đồ bảo hộ của 1 người (state = tr.state). Trả {"missing": [...], "forbidden": [...]} khi tới
    lúc phải cảnh báo, ngược lại None. Kết quả hiển thị để ở state["ppe"]."""
    P = state.setdefault("ppe_items", {})
    last = state.get("ppe_t")
    dt = 0.0 if last is None else min(1.5, max(0.0, t - last))
    state["ppe_t"] = t
    need = float(cfg.ppe_missing_s)
    view = {"missing": [], "forbidden": [], "ok": [], "pending": []}
    due = False
    for it in rules["required"] + rules["forbidden"]:
        forbid = it in rules["forbidden"]
        S = P.setdefault(it, {"votes": deque(), "bad_s": 0.0, "fired_t": None, "hold": "unknown", "known_t": t})
        v = status.get(it, "unknown")
        if v != "unknown":
            S["votes"].append((t, v))
            S["known_t"] = t
        while S["votes"] and t - S["votes"][0][0] > VOTE_S:
            S["votes"].popleft()
        n = len(S["votes"])
        w = sum(1 for _, x in S["votes"] if x == "worn")
        if n and w * 3 >= n:
            hold = "worn"
        elif n >= 3:
            hold = "missing"
        else:
            hold = "unknown" if t - S["known_t"] > VOTE_S else S["hold"]
        S["hold"] = hold
        bad = (hold == "worn") if forbid else (hold == "missing")
        good = (hold == "missing") if forbid else (hold == "worn")
        if bad:
            S["bad_s"] += dt
        elif good:
            S["bad_s"], S["fired_t"] = 0.0, None  # đã đeo lại (hoặc đã tháo găng) -> lần sau vi phạm sẽ báo lại
        elif t - S["known_t"] > 20.0:
            S["bad_s"] = 0.0
        if bad and S["bad_s"] >= need:
            view["forbidden" if forbid else "missing"].append(it)
            if S["fired_t"] is None or t - S["fired_t"] >= float(cfg.ppe_remind_s):
                due = True
        elif bad:
            view["pending"].append(it)
        elif good and not forbid:
            view["ok"].append(it)
    view["all_ok"] = bool(rules["required"]) and len(view["ok"]) == len(rules["required"]) and not view["forbidden"]
    state["ppe"] = view
    if not due:
        return None
    for it in view["missing"] + view["forbidden"]:
        P[it]["fired_t"] = t
    again = bool(state.get("ppe_alerted"))  # người này đã từng bị nhắc: lần này là vi phạm mới / nhắc lại có chủ đích
    state["ppe_alerted"] = True
    return {"missing": list(view["missing"]), "forbidden": list(view["forbidden"]), "again": again}


def short_text(view: dict | None) -> str:
    """Chữ ngắn vẽ cạnh nhãn người: "Thiếu kính, chụp tai" / "Đeo găng tay" / ""."""
    if not view:
        return ""
    parts = []
    if view["missing"]:
        parts.append("Thiếu " + ", ".join(ITEMS[i][1] for i in view["missing"]))
    if view["forbidden"]:
        parts.append("Đeo " + ", ".join(ITEMS[i][1] for i in view["forbidden"]) + " (cấm)")
    return " · ".join(parts)
