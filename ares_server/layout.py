"""Bố trí vùng + đèn cho thử video phần Người & môi trường (thuần Python — server và máy AI dùng chung)."""

from __future__ import annotations


def clean_layout(opts: dict | None) -> dict:
    """Kiểm tra + chuẩn hoá bố trí (dùng chung cho server và máy AI)."""
    opts = opts if isinstance(opts, dict) else {}
    zones, ids = [], set()
    for k, z in enumerate((opts.get("zones") or [])[:16]):
        if not isinstance(z, dict):
            continue
        pts = []
        for p in (z.get("poly") or [])[:32]:
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            pts.append([round(min(1.0, max(0.0, x)), 4), round(min(1.0, max(0.0, y)), 4)])
        if len(pts) < 3:
            continue
        zid = "".join(c for c in str(z.get("id") or f"z{k + 1}") if c.isalnum() or c in "-_")[:12] or f"z{k + 1}"
        while zid in ids:
            zid += "x"
        ids.add(zid)
        zones.append({"id": zid, "name": str(z.get("name") or f"Vùng {k + 1}").strip()[:40] or f"Vùng {k + 1}",
                      "poly": pts, "entry": bool(z.get("entry"))})
    lights = []
    for k, li in enumerate((opts.get("lights") or [])[:24]):
        if not isinstance(li, dict):
            continue
        zs = [str(x) for x in (li.get("zones") or []) if str(x) in ids][:16]
        try:
            dev = int(li.get("device_id")) if li.get("device_id") not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            dev = None
        o = str(li.get("output") or "")
        out = o if o in ("relay", "led") or (o[:2] == "ch" and o[2:].isdigit() and 1 <= int(o[2:]) <= 8) else "relay"  # chN = kênh đèn của mạch
        try:
            level = int(max(1, min(100, float(li.get("level") or 100))))
        except (TypeError, ValueError):
            level = 100
        lid = "".join(c for c in str(li.get("id") or f"l{k + 1}") if c.isalnum() or c in "-_")[:12] or f"l{k + 1}"
        mode = li.get("mode") if li.get("mode") in ("auto", "on", "off") else "auto"  # camera thật: Tự động / Luôn bật / Luôn tắt
        lights.append({"id": lid, "name": str(li.get("name") or f"Đèn {k + 1}").strip()[:40] or f"Đèn {k + 1}",
                       "zones": zs, "device_id": dev, "output": out, "level": level, "mode": mode})
    try:
        delay = float(opts.get("off_delay_s", 10))
    except (TypeError, ValueError):
        delay = 10.0
    try:
        mn = int(opts.get("min_people", 1))
    except (TypeError, ValueError):
        mn = 1
    return {"zones": zones, "lights": lights, "off_delay_s": round(max(0.0, min(600.0, delay)), 1),
            "min_people": max(1, min(50, mn))}
