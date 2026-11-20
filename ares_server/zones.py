"""Đèn theo vùng có người — chạy trên SERVER cho camera thật (phần Người & môi trường > Vùng & đèn).

    máy AI: YOLO người + bám vết -> điểm giữa chân từng người -> đếm người trong từng vùng của camera
            -> POST /api/v1/zones mỗi ~1 giây
    server: ZoneEngine -> đèn của vùng có người thì BẬT, vùng vắng đủ off_delay_s giây thì TẮT
            -> ghi lệnh mong muốn (relay 0/1, LED 0..100 %) vào mạch
    mạch:   ESP32/ESP8266 nhận lệnh trong câu trả lời của lần gửi số đo kế tiếp (≤ 2 giây);
            Arduino cắm USB ở máy khác nhận qua dashboard đang mở trên máy đó (Web Serial, ≤ 1 giây)

An toàn: camera mất hình / máy AI không gửi số liệu quá stale_s giây -> đèn chế độ Tự động được BẬT (không để người
ngồi trong tối vì camera hỏng). Mỗi đèn có chế độ Tự động / Luôn bật / Luôn tắt. Nhiều đèn dùng chung 1 cổng của
1 mạch: cổng bật khi có ít nhất 1 đèn bật (LED lấy độ sáng lớn nhất).
"""

from __future__ import annotations

import threading
import time

NEVER = -1e18


class ZoneEngine:
    def __init__(self, db, stale_s: float = 15.0):
        self.db, self.stale_s = db, stale_s
        self._lock = threading.Lock()
        self.layouts: dict[tuple, dict] = {}
        self.state: dict[tuple, dict] = {}
        self._sent: dict[tuple[int, str], int] = {}  # (mạch, cổng) -> giá trị đã ghi lần trước (không ghi lại mỗi giây)
        self.reload()

    # ------------------------------------------------------------------ bố trí
    def reload(self) -> None:
        with self._lock:
            self.layouts = self.db.cam_layouts()
            for k in list(self.state):
                if k not in self.layouts:
                    del self.state[k]
            for k in self.layouts:
                self._st(k)
        self.tick(time.time())

    def zones_for(self, site_id: int, edge_id: str) -> dict:
        """Gửi cho máy AI: chỉ hình dạng vùng (đèn do server quyết)."""
        return {k[2]: {"zones": [{"id": z["id"], "poly": z["poly"], "entry": z.get("entry", False)} for z in L.get("zones", [])]}
                for k, L in self.layouts.items() if k[0] == site_id and k[1] == edge_id and L.get("zones")}

    def _st(self, key) -> dict:
        st = self.state.get(key)
        if st is None:
            st = self.state[key] = {"counts": {}, "n": 0, "t": NEVER, "lights": {}}
        ids = [li["id"] for li in self.layouts.get(key, {}).get("lights", [])]
        for lid in ids:
            st["lights"].setdefault(lid, {"on": False, "last_seen": NEVER, "since": time.time(), "reason": ""})
        for lid in [x for x in st["lights"] if x not in ids]:  # đèn đã xoá khỏi bố trí
            del st["lights"][lid]
        return st

    # ------------------------------------------------------------------ số liệu từ máy AI
    def feed(self, key, counts: dict, n: int, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            if key not in self.layouts:
                return False
            st = self._st(key)
            zids = {z["id"] for z in self.layouts[key].get("zones", [])}
            st["counts"] = {z: int(max(0, min(999, int(counts.get(z, 0) or 0)))) for z in zids}
            st["n"], st["t"] = int(max(0, n or 0)), now
        self.tick(now)
        return True

    def tick(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            for key, L in self.layouts.items():
                self._step(key, L, now)
            want = self._outputs()
        self._apply(want)

    def _step(self, key, L: dict, now: float) -> None:
        st = self._st(key)
        stale = now - st["t"] > self.stale_s
        delay, mn = float(L.get("off_delay_s", 10)), int(L.get("min_people", 1))
        names = {z["id"]: z["name"] for z in L.get("zones", [])}
        for li in L.get("lights", []):
            ls = st["lights"][li["id"]]
            mode = li.get("mode", "auto")
            if mode in ("on", "off"):
                on, why = mode == "on", "Luôn bật (đặt tay)" if mode == "on" else "Luôn tắt (đặt tay)"
            elif not li["zones"]:
                on, why = False, "Chưa gán vùng"
            elif stale:
                on, why = True, "Mất hình camera — bật để an toàn"
            else:
                n = sum(st["counts"].get(z, 0) for z in li["zones"])
                if n >= mn:
                    ls["last_seen"] = now
                    on, why = True, f"{n} người ở " + ", ".join(names.get(z, z) for z in li["zones"] if st["counts"].get(z))
                else:
                    left = delay - (now - ls["last_seen"])
                    on = left >= 0
                    why = f"Vắng người — tắt sau {max(0, round(left))} giây" if on else "Vắng người"
            if on != ls["on"]:
                ls["on"], ls["since"] = on, now
            ls["reason"] = why
        st["stale"] = stale

    def _outputs(self) -> dict[tuple[int, str], int]:
        want: dict[tuple[int, str], int] = {}
        types: dict[tuple[int, str], str] = {}  # (mạch, kênh) -> led | relay
        for key, L in self.layouts.items():
            st = self.state.get(key) or {}
            for li in L.get("lights", []):
                if not li.get("device_id"):
                    continue
                out = li.get("output", "relay")
                k = (int(li["device_id"]), out)
                if out.startswith("ch") and k not in types:
                    d = self.db.device(k[0]) or {}
                    types[k] = next((c["type"] for c in d.get("channels") or [] if c.get("key") == out), "led")
                is_led = out == "led" or (out.startswith("ch") and types.get(k) == "led")
                on = (st.get("lights", {}).get(li["id"]) or {}).get("on", False)
                val = (int(li.get("level", 100)) if is_led else 1) if on else 0
                want[k] = max(want.get(k, 0), val)
        return want

    def _apply(self, want: dict[tuple[int, str], int]) -> None:
        for (did, out), val in want.items():
            if self._sent.get((did, out)) == val:
                continue
            d = self.db.device(did)
            if d is None:
                continue
            des = dict(d["desired"] or {})
            des[out] = val
            try:
                self.db.set_device(did, desired=des)
                self._sent[(did, out)] = val
            except KeyError:
                pass

    # ------------------------------------------------------------------ cho dashboard
    def view(self, key, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            L, st = self.layouts.get(key), self.state.get(key)
            if L is None or st is None:
                return {"active": False}
            return {"active": True, "counts": dict(st["counts"]), "n": st["n"],
                    "age_s": None if st["t"] == NEVER else round(now - st["t"], 1), "stale": st.get("stale", True),
                    "lights": {lid: {"on": v["on"], "reason": v["reason"], "since": v["since"]} for lid, v in st["lights"].items()}}
