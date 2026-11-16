"""Công cụ khai báo vùng/vạch/đèn/hiệu chuẩn trên một khung hình của camera → ghi file cấu hình YAML.

Cách dùng (cửa sổ OpenCV):
    python -m traffic_safety_ai calibrate --source video.mp4 --scene configs/cong_truong_A.yaml

Phím:
    1 vỉa hè      2 vạch qua đường     3 làn đường (+ mũi tên hướng đi)   4 vùng cấm dừng/đỗ
    5 vùng tầm nhìn (cổng/lối ra)      6 lòng đường       7 vùng che riêng tư   8 làn hạn chế loại xe
    s vạch dừng (+ mũi tên + vạch xác nhận tuỳ chọn)      c vạch đếm (+ mũi tên)  v cặp vạch đo tốc độ
    l ô đèn tín hiệu (2 góc)          h 4 điểm hiệu chuẩn mặt đường (hình chữ nhật đã biết kích thước)
    Chuột trái: thêm điểm   Chuột phải / Backspace: xoá điểm vừa thêm   Enter/Space: xong hình
    z: xoá hình vừa tạo    n/p: tua tới/lui 1 giây để chọn khung    w: lưu    q/Esc: thoát

Phần LOGIC (trạng thái, chuyển hình → cấu hình) tách khỏi giao diện để kiểm thử tự động.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from traffic_safety_ai.core.config import (CalibrationCfg, CountingLineCfg, SiteConfig, SpeedTrapCfg, StopLineCfg,
                                           TrafficLightCfg, ZoneCfg, save_site_config)
from traffic_safety_ai.core.homography import rectangle_world_points

Pt = tuple[float, float]

# mode → (tên hiển thị, danh sách bước: (khoá, số điểm tối thiểu, số điểm cố định hoặc None, gợi ý))
MODES: dict[str, tuple[str, list[tuple[str, int, int | None, str]]]] = {
    "1": ("Vỉa hè", [("polygon", 3, None, "click các đỉnh đa giác vỉa hè, Enter để xong")]),
    "2": ("Vạch qua đường", [("polygon", 3, None, "click các đỉnh vùng vạch qua đường")]),
    "3": ("Làn đường", [("polygon", 3, None, "click các đỉnh của làn"),
                        ("direction", 2, 2, "click 2 điểm: đầu → cuối mũi tên HƯỚNG ĐI ĐÚNG")]),
    "4": ("Vùng cấm dừng/đỗ", [("polygon", 3, None, "click các đỉnh vùng cấm dừng/đỗ (trước cổng trường...)")]),
    "5": ("Vùng tầm nhìn", [("polygon", 3, None, "click vùng cần thông thoáng (lối ra cổng, góc khuất)")]),
    "6": ("Lòng đường", [("polygon", 3, None, "click các đỉnh phần lòng đường (cho đi bộ qua đường ngoài vạch)")]),
    "7": ("Vùng che riêng tư", [("polygon", 3, None, "click vùng luôn làm mờ (cửa sổ nhà dân...)")]),
    "8": ("Làn hạn chế", [("polygon", 3, None, "click các đỉnh làn chỉ dành cho một số loại xe")]),
    "s": ("Vạch dừng", [("points", 2, 2, "click 2 đầu vạch dừng"),
                        ("direction", 2, 2, "click 2 điểm: hướng xe chạy QUA vạch"),
                        ("confirm", 0, 2, "(tuỳ chọn) 2 đầu vạch xác nhận sau giao lộ — Enter để bỏ qua")]),
    "c": ("Vạch đếm", [("points", 2, 2, "click 2 đầu vạch đếm lưu lượng"),
                       ("direction", 2, 2, "click 2 điểm: chiều được gọi là 'thuận'")]),
    "v": ("Vạch đo tốc độ", [("line_a", 2, 2, "click 2 đầu vạch A (vào)"), ("line_b", 2, 2, "click 2 đầu vạch B (ra)"),
                             ("direction", 2, 2, "click 2 điểm: chiều xe chạy từ A sang B")]),
    "l": ("Đèn tín hiệu", [("roi", 2, 2, "click góc trên-trái rồi góc dưới-phải của cụm đèn")]),
    "h": ("Hiệu chuẩn mặt đường", [("calib", 4, 4, "click 4 góc hình chữ nhật trên mặt đường: gần-trái, gần-phải, "
                                                  "xa-phải, xa-trái")]),
}
ZONE_GROUP = {"1": "sidewalks", "2": "crosswalks", "3": "lanes", "4": "no_parking", "5": "sightline",
              "6": "roadway", "7": "privacy_mask", "8": "restricted"}
COLORS = {"1": (150, 150, 150), "2": (255, 255, 255), "3": (229, 136, 30), "4": (60, 60, 220), "5": (0, 170, 255),
          "6": (90, 200, 90), "7": (120, 120, 120), "8": (200, 90, 200), "s": (240, 240, 240), "c": (255, 200, 0),
          "v": (0, 220, 255), "l": (0, 255, 0), "h": (255, 0, 255)}


@dataclass
class Shape:
    mode: str
    data: dict[str, list[Pt]]
    meta: dict[str, Any] = field(default_factory=dict)


class CalibState:
    """Trạng thái công cụ (không phụ thuộc giao diện)."""

    def __init__(self, ask: Callable[[str, str], str] | None = None):
        self.mode: str | None = None
        self.step = 0
        self.cur: list[Pt] = []
        self.data: dict[str, list[Pt]] = {}
        self.shapes: list[Shape] = []
        self.ask = ask or (lambda prompt, default: default)
        self.message = "Chọn chế độ: 1–8, s, c, v, l, h. Nhấn w để lưu."

    # ------------------------------------------------------------------ thao tác
    def set_mode(self, key: str) -> None:
        if key not in MODES:
            return
        self.mode, self.step, self.cur, self.data = key, 0, [], {}
        self.message = f"[{MODES[key][0]}] {MODES[key][1][0][3]}"

    def click(self, x: float, y: float) -> None:
        if self.mode is None:
            self.message = "Chưa chọn chế độ (1–8, s, c, v, l, h)"
            return
        self.cur.append((float(x), float(y)))
        _, steps = MODES[self.mode]
        fixed = steps[self.step][2]
        if fixed is not None and len(self.cur) >= fixed:
            self.next_step()

    def undo_point(self) -> None:
        if self.cur:
            self.cur.pop()

    def next_step(self) -> None:
        if self.mode is None:
            return
        name, steps = MODES[self.mode]
        key, min_n, fixed, _ = steps[self.step]
        if len(self.cur) < min_n:
            self.message = f"[{name}] cần ít nhất {min_n} điểm"
            return
        if self.cur:
            self.data[key] = list(self.cur)
        self.cur = []
        self.step += 1
        if self.step >= len(steps):
            self._finish()
        else:
            self.message = f"[{name}] {steps[self.step][3]}"

    def _finish(self) -> None:
        assert self.mode is not None
        name = MODES[self.mode][0]
        meta: dict[str, Any] = {}
        n = sum(1 for s in self.shapes if s.mode == self.mode) + 1
        meta["id"] = self.ask(f"Mã cho {name} (Enter = mặc định)", f"{self._prefix(self.mode)}-{n}")
        meta["name"] = self.ask(f"Tên/mô tả cho {name}", f"{name} {n}")
        if self.mode == "4":
            meta["kind"] = self.ask("Loại vùng: no_stopping | no_parking | crosswalk | sidewalk", "no_stopping")
            meta["max_stationary_s"] = float(self.ask("Dừng tối đa bao nhiêu giây thì báo", "15"))
        if self.mode == "8":
            meta["allowed"] = self.ask("Loại xe được phép (vd: bus,car)", "bus")
        if self.mode == "v":
            meta["distance_m"] = float(self.ask("Khoảng cách thực giữa 2 vạch (mét)", "10"))
        if self.mode == "h":
            meta["width_m"] = float(self.ask("Bề rộng hình chữ nhật (mét, theo phương ngang)", "3.5"))
            meta["length_m"] = float(self.ask("Chiều dài hình chữ nhật (mét, theo phương dọc đường)", "10"))
        self.shapes.append(Shape(self.mode, dict(self.data), meta))
        self.message = f"Đã thêm {name}: {meta['id']}. Chọn chế độ khác hoặc w để lưu."
        self.mode, self.step, self.cur, self.data = None, 0, [], {}

    @staticmethod
    def _prefix(mode: str) -> str:
        return {"1": "VH", "2": "VACH", "3": "LAN", "4": "CAM", "5": "TN", "6": "LD", "7": "RIENG", "8": "HC",
                "s": "VD", "c": "MC", "v": "DO", "l": "DEN", "h": "HC"}.get(mode, "X")

    def delete_last(self) -> None:
        if self.shapes:
            s = self.shapes.pop()
            self.message = f"Đã xoá {MODES[s.mode][0]} {s.meta.get('id')}"

    # ------------------------------------------------------------------ → cấu hình
    def apply_to(self, cfg: SiteConfig) -> SiteConfig:
        out = cfg.model_copy(deep=True)
        lights = []
        for s in self.shapes:
            m, d, meta = s.mode, s.data, s.meta
            if m in ZONE_GROUP:
                z = ZoneCfg(id=meta["id"], name=meta.get("name", ""), polygon=d["polygon"],
                            direction=d.get("direction"),
                            kind=meta.get("kind"), max_stationary_s=meta.get("max_stationary_s"),
                            allowed_classes=[c.strip() for c in meta["allowed"].split(",")] if meta.get("allowed") else None)
                getattr(out.zones, ZONE_GROUP[m]).append(z)
            elif m == "s":
                out.lines.stop_lines.append(StopLineCfg(id=meta["id"], name=meta.get("name", ""), points=d["points"],
                                                        direction=d["direction"], confirm_points=d.get("confirm"),
                                                        light_id=None))
            elif m == "c":
                out.lines.counting.append(CountingLineCfg(id=meta["id"], name=meta.get("name", ""), points=d["points"],
                                                          direction=d["direction"]))
            elif m == "v":
                out.lines.speed_traps.append(SpeedTrapCfg(id=meta["id"], line_a=d["line_a"], line_b=d["line_b"],
                                                          distance_m=meta["distance_m"], direction=d["direction"]))
            elif m == "l":
                (x1, y1), (x2, y2) = d["roi"]
                lt = TrafficLightCfg(id=meta["id"], name=meta.get("name", ""),
                                     roi=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                out.traffic_lights.append(lt)
                lights.append(lt.id)
            elif m == "h":
                out.calibration = CalibrationCfg(image_points=d["calib"],
                                                 world_points=rectangle_world_points(meta["width_m"], meta["length_m"]))
        if lights:
            for sl in out.lines.stop_lines:
                if sl.light_id is None:
                    sl.light_id = lights[0]
        return out


# ====================================================================== vẽ
def draw_state(img: np.ndarray, st: CalibState, scene: SiteConfig | None = None) -> np.ndarray:
    from traffic_safety_ai.utils.text import default_renderer
    from traffic_safety_ai.utils.visualizer import Visualizer

    out = img.copy()
    tr = default_renderer()
    if scene is not None:
        Visualizer(tr).draw_scene(out, scene)
    for s in st.shapes:
        col = COLORS.get(s.mode, (0, 255, 0))
        for key, pts in s.data.items():
            arr = np.array(pts, np.int32)
            if key == "polygon":
                cv2.polylines(out, [arr], True, col, 2, cv2.LINE_AA)
            elif key in ("direction",):
                cv2.arrowedLine(out, tuple(arr[0]), tuple(arr[1]), (229, 136, 30), 3, cv2.LINE_AA, tipLength=0.15)
            elif key == "roi":
                cv2.rectangle(out, tuple(arr[0]), tuple(arr[1]), col, 2)
            else:
                cv2.polylines(out, [arr], key == "calib", col, 2, cv2.LINE_AA)
        p0 = s.data[next(iter(s.data))][0]
        tr.put(out, str(s.meta.get("id")), (p0[0] + 4, p0[1] + 4), size=16, color=(255, 255, 255), bg=(0, 0, 0))
    if st.mode is not None:
        col = COLORS.get(st.mode, (0, 255, 0))
        for key, pts in st.data.items():
            cv2.polylines(out, [np.array(pts, np.int32)], key == "polygon", col, 1, cv2.LINE_AA)
        if st.cur:
            arr = np.array(st.cur, np.int32)
            cv2.polylines(out, [arr], False, col, 2, cv2.LINE_AA)
            for p in arr:
                cv2.circle(out, tuple(p), 4, col, -1)
    h = out.shape[0]
    cv2.rectangle(out, (0, h - 34), (out.shape[1], h), (20, 20, 20), -1)
    tr.put(out, st.message, (8, h - 30), size=18, color=(255, 255, 255), pad=1)
    return out


def _console_ask(prompt: str, default: str) -> str:
    try:
        v = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        v = ""
    return v or default


def run_calibrator(source: str, scene_path: str, base: str | None = None, start_s: float = 0.0,
                   max_display_w: int = 1600) -> None:
    """Mở cửa sổ khai báo. `base`: file cấu hình mẫu để kế thừa (camera, tham số rule...)."""
    from traffic_safety_ai.core.config import load_site_config

    out_path = Path(scene_path)
    if out_path.exists():
        cfg = load_site_config(out_path)
    elif base:
        cfg = load_site_config(base)
        cfg.zones = type(cfg.zones)()
        cfg.lines = type(cfg.lines)()
        cfg.traffic_lights = []
        cfg.calibration = CalibrationCfg()
    else:
        cfg = SiteConfig()
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise OSError(f"Không mở được nguồn video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pos = start_s

    def grab(t: float) -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        return fr if ok else None

    frame = grab(pos)
    if frame is None:
        raise OSError("Không đọc được khung hình")
    H, W = frame.shape[:2]
    cfg.camera.frame_size = (W, H)
    scale = min(1.0, max_display_w / float(W))
    st = CalibState(_console_ask)
    win = "Khai bao vung - traffic_safety_ai"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(ev: int, x: int, y: int, *_: Any) -> None:
        if ev == cv2.EVENT_LBUTTONDOWN:
            st.click(x / scale, y / scale)
        elif ev == cv2.EVENT_RBUTTONDOWN:
            st.undo_point()

    cv2.setMouseCallback(win, on_mouse)
    print(__doc__)
    while True:
        view = draw_state(frame, st, cfg)
        if scale < 1.0:
            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            continue
        ch = chr(k) if k < 128 else ""
        if k in (27, ord("q")):
            break
        if k in (13, 32):
            st.next_step()
        elif k == 8:
            st.undo_point()
        elif ch == "z":
            st.delete_last()
        elif ch in ("n", "p"):
            pos = max(0.0, pos + (1.0 if ch == "n" else -1.0))
            fr = grab(pos)
            if fr is not None:
                frame = fr
            st.message = f"Khung tại {pos:.1f} s (fps {fps:.0f})"
        elif ch == "w":
            new_cfg = st.apply_to(cfg)
            save_site_config(new_cfg, out_path)
            cv2.imwrite(str(out_path.with_suffix(".preview.jpg")), draw_state(frame, CalibState(), new_cfg))
            st.message = f"Đã lưu {out_path} (+ ảnh xem trước)"
            print(st.message)
        elif ch in MODES:
            st.set_mode(ch)
    cap.release()
    cv2.destroyAllWindows()


GROUP_STYLE = {"sidewalks": ("Vỉa hè", (150, 150, 150)), "crosswalks": ("Vạch qua đường", (255, 255, 255)),
               "lanes": ("Làn", (229, 136, 30)), "no_parking": ("Cấm dừng/đỗ", (60, 60, 220)),
               "sightline": ("Tầm nhìn", (0, 170, 255)), "roadway": ("Lòng đường", (90, 200, 90)),
               "privacy_mask": ("Che riêng tư", (120, 120, 120)), "restricted": ("Làn hạn chế", (200, 90, 200)),
               "speed_zone": ("Vùng tốc độ", (0, 220, 255))}


def outline_zones(img: np.ndarray, sc: SiteConfig) -> None:
    """Viền + nhãn tên cho MỌI vùng/vạch (ảnh kiểm tra cần thấy rõ từng đa giác, không chỉ mũi tên)."""
    from traffic_safety_ai.utils.text import default_renderer

    tr = default_renderer()
    for attr, (label, col) in GROUP_STYLE.items():
        for z in getattr(sc.zones, attr, []):
            pts = np.array(z.polygon, np.int32)
            cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
            c = pts.mean(axis=0)
            tr.put(img, f"{label}: {z.name or z.id}", (c[0], c[1]), size=14, color=(255, 255, 255), bg=(20, 20, 20),
                   anchor="ct", pad=2)
    for group, label, col in (("stop_lines", "Vạch dừng", (240, 240, 240)), ("counting", "Vạch đếm", (255, 200, 0))):
        for ln in getattr(sc.lines, group, []):
            a, b = ln.points[0], ln.points[1]
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), col, 2, cv2.LINE_AA)
            if getattr(ln, "direction", None):
                d0, d1 = ln.direction
                cv2.arrowedLine(img, tuple(int(v) for v in d0), tuple(int(v) for v in d1), col, 2, cv2.LINE_AA,
                                tipLength=0.2)
            tr.put(img, f"{label}: {ln.name or ln.id}", (a[0] + 4, a[1] - 20), size=14, color=(0, 0, 0), bg=col, pad=2)
    for trap in sc.lines.speed_traps:
        for pts in (trap.line_a, trap.line_b):
            cv2.line(img, tuple(int(v) for v in pts[0]), tuple(int(v) for v in pts[1]), (0, 220, 255), 2, cv2.LINE_AA)
        tr.put(img, f"Đo tốc độ {trap.id}: {trap.distance_m:g} m", tuple(trap.line_a[0]), size=14, color=(0, 0, 0),
               bg=(0, 220, 255), pad=2)
    for lt in sc.traffic_lights:
        tr.put(img, f"Đèn {lt.name or lt.id}", (lt.roi[0], lt.roi[3] + 2), size=14, color=(255, 255, 255),
               bg=(20, 20, 20), pad=2)


def preview_scene(source: str, scene: SiteConfig, out: str, at_s: float = 1.0) -> str:
    """Vẽ toàn bộ vùng/vạch của cấu hình lên một khung hình → ảnh kiểm tra (không cần màn hình)."""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise OSError(f"Không đọc được khung hình từ {source}")
    H, W = frame.shape[:2]
    sc = scene.scaled_to(W, H)
    st = CalibState()
    st.message = f"Ảnh kiểm tra cấu hình — {sc.camera.name or sc.camera.id} ({W}×{H})"
    img = draw_state(frame, st, sc)
    outline_zones(img, sc)
    if sc.calibration.enabled:
        from traffic_safety_ai.core.homography import GroundCalibration

        try:
            cal = GroundCalibration(sc.calibration.image_points, sc.calibration.world_points,
                                    sc.calibration.max_reproj_error_m)
            pts = np.array(sc.calibration.image_points, np.int32)
            cv2.polylines(img, [pts], True, (255, 0, 255), 2)
            # lưới 1 m để kiểm tra bằng mắt hiệu chuẩn có hợp lý không
            wp = np.array(sc.calibration.world_points)
            x0, y0 = wp.min(axis=0)
            x1, y1 = wp.max(axis=0)
            for yy in np.arange(np.floor(y0), np.ceil(y1) + 1):
                a, b = cal.to_image([(x0, yy), (x1, yy)])
                cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 0, 255), 1)
            for xx in np.arange(np.floor(x0), np.ceil(x1) + 1):
                a, b = cal.to_image([(xx, y0), (xx, y1)])
                cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 0, 255), 1)
        except Exception as e:  # pragma: no cover - hiển thị lỗi lên ảnh
            cv2.putText(img, f"calib error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(out, img)
    return out
