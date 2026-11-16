"""Kịch bản mô phỏng (không cần camera, không cần model) để chạy thử TOÀN BỘ pipeline và xem đầu ra.

Ảnh do mô phỏng tự vẽ (đường, vỉa hè, vạch, đèn tín hiệu, xe, biển số). Bộ "nhận thức" giả trả về hộp gần đúng
(có nhiễu, có lúc bỏ sót), nhãn mũ/điện thoại/hàng hoá và các lần đọc biển số CÓ LỖI OCR — để kiểm tra đúng các
khâu tracking, rule, hợp nhất biển số, bằng chứng, báo cáo. Đèn tín hiệu và xi-nhan được nhận từ PIXEL thật.

LƯU Ý: kết quả mô phỏng chỉ để kiểm tra logic, KHÔNG phản ánh độ chính xác trên video thật.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from traffic_safety_ai.core.config import (CalibrationCfg, CameraCfg, CountingLineCfg, SiteConfig, StopLineCfg,
                                           TrafficLightCfg, ZoneCfg)
from traffic_safety_ai.core.types import Detection, FrameContext
from traffic_safety_ai.lpr.fusion import PlateRead
from traffic_safety_ai.lpr.vn_plate import normalize_text, parse_plate
from traffic_safety_ai.utils.geometry import iou

W, H = 1280, 720
CONFUSE = {"0": "D", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "4": "A", "B": "8", "S": "5", "D": "0",
           "G": "6", "A": "4", "Z": "2"}
COLORS = {"car": (180, 120, 60), "bus": (60, 150, 200), "truck": (90, 90, 160), "motorcycle": (40, 40, 40),
          "bicycle": (30, 140, 30), "person": (90, 60, 200)}


@dataclass
class Actor:
    aid: int
    cls: str
    t0: float
    t1: float
    path: Any                         # hàm t → (x, y) mét (CCTV) hoặc (X, Zw) (POV)
    size: tuple[float, float]         # (rộng m, cao m)
    plate: str | None = None
    riders: int = 0
    child_rider: bool = False
    helmet: bool = True
    phone: bool = False
    cargo: tuple[float, float] | None = None     # (tỉ lệ bề rộng, bề rộng m)
    blink: str | None = None          # "left" | "right" (POV: vẽ đèn nháy thật)
    color: tuple[int, int, int] | None = None
    rider_ids: list[int] = field(default_factory=list)


def _lin(t0: float, p0: tuple[float, float], v: tuple[float, float]):
    return lambda t: (p0[0] + v[0] * (t - t0), p0[1] + v[1] * (t - t0))


def _brake(t0: float, p0: tuple[float, float], v0: float, t_brake: float, decel: float, direction: int = 1):
    def f(t: float) -> tuple[float, float]:
        dt = t - t0
        if dt <= t_brake:
            return p0[0], p0[1] + direction * v0 * dt
        tb = dt - t_brake
        t_stop = v0 / decel
        tb = min(tb, t_stop)
        s = v0 * t_brake + v0 * tb - 0.5 * decel * tb * tb
        return p0[0], p0[1] + direction * s
    return f


def _profile(t0: float, t1: float, x: float, y0: float, direction: int, v0: float,
             plan: list[tuple[float, float, float]], dt: float = 0.005):
    """Chuyển động thẳng theo kế hoạch tốc độ: plan = [(từ_thời_điểm, tốc_độ_đích m/s, gia_tốc |m/s²|)].

    Tăng/giảm tốc mượt (không "dừng tức thì" như hàm bậc thang) → mô phỏng sát thực tế hơn.
    """
    ts = np.arange(t0, t1 + dt, dt)
    ys = np.empty_like(ts)
    v, y = v0, y0
    plan = sorted(plan)
    for i, t in enumerate(ts):
        ys[i] = y
        target = None
        for tf, vt, a in plan:
            if t >= tf:
                target, rate = vt, a
        if target is not None:
            v = min(target, v + rate * dt) if v < target else max(target, v - rate * dt)
        y += direction * v * dt
    return lambda t: (x, float(np.interp(t, ts, ys)))


def _brake_point(v: float, decel: float, y_stop: float, y0: float, t0: float) -> float:
    """Thời điểm bắt đầu phanh (giảm tốc đều `decel`) để dừng đúng tại y_stop."""
    d_brake = v * v / (2 * decel)
    return t0 + max(0.0, abs(y_stop - y0) - d_brake) / v


# ======================================================================== CCTV
class CCTVSim:
    """Đường 2 chiều trước cổng trường, nhìn từ trên cao (phối cảnh)."""

    LIGHT_ROI = (1190, 30, 1226, 132)

    def __init__(self, fps: float = 15.0, duration: float = 46.0, seed: int = 7):
        self.fps = fps
        self.duration = duration
        self.rng = np.random.default_rng(seed)
        src = np.float32([[-3, 0], [17, 0], [17, 42], [-3, 42]])
        dst = np.float32([[-60, 720], [1340, 720], [820, 170], [470, 170]])
        self.Hm = cv2.getPerspectiveTransform(src, dst)
        self.actors = self._script()
        self.bg = self._background()

    # ---------------------------------------------------------------- toạ độ
    def to_img(self, pts: list[tuple[float, float]]) -> np.ndarray:
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.Hm).reshape(-1, 2)

    def px_per_m(self, x: float, y: float) -> float:
        a, b = self.to_img([(x - 0.5, y), (x + 0.5, y)])
        return float(np.linalg.norm(b - a))

    def poly(self, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(float(u), float(v)) for u, v in self.to_img(pts)]

    def light_state(self, t: float) -> str:
        if t < 10:
            return "green"
        if t < 13:
            return "yellow"
        if t < 25:
            return "red"
        return "green"

    # ---------------------------------------------------------------- kịch bản
    def _script(self) -> list[Actor]:
        A: list[Actor] = []
        a = lambda *x, **k: A.append(Actor(len(A) + 1, *x, **k))  # noqa: E731
        kmh = lambda v: v / 3.6  # noqa: E731
        a("car", 1.0, 4.0, _lin(1.0, (10.5, -1), (0, kmh(62))), (1.8, 1.5), plate="51F67890")             # RB-10
        a("motorcycle", 3.0, 10.0, _lin(3.0, (3.0, 41), (0, -kmh(24))), (0.7, 1.1), plate="59X123456", riders=4)  # RB-06
        a("motorcycle", 6.0, 12.5, _lin(6.0, (9.0, -1), (0, kmh(22))), (0.7, 1.1), plate="29AB56789", riders=3,
          child_rider=True)                                                                                  # RB-06 nghi vấn
        a("motorcycle", 9.0, 15.0, _lin(9.0, (12.5, 40), (0, -kmh(26))), (0.7, 1.1), plate="51K345678", riders=1)  # RB-03
        a("motorcycle", 11.5, 19.0, _lin(11.5, (-1.5, -1), (0, kmh(16))), (0.7, 1.1), plate="52H813579", riders=1)  # RB-04
        a("motorcycle", 13.0, 19.0, _lin(13.0, (8.5, -1), (0, kmh(30))), (0.7, 1.1), plate="60B598765", riders=1)  # RB-05
        tb = _brake_point(kmh(30), 2.5, 24.6, -1, 14.0)
        a("car", 14.0, 30.0, _profile(14.0, 30.0, 11.6, -1, 1, kmh(30), [(tb, 0.0, 2.5), (25.6, kmh(30), 2.0)]),
          (1.8, 1.5), plate="51G24680")                                                                       # dừng đèn đỏ
        a("person", 19.5, 33.0, _lin(19.5, (-1.0, 13.6), (1.2, 0)), (0.45, 1.25))                               # trẻ qua đường
        a("car", 21.6, 26.0, _lin(21.6, (5.0, 41), (0, -kmh(40))), (1.8, 1.5), plate="51H11223")              # RB-07
        tp = _brake_point(kmh(18), 1.5, 8.0, 41.0, 13.0)
        a("car", 13.0, 46.0, _profile(13.0, 46.0, 1.6, 41.0, -1, kmh(18), [(tp, 0.0, 1.5)]), (1.8, 1.5),
          plate="30E55566")                                                                                   # RB-11
        for k, x in enumerate((8.4, 9.4, 10.4)):
            a("bicycle", 27.0, 38.0, _lin(27.0, (x, -1), (0, kmh(14))), (0.55, 1.05), riders=1)                  # RB-12
        a("person", 29.5, 42.0, _lin(29.5, (15.0, 21.0), (-1.3, 0)), (0.5, 1.65))                              # HZ-04
        a("motorcycle", 32.5, 39.0, _lin(32.5, (2.5, 41), (0, -kmh(27))), (0.7, 1.1), plate="65F177788", riders=1,
          helmet=False)                                                                                       # RB-01
        a("motorcycle", 34.0, 41.0, _lin(34.0, (9.6, -1), (0, kmh(20))), (0.7, 1.1), plate="43C186868", riders=1,
          phone=True)                                                                                         # RB-02
        a("motorcycle", 35.0, 43.0, _lin(35.0, (4.6, 41), (0, -kmh(20))), (0.7, 1.1), plate="61D290909", riders=1,
          cargo=(2.2, 1.7))                                                                                   # RB-09
        a("car", 38.0, 44.0, _brake(38.0, (12.3, -1), kmh(42), 1.2, 7.5), (1.8, 1.5), plate="51K34567")       # RB-08
        # xe chấp hành bình thường làm nền
        a("motorcycle", 0.0, 7.0, _lin(0.0, (5.5, 41), (0, -kmh(22))), (0.7, 1.1), plate="59Y167890", riders=2)
        a("car", 26.0, 33.0, _lin(26.0, (3.5, 41), (0, -kmh(28))), (1.8, 1.5), plate="51A99887")
        a("motorcycle", 40.0, 46.0, _lin(40.0, (10.0, -1), (0, kmh(25))), (0.7, 1.1), plate="59Z124680", riders=1)
        nid = 1000
        for ac in A:
            ac.rider_ids = list(range(nid, nid + ac.riders))
            nid += max(ac.riders, 0)
        return A

    # ---------------------------------------------------------------- cảnh cho pipeline
    def scene(self) -> SiteConfig:
        s = SiteConfig()
        s.camera = CameraCfg(id="SIM-CCTV", name="Cổng trường mô phỏng", location="Đường mô phỏng trước cổng trường",
                             channel="cctv", frame_size=(W, H),
                             detect_unit="Hệ thống AI giám sát ATGT khu vực trường học (MÔ PHỎNG)")
        cal_w = [(0, 2), (14, 2), (14, 30), (0, 30)]
        s.calibration = CalibrationCfg(image_points=self.poly(cal_w), world_points=cal_w)
        z = s.zones
        z.sidewalks.append(ZoneCfg(id="VH-T", name="Vỉa hè trái (phía cổng trường)", polygon=self.poly([(-3, 0), (0, 0), (0, 42), (-3, 42)])))
        z.sidewalks.append(ZoneCfg(id="VH-P", name="Vỉa hè phải", polygon=self.poly([(14, 0), (17, 0), (17, 42), (14, 42)])))
        z.crosswalks.append(ZoneCfg(id="VACH", name="Vạch qua đường trước cổng", polygon=self.poly([(0, 12), (14, 12), (14, 15.2), (0, 15.2)])))
        z.lanes.append(ZoneCfg(id="PHAI", name="Chiều đi ra (xa camera)", polygon=self.poly([(7, 0), (14, 0), (14, 42), (7, 42)]),
                               direction=self.poly([(10.5, 2), (10.5, 9)])))
        z.lanes.append(ZoneCfg(id="TRAI", name="Chiều đi vào (về phía camera)", polygon=self.poly([(0, 0), (7, 0), (7, 42), (0, 42)]),
                               direction=self.poly([(3.5, 9), (3.5, 2)])))
        z.no_parking.append(ZoneCfg(id="CONG", name="Trước cổng trường", kind="no_stopping", max_stationary_s=12,
                                    polygon=self.poly([(0, 5), (3.2, 5), (3.2, 11), (0, 11)])))
        z.roadway.append(ZoneCfg(id="LD", name="Lòng đường", polygon=self.poly([(0, 0), (14, 0), (14, 42), (0, 42)])))
        z.sightline.append(ZoneCfg(id="TN", name="Tầm nhìn lối ra cổng", polygon=self.poly([(0, 4), (3.2, 4), (3.2, 12), (0, 12)])))
        s.lines.stop_lines.append(StopLineCfg(id="VD", name="Vạch dừng nút giao", points=self.poly([(7, 26), (14, 26)]),
                                              direction=self.poly([(10.5, 24), (10.5, 28)]),
                                              confirm_points=self.poly([(7, 33), (14, 33)]), light_id="NG1"))
        s.lines.counting.append(CountingLineCfg(id="MC1", name="Mặt cắt trước cổng", points=self.poly([(-3, 8), (17, 8)]),
                                                direction=self.poly([(7, 6), (7, 10)]), positive_label="Đi ra",
                                                negative_label="Đi vào"))
        s.traffic_lights.append(TrafficLightCfg(id="NG1", name="Đèn nút giao", roi=self.LIGHT_ROI, layout="vertical"))
        s.speed.limit_kmh = 40
        s.evidence.pre_roll_s = 2.0
        s.evidence.post_roll_s = 1.5
        s.evidence.max_plate_wait_s = 4.0
        return s

    # ---------------------------------------------------------------- vẽ
    def _background(self) -> np.ndarray:
        img = np.full((H, W, 3), (95, 120, 85), np.uint8)
        cv2.fillPoly(img, [self.to_img([(-3, 0), (0, 0), (0, 42), (-3, 42)]).astype(np.int32)], (175, 175, 170))
        cv2.fillPoly(img, [self.to_img([(14, 0), (17, 0), (17, 42), (14, 42)]).astype(np.int32)], (175, 175, 170))
        cv2.fillPoly(img, [self.to_img([(0, 0), (14, 0), (14, 42), (0, 42)]).astype(np.int32)], (70, 70, 72))
        for y in np.arange(0, 42, 3.0):
            a, b = self.to_img([(7, y), (7, y + 1.5)]).astype(int)
            cv2.line(img, tuple(a), tuple(b), (230, 230, 230), 2)
        for x in np.arange(0.4, 14, 1.0):
            cv2.fillPoly(img, [self.to_img([(x, 12), (x + 0.5, 12), (x + 0.5, 15.2), (x, 15.2)]).astype(np.int32)], (225, 225, 225))
        a, b = self.to_img([(7, 26), (14, 26)]).astype(int)
        cv2.line(img, tuple(a), tuple(b), (240, 240, 240), 3)
        gate = self.to_img([(-3, 6), (-2.6, 6), (-2.6, 10), (-3, 10)]).astype(np.int32)
        cv2.fillPoly(img, [gate], (40, 90, 160))
        x1, y1, x2, y2 = self.LIGHT_ROI
        cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (25, 25, 25), -1)
        cv2.line(img, ((x1 + x2) // 2, y2 + 4), ((x1 + x2) // 2, y2 + 140), (60, 60, 60), 4)
        return img

    def _draw_light(self, img: np.ndarray, state: str) -> None:
        x1, y1, x2, y2 = self.LIGHT_ROI
        cx = (x1 + x2) // 2
        hh = (y2 - y1) / 3
        for i, (name, col) in enumerate((("red", (30, 30, 235)), ("yellow", (0, 200, 255)), ("green", (110, 230, 40)))):
            cy = int(y1 + hh * (i + 0.5))
            on = state == name
            cv2.circle(img, (cx, cy), int(hh * 0.38), col if on else (45, 45, 45), -1)

    def actor_box(self, ac: Actor, t: float) -> np.ndarray | None:
        if not (ac.t0 <= t <= ac.t1):
            return None
        x, y = ac.path(t)
        if y < -1.5 or y > 42.5:
            return None
        s = self.px_per_m(x, y)
        u, v = self.to_img([(x, y)])[0]
        w = ac.size[0] * s * (1.6 if ac.cls in ("car", "bus", "truck") else 1.0)
        h = ac.size[1] * s * (1.25 if ac.cls in ("car", "bus", "truck") else 1.0)
        if ac.cls in ("motorcycle", "bicycle"):
            h *= 0.55
        return np.array([u - w / 2, v - h, u + w / 2, v])

    def rider_boxes(self, ac: Actor, mbox: np.ndarray, t: float) -> list[np.ndarray]:
        out = []
        x1, y1, x2, y2 = mbox
        w, h = x2 - x1, y2 - y1
        x, y = ac.path(t)
        vy = ac.path(t + 0.1)[1] - y
        fwd = 1 if vy < 0 else -1          # xe đi về phía camera → người lái ở phía dưới khung
        for k in range(ac.riders):
            ph = h * (1.85 if not (ac.child_rider and k == ac.riders - 1) else 1.05)
            off = fwd * (-k) * 0.16 * h
            cx = (x1 + x2) / 2
            bot = y2 - 0.12 * h + off
            out.append(np.array([cx - 0.36 * w, bot - ph, cx + 0.36 * w, bot]))
        return out

    def render(self, t: float) -> tuple[np.ndarray, list[tuple[int, str, np.ndarray, Actor]]]:
        img = self.bg.copy()
        self._draw_light(img, self.light_state(t))
        objs: list[tuple[int, str, np.ndarray, Actor]] = []
        items = []
        for ac in self.actors:
            b = self.actor_box(ac, t)
            if b is not None:
                items.append((b[3], ac, b))
        for _, ac, b in sorted(items, key=lambda x: x[0]):
            col = ac.color or COLORS[ac.cls]
            x1, y1, x2, y2 = b.astype(int)
            if ac.cls in ("car", "bus", "truck"):
                cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)
                cv2.rectangle(img, (x1 + 4, y1 + 3), (x2 - 4, y1 + (y2 - y1) // 3), (170, 200, 220), -1)
            elif ac.cls == "person":
                cv2.rectangle(img, (x1, y1 + (y2 - y1) // 5), (x2, y2), col, -1)
                cv2.circle(img, ((x1 + x2) // 2, y1 + (y2 - y1) // 9), max(2, (x2 - x1) // 2), (150, 180, 230), -1)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)
                if ac.cargo:
                    cw = (x2 - x1) * ac.cargo[0]
                    cx = (x1 + x2) / 2
                    cv2.rectangle(img, (int(cx - cw / 2), y1 - (y2 - y1)), (int(cx + cw / 2), y1 + (y2 - y1) // 3), (40, 120, 180), -1)
            objs.append((ac.aid, ac.cls, b, ac))
            if ac.plate:
                self._draw_plate(img, ac, b)
            if ac.riders:
                for k, rb in enumerate(self.rider_boxes(ac, b, t)):
                    rx1, ry1, rx2, ry2 = rb.astype(int)
                    cv2.rectangle(img, (rx1, ry1 + (ry2 - ry1) // 5), (rx2, ry2), (150, 80, 40), -1)
                    head = (180, 180, 180) if (ac.helmet or k > 0) else (30, 30, 30)
                    cv2.circle(img, ((rx1 + rx2) // 2, ry1 + (ry2 - ry1) // 9), max(2, (rx2 - rx1) // 2), head, -1)
                    objs.append((ac.rider_ids[k], "person", rb, ac))
        return img, objs

    def _draw_plate(self, img: np.ndarray, ac: Actor, b: np.ndarray) -> None:
        p = parse_plate(ac.plate, ac.cls)
        x1, y1, x2, y2 = b
        w = x2 - x1
        pw = max(18, w * (0.55 if ac.cls == "car" else 0.8))
        ph = pw / (4.2 if ac.cls == "car" else 1.4)
        cx, by = (x1 + x2) / 2, y2 - 0.08 * (y2 - y1)
        px1, py1, px2, py2 = int(cx - pw / 2), int(by - ph), int(cx + pw / 2), int(by)
        cv2.rectangle(img, (px1, py1), (px2, py2), (245, 245, 245), -1)
        cv2.rectangle(img, (px1, py1), (px2, py2), (20, 20, 20), 1)
        text = (p.display if p else ac.plate).replace("Đ", "D")
        lines = text.split(" ") if ac.cls != "car" else [text]
        for i, ln in enumerate(lines):
            scale = max(0.2, (py2 - py1) / (28.0 * len(lines)))
            cv2.putText(img, ln, (px1 + 2, int(py1 + (i + 0.85) * (py2 - py1) / len(lines))), cv2.FONT_HERSHEY_PLAIN,
                        scale, (15, 15, 15), 1, cv2.LINE_AA)

    def frames(self):
        n = int(self.duration * self.fps)
        for i in range(n):
            t = i / self.fps
            img, objs = self.render(t)
            yield i, t, img, objs


# ======================================================================== POV
class POVSim:
    """Camera hành trình gắn trên xe máy chạy thẳng; đường có chấm vân để optical flow bám được."""

    def __init__(self, fps: float = 12.0, duration: float = 30.0, v_ego: float = 8.0, seed: int = 3):
        self.fps = fps
        self.duration = duration
        self.v = v_ego
        self.f = (W / 2) / math.tan(math.radians(110) / 2)
        self.cam_h = 1.2
        self.cy = 0.45 * H
        rng = np.random.default_rng(seed)
        self.dots = np.c_[rng.uniform(-7, 7, 2600), rng.uniform(0, 300, 2600)]
        self.dot_col = rng.integers(40, 200, 2600)
        self.actors = self._script()

    def proj(self, X: float, Z: float, h: float = 0.0) -> tuple[float, float]:
        return W / 2 + self.f * X / Z, self.cy + self.f * (self.cam_h - h) / Z

    def ego_s(self, t: float) -> float:
        return self.v * t

    def _script(self) -> list[Actor]:
        A: list[Actor] = []
        a = lambda *x, **k: A.append(Actor(len(A) + 1, *x, **k))  # noqa: E731
        v = self.v
        a("motorcycle", 2.0, 6.5, lambda t: (0.3, v * 2.0 + 60 - 6.0 * (t - 2.0)), (0.8, 1.4), plate="59X277889")  # RB-03 POV
        # xe phía trước: chạy cùng tốc độ, phanh gấp 6 m/s² xuống 1,5 m/s, rồi tăng tốc bỏ đi → TTC thấp (RB-08)
        lead = _profile(7.0, 16.0, 0.0, v * 7.0 + 22, 1, v, [(9.0, 1.5, 6.0), (11.8, 10.0, 3.0)])
        a("car", 7.0, 16.0, lead, (1.8, 1.5), plate="51F24680")                                                  # RB-08 POV
        a("truck", 13.0, 19.0, lambda t: (3.3, 150.0), (2.4, 3.2))                                              # HZ-02
        a("person", 17.4, 18.4, lambda t: (2.0 - 1.9 * (t - 17.4), 149.0), (0.45, 1.2))                         # HZ-03
        a("car", 22.0, 30.0, lambda t: (0.6 - (0.0 if t < 25 else 1.6 * (t - 25)), v * t + 16), (1.8, 1.5),
          plate="51H86420")                                                                                     # HZ-01 không xi-nhan
        a("car", 22.0, 30.0, lambda t: (3.6 + (0.0 if t < 25 else 1.4 * (t - 25)), v * t + 24), (1.8, 1.5),
          plate="51G13579", blink="right")                                                                      # có xi-nhan
        return A

    def scene(self) -> SiteConfig:
        s = SiteConfig()
        s.camera = CameraCfg(id="SIM-POV", name="Camera hành trình mô phỏng", location="Tuyến khảo sát mô phỏng",
                             channel="pov", frame_size=(W, H),
                             detect_unit="Hệ thống AI giám sát ATGT khu vực trường học (MÔ PHỎNG)")
        s.pov.horizon_y_ratio = 0.45
        s.pov.camera_height_m = self.cam_h
        s.pov.hfov_deg = 110
        s.evidence.pre_roll_s = 2.0
        s.evidence.post_roll_s = 1.5
        s.evidence.max_plate_wait_s = 3.0
        return s

    def actor_box(self, ac: Actor, t: float) -> np.ndarray | None:
        if not (ac.t0 <= t <= ac.t1):
            return None
        X, Zw = ac.path(t)
        Z = Zw - self.ego_s(t)
        if Z < 2.0 or Z > 90:
            return None
        u1, vb = self.proj(X - ac.size[0] / 2, Z)
        u2, vt = self.proj(X + ac.size[0] / 2, Z, ac.size[1])
        if u2 < -50 or u1 > W + 50:
            return None
        return np.array([u1, vt, u2, vb])

    def render(self, t: float) -> tuple[np.ndarray, list[tuple[int, str, np.ndarray, Actor]]]:
        img = np.zeros((H, W, 3), np.uint8)
        img[: int(self.cy)] = (200, 170, 140)
        img[int(self.cy):] = (70, 70, 72)
        s = self.ego_s(t)
        Z = self.dots[:, 1] - (s % 300)
        Z = np.where(Z < 1.0, Z + 300, Z)
        ok = (Z > 1.5) & (Z < 120)
        us = W / 2 + self.f * self.dots[ok, 0] / Z[ok]
        vs = self.cy + self.f * self.cam_h / Z[ok]
        for u, vv, c, zz in zip(us, vs, self.dot_col[ok], Z[ok]):
            r = max(1, int(6 / zz * 3))
            cv2.circle(img, (int(u), int(vv)), r, (int(c), int(c), int(c)), -1)
        for X in (-1.75, 1.75):
            for z0 in np.arange(-(s % 6), 90, 6.0):
                if z0 < 2:
                    continue
                a = self.proj(X, z0)
                b = self.proj(X, z0 + 3)
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (230, 230, 230), 3)
        objs = []
        items = []
        for ac in self.actors:
            b = self.actor_box(ac, t)
            if b is not None:
                items.append((b[3], ac, b))
        for _, ac, b in sorted(items, key=lambda x: x[0]):
            x1, y1, x2, y2 = b.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), ac.color or COLORS[ac.cls], -1)
            if ac.cls in ("car", "truck", "bus"):
                cv2.rectangle(img, (x1 + 3, y1 + 3), (x2 - 3, y1 + (y2 - y1) // 3), (160, 190, 210), -1)
                w, h = x2 - x1, y2 - y1
                on = ac.blink is not None and ac.t0 + 0.5 < t and int(t * 3.0) % 2 == 0     # 1,5 Hz
                for side in ("left", "right"):
                    lx1 = x1 + 2 if side == "left" else x2 - max(4, int(0.2 * w))
                    col = (0, 165, 255) if (on and ac.blink == side) else (40, 40, 140)
                    cv2.rectangle(img, (lx1, y1 + int(0.5 * h)), (lx1 + max(3, int(0.18 * w)), y1 + int(0.7 * h)), col, -1)
            objs.append((ac.aid, ac.cls, b, ac))
        return img, objs

    def frames(self):
        n = int(self.duration * self.fps)
        for i in range(n):
            t = i / self.fps
            img, objs = self.render(t)
            yield i, t, img, objs


# ======================================================================== nhận thức giả
class SimPerception:
    """Detector + model tầng 2 giả: hộp từ kịch bản (nhiễu, bỏ sót), nhãn thuộc tính, biển số có lỗi OCR."""

    def __init__(self, sim: CCTVSim | POVSim, plate_reader: Any | None = None, miss_rate: float = 0.04,
                 jitter_px: float = 1.2, seed: int = 11):
        self.sim = sim
        self.plates = plate_reader
        self.rng = np.random.default_rng(seed)
        self.miss = miss_rate
        self.jitter = jitter_px
        self._objs: list[tuple[int, str, np.ndarray, Actor]] = []
        self.last_ms = 0.0

    def set_frame_objects(self, objs: list[tuple[int, str, np.ndarray, Actor]]) -> None:
        self._objs = objs

    def capabilities(self) -> set[str]:
        return {"vehicles", "persons", "helmet", "phone", "pose", "plate_detect", "plate_ocr", "traffic_light_det",
                "cargo"}

    def detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]:
        out = []
        for oid, cls, box, _ in self._objs:
            if self.rng.random() < self.miss:
                continue
            b = box + self.rng.normal(0, self.jitter, 4)
            out.append(Detection(b, float(self.rng.uniform(0.6, 0.95)), cls, "sim", {"gt": oid}))
        return out

    def _match(self, ctx: FrameContext) -> dict[int, tuple[int, Actor, np.ndarray]]:
        m = {}
        for tr in ctx.tracks:
            best, bi = 0.3, None
            for oid, cls, box, ac in self._objs:
                v = iou(tr.xyxy, box)
                if v > best:
                    best, bi = v, (oid, ac, box)
            if bi is not None:
                m[tr.id] = bi
        return m

    def attributes(self, ctx: FrameContext, aux_dets: list[Detection]) -> None:
        from collections import deque

        match = self._match(ctx)
        for tr in ctx.tracks:
            if tr.id not in match:
                continue
            oid, ac, box = match[tr.id]
            if tr.cls == "person" and ac.cls in ("motorcycle", "bicycle") and oid in ac.rider_ids:
                k = ac.rider_ids.index(oid)
                lab = "no_helmet" if (not ac.helmet and k == 0) else "helmet"
                if self.rng.random() < 0.1:
                    lab = "unknown"
                tr.attrs.setdefault("helmet_obs", deque(maxlen=60)).append((ctx.t, lab, 0.85))
                if k == 0:
                    tr.attrs.setdefault("phone_obs", deque(maxlen=60)).append(
                        (ctx.t, bool(ac.phone and self.rng.random() < 0.85), False, 0.7))
            if tr.cls == "motorcycle" and ac.cargo and oid == ac.aid:
                r, wm = ac.cargo
                tr.attrs.setdefault("cargo_obs", deque(maxlen=60)).append(
                    (ctx.t, r + float(self.rng.normal(0, 0.05)), wm + float(self.rng.normal(0, 0.05)), "model"))
            if self.plates is not None and ac.plate and oid == ac.aid and tr.cls in ("car", "motorcycle", "bus", "truck"):
                if (ctx.frame_idx + tr.id) % 2 == 0:
                    self._plate_read(ctx, tr, ac, box)
        if isinstance(self.sim, POVSim):
            from traffic_safety_ai.core.attributes import TurnSignalTracker

            if not hasattr(self, "_blink"):
                self._blink = TurnSignalTracker()
            self._blink.update(ctx)

    def _plate_read(self, ctx: FrameContext, tr: Any, ac: Actor, box: np.ndarray) -> None:
        txt = normalize_text(ac.plate)
        h = box[3] - box[1]
        q = float(min(1.0, h / 90.0))
        chars = list(txt)
        n_err = int(self.rng.random() < 0.45) + int(self.rng.random() < 0.15)
        for _ in range(n_err):
            i = int(self.rng.integers(0, len(chars)))
            chars[i] = CONFUSE.get(chars[i], chars[i])
        raw = "".join(chars)
        x1, y1, x2, y2 = box
        w = x2 - x1
        pw = max(18, w * (0.55 if ac.cls == "car" else 0.8))
        ph = pw / (4.2 if ac.cls == "car" else 1.4)
        cx, by = (x1 + x2) / 2, y2 - 0.08 * (y2 - y1)
        pb = np.array([cx - pw / 2 - 2, by - ph - 2, cx + pw / 2 + 2, by + 2])
        X1, Y1, X2, Y2 = (int(max(0, v)) for v in pb)
        crop = ctx.frame[Y1:Y2, X1:X2].copy() if Y2 > Y1 and X2 > X1 else None
        tr.attrs["plate_box"] = pb
        self.plates.add_read(tr.id, tr.cls, PlateRead(raw, float(self.rng.uniform(0.6, 0.95)), q, ctx.frame_idx,
                                                      ctx.t, crop=crop, color="white",
                                                      layout="1line" if ac.cls == "car" else "2line"))
