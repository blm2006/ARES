"""Kiểu dữ liệu dùng chung giữa detector, tracker, rule, bằng chứng."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # tránh import vòng
    from traffic_safety_ai.core.config import SiteConfig
    from traffic_safety_ai.core.ego_motion import EgoMotion
    from traffic_safety_ai.core.homography import GroundCalibration
    from traffic_safety_ai.core.traffic_light import LightState

PERSON = "person"
VEHICLE_CLASSES: tuple[str, ...] = ("motorcycle", "bicycle", "car", "bus", "truck")
MOTOR_VEHICLES: tuple[str, ...] = ("motorcycle", "car", "bus", "truck")
FOUR_WHEELERS: tuple[str, ...] = ("car", "bus", "truck")
TWO_WHEELERS: tuple[str, ...] = ("motorcycle", "bicycle")
LARGE_VEHICLES: tuple[str, ...] = ("bus", "truck")
TRACKED_CLASSES: tuple[str, ...] = (PERSON,) + VEHICLE_CLASSES

CLASS_LABEL_VI: dict[str, str] = {
    "person": "Người",
    "motorcycle": "Xe máy",
    "bicycle": "Xe đạp",
    "car": "Ô tô",
    "bus": "Xe buýt/khách",
    "truck": "Xe tải",
    "traffic_light": "Đèn tín hiệu",
    "phone": "Điện thoại",
    "helmet": "Mũ bảo hiểm",
    "no_helmet": "Không mũ",
    "plate": "Biển số",
    "cargo": "Hàng hoá",
}

# Cách ghi "Loại phương tiện" giống trang tra cứu phạt nguội.
VEHICLE_TYPE_VI: dict[str, str] = {
    "motorcycle": "Xe máy",
    "bicycle": "Xe đạp",
    "car": "Ô tô",
    "bus": "Ô tô khách",
    "truck": "Ô tô tải",
}

SEVERITY_VI: dict[str, str] = {
    "vi_pham": "Vi phạm",
    "nghi_van": "Nghi vấn — cần xác minh",
    "nguy_co": "Nguy cơ an toàn",
}

STATUS_VI: dict[str, str] = {
    "cho_xac_minh": "Chờ xác minh",
    "da_xac_minh": "Đã xác minh",
    "loai_bo": "Loại bỏ (báo nhầm)",
}

# Cặp lớp được phép ghép khi tracking (model hay nhầm xe máy/xe đạp, ô tô/xe tải).
CLASS_COMPAT: dict[str, frozenset[str]] = {
    "person": frozenset({"person"}),
    "motorcycle": frozenset({"motorcycle", "bicycle"}),
    "bicycle": frozenset({"bicycle", "motorcycle"}),
    "car": frozenset({"car", "truck", "bus"}),
    "truck": frozenset({"truck", "car", "bus"}),
    "bus": frozenset({"bus", "truck", "car"}),
}


def classes_compatible(a: str, b: str) -> bool:
    return b in CLASS_COMPAT.get(a, frozenset({a}))


@dataclass
class Detection:
    xyxy: np.ndarray
    conf: float
    cls: str
    source: str = "base"
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float64).reshape(4)
        self.conf = float(self.conf)

    @property
    def w(self) -> float:
        return float(self.xyxy[2] - self.xyxy[0])

    @property
    def h(self) -> float:
        return float(self.xyxy[3] - self.xyxy[1])

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (float(self.xyxy[0] + self.xyxy[2]) / 2, float(self.xyxy[1] + self.xyxy[3]) / 2)

    @property
    def bottom(self) -> tuple[float, float]:
        return (float(self.xyxy[0] + self.xyxy[2]) / 2, float(self.xyxy[3]))


@dataclass
class TrackPoint:
    frame_idx: int
    t: float
    xyxy: np.ndarray
    conf: float
    ground_px: tuple[float, float]
    ground_m: tuple[float, float] | None = None


@dataclass
class Kinematics:
    """Động học của một track tại frame hiện tại (tính một lần, mọi rule dùng chung)."""

    vx_px: float = 0.0
    vy_px: float = 0.0
    speed_px: float = 0.0          # pixel/giây (điểm tiếp đất)
    speed_norm: float = 0.0        # "chiều cao hộp mỗi giây" — đo chuyển động không phụ thuộc khoảng cách
    vx_m: float | None = None      # m/s trên mặt đường (cần hiệu chuẩn homography)
    vy_m: float | None = None
    speed_kmh: float | None = None
    speed_sigma_kmh: float | None = None
    accel_ms2: float | None = None
    accel_sigma_ms2: float | None = None
    scale_rate: float | None = None  # d ln(chiều cao)/dt (1/s) — >0 là đang to ra (lại gần camera)
    moving: bool = False
    window_s: float = 0.0
    n: int = 0


class Track:
    """Một đối tượng được theo vết. Trạng thái Kalman do tracker quản lý."""

    def __init__(self, track_id: int, det: Detection, frame_idx: int, t: float, history_len: int = 90):
        self.id = track_id
        self.cls_votes: Counter[str] = Counter({det.cls: det.conf})
        self.xyxy = det.xyxy.copy()
        self.conf = det.conf
        self.det_attrs: dict[str, Any] = dict(det.attrs)
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.first_t = t
        self.last_t = t
        self.hits = 1
        self.time_since_update = 0
        self.state = "tentative"   # tentative | confirmed | lost | removed
        self.history: deque[TrackPoint] = deque(maxlen=history_len)
        self.kin = Kinematics()
        self.attrs: dict[str, Any] = {}   # chỗ trao đổi giữa các mô-đun (biển số, mũ, chớp đèn...)
        # Kalman
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None

    # --------------------------------------------------------------- tiện ích
    @property
    def cls(self) -> str:
        return self.cls_votes.most_common(1)[0][0]

    @property
    def is_confirmed(self) -> bool:
        return self.state == "confirmed"

    @property
    def age(self) -> int:
        """Số frame kể từ khi sinh track."""
        return self.last_frame - self.first_frame + 1

    @property
    def duration_s(self) -> float:
        return self.last_t - self.first_t

    @property
    def w(self) -> float:
        return float(self.xyxy[2] - self.xyxy[0])

    @property
    def h(self) -> float:
        return float(self.xyxy[3] - self.xyxy[1])

    @property
    def center(self) -> tuple[float, float]:
        return (float(self.xyxy[0] + self.xyxy[2]) / 2, float(self.xyxy[1] + self.xyxy[3]) / 2)

    @property
    def bottom(self) -> tuple[float, float]:
        return (float(self.xyxy[0] + self.xyxy[2]) / 2, float(self.xyxy[3]))

    @property
    def ground_m(self) -> tuple[float, float] | None:
        return self.history[-1].ground_m if self.history else None

    def point_at_or_before(self, t: float) -> TrackPoint | None:
        best = None
        for p in self.history:
            if p.t <= t:
                best = p
            else:
                break
        return best

    def recent(self, seconds: float) -> list[TrackPoint]:
        if not self.history:
            return []
        t_end = self.history[-1].t
        return [p for p in self.history if p.t >= t_end - seconds]

    def __repr__(self) -> str:  # pragma: no cover - chỉ để debug
        return f"Track(id={self.id}, cls={self.cls}, state={self.state}, box={self.xyxy.round(1).tolist()})"


@dataclass
class Overlay:
    """Lệnh vẽ lên ảnh bằng chứng / khung hiển thị."""

    kind: str                     # box | arrow | polygon | line | text | point
    pts: list[tuple[float, float]]
    color: tuple[int, int, int] = (53, 57, 229)   # BGR của #E53935
    alpha: float = 0.0
    thickness: int = 2
    text: str = ""


@dataclass
class ViolationEvent:
    rule_code: str
    rule_name: str
    severity: str                 # vi_pham | nghi_van | nguy_co
    channel: str
    frame_idx: int
    t: float
    wall_time: datetime
    track_ids: list[int]
    primary_track_id: int | None
    vehicle_cls: str | None
    bbox: np.ndarray | None
    confidence: float
    variant: str | None = None
    measurements: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    overlays: list[Overlay] = field(default_factory=list)
    key: str = ""                 # khoá chống trùng (vd "RB-05:12")
    legal: dict[str, Any] = field(default_factory=dict)   # căn cứ tham khảo (từ rules_catalog.yaml)


@dataclass
class FrameContext:
    frame_idx: int
    t: float
    wall_time: datetime
    frame: np.ndarray
    channel: str
    camera_id: str
    tracks: list[Track]
    all_tracks: dict[int, Track]
    scene: SiteConfig
    fps: float = 30.0
    calib: GroundCalibration | None = None
    ego: EgoMotion | None = None
    lights: dict[str, LightState] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)
    riders: dict[int, list[int]] = field(default_factory=dict)   # id xe → [id người], người lái đứng đầu
    rider_of: dict[int, int] = field(default_factory=dict)       # id người → id xe
    new_track_ids: set[int] = field(default_factory=set)

    @property
    def frame_size(self) -> tuple[int, int]:
        h, w = self.frame.shape[:2]
        return w, h

    def by_id(self, tid: int) -> Track | None:
        return self.all_tracks.get(tid)

    def of_class(self, *classes: str) -> list[Track]:
        return [t for t in self.tracks if t.cls in classes]

    def vehicles(self, classes: tuple[str, ...] = VEHICLE_CLASSES) -> list[Track]:
        return [t for t in self.tracks if t.cls in classes]

    def persons(self) -> list[Track]:
        return [t for t in self.tracks if t.cls == PERSON]

    def pedestrians(self) -> list[Track]:
        """Người không ngồi trên xe nào."""
        return [t for t in self.tracks if t.cls == PERSON and t.id not in self.rider_of]

    @property
    def light(self) -> LightState | None:
        """Đèn đầu tiên (đa số camera chỉ có một đèn điều khiển)."""
        return next(iter(self.lights.values()), None)
