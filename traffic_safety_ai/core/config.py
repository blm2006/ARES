"""Cấu hình: địa điểm/camera (YAML), model (YAML), đường dẫn chạy.

Mọi toạ độ trong file địa điểm là PIXEL theo `camera.frame_size`. Khi video thực tế có kích thước khác,
`SiteConfig.scaled_to()` tự co giãn toàn bộ toạ độ (ROI, vạch, điểm hiệu chuẩn) theo tỉ lệ.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from traffic_safety_ai import CONFIG_DIR, PACKAGE_ROOT

Pt = tuple[float, float]


class ConfigError(ValueError):
    """Lỗi cấu hình, thông báo tiếng Việt dễ sửa."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ======================================================================== địa điểm
class CameraCfg(_Base):
    id: str = "CAM-01"
    name: str = "Camera mẫu"
    location: str = "Chưa khai báo địa điểm"
    channel: Literal["cctv", "pov"] = "cctv"
    timezone: str = "Asia/Ho_Chi_Minh"
    frame_size: tuple[int, int] | None = None
    detect_unit: str = "Hệ thống AI giám sát ATGT khu vực trường học (thử nghiệm)"
    handling_note: str = "Dữ liệu chờ người có thẩm quyền xác minh trước khi chuyển cơ quan chức năng."


class CalibrationCfg(_Base):
    image_points: list[Pt] = Field(default_factory=list)
    world_points: list[Pt] = Field(default_factory=list)
    max_reproj_error_m: float = 0.5

    @model_validator(mode="after")
    def _check(self) -> CalibrationCfg:
        n1, n2 = len(self.image_points), len(self.world_points)
        if n1 or n2:
            if n1 != n2:
                raise ValueError(f"calibration: số điểm ảnh ({n1}) khác số điểm thực ({n2})")
            if n1 < 4:
                raise ValueError("calibration: cần ít nhất 4 cặp điểm (ảnh ↔ mét)")
        return self

    @property
    def enabled(self) -> bool:
        return len(self.image_points) >= 4


class ZoneCfg(_Base):
    id: str
    name: str = ""
    polygon: list[Pt]
    direction: list[Pt] | None = None            # mũi tên [đầu, cuối] = hướng đi đúng của làn
    allowed_classes: list[str] | None = None     # làn/khu vực chỉ cho phép các loại xe này
    kind: str | None = None                      # no_parking: crosswalk | no_stopping | no_parking | sidewalk
    max_stationary_s: float | None = None
    active_hours: list[str] | None = None        # ["06:30-07:30", "16:30-17:30"]; None = luôn áp dụng

    @field_validator("polygon")
    @classmethod
    def _poly(cls, v: list[Pt]) -> list[Pt]:
        if len(v) < 3:
            raise ValueError("đa giác cần ít nhất 3 điểm")
        return v

    @field_validator("direction")
    @classmethod
    def _dir(cls, v: list[Pt] | None) -> list[Pt] | None:
        if v is not None and len(v) != 2:
            raise ValueError("direction phải gồm đúng 2 điểm [đầu, cuối]")
        return v


def _two_points(v: list[Pt] | None, what: str) -> list[Pt] | None:
    if v is not None and len(v) != 2:
        raise ValueError(f"{what} phải gồm đúng 2 điểm")
    return v


class StopLineCfg(_Base):
    id: str
    name: str = ""
    points: list[Pt]                              # vạch dừng (2 điểm)
    direction: list[Pt]                           # mũi tên hướng xe chạy qua vạch
    confirm_points: list[Pt] | None = None        # vạch xác nhận phía sau giao lộ (xe thực sự đi qua)
    confirm_distance_m: float = 6.0               # nếu không có vạch xác nhận: đi quá vạch dừng bao nhiêu mét
    light_id: str | None = None
    exempt_zones: list[list[Pt]] = Field(default_factory=list)   # vd làn rẽ phải được phép khi đèn đỏ
    applies_to: list[str] = Field(default_factory=lambda: ["motorcycle", "bicycle", "car", "bus", "truck"])

    @field_validator("points", "direction", "confirm_points")
    @classmethod
    def _pts2(cls, v: list[Pt] | None) -> list[Pt] | None:
        return _two_points(v, "vạch dừng / mũi tên / vạch xác nhận")


class SolidLineCfg(_Base):
    id: str
    name: str = ""
    points: list[Pt]

    @field_validator("points")
    @classmethod
    def _pts(cls, v: list[Pt]) -> list[Pt]:
        if len(v) < 2:
            raise ValueError("vạch liền cần ít nhất 2 điểm")
        return v


class CountingLineCfg(_Base):
    id: str
    name: str = ""
    points: list[Pt]
    direction: list[Pt]
    positive_label: str = "Chiều thuận"
    negative_label: str = "Chiều ngược"

    @field_validator("points", "direction")
    @classmethod
    def _pts2(cls, v: list[Pt]) -> list[Pt] | None:
        return _two_points(v, "vạch đếm / mũi tên")


class SpeedTrapCfg(_Base):
    id: str
    line_a: list[Pt]
    line_b: list[Pt]
    distance_m: float = Field(gt=0)
    direction: list[Pt]

    @field_validator("line_a", "line_b", "direction")
    @classmethod
    def _pts2(cls, v: list[Pt]) -> list[Pt] | None:
        return _two_points(v, "vạch đo tốc độ / mũi tên")


class TrafficLightCfg(_Base):
    id: str
    name: str = ""
    roi: tuple[float, float, float, float]
    layout: Literal["vertical", "horizontal", "auto"] = "auto"

    @field_validator("roi")
    @classmethod
    def _roi(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if v[2] <= v[0] or v[3] <= v[1]:
            raise ValueError("roi phải là [x1, y1, x2, y2] với x2 > x1 và y2 > y1")
        return v


class ZonesCfg(_Base):
    sidewalks: list[ZoneCfg] = Field(default_factory=list)
    crosswalks: list[ZoneCfg] = Field(default_factory=list)
    lanes: list[ZoneCfg] = Field(default_factory=list)
    restricted: list[ZoneCfg] = Field(default_factory=list)
    no_parking: list[ZoneCfg] = Field(default_factory=list)
    sightline: list[ZoneCfg] = Field(default_factory=list)
    roadway: list[ZoneCfg] = Field(default_factory=list)
    speed_zone: list[ZoneCfg] = Field(default_factory=list)
    ego_corridor: list[ZoneCfg] = Field(default_factory=list)
    privacy_mask: list[ZoneCfg] = Field(default_factory=list)   # vùng luôn che (cửa sổ nhà dân...)


class LinesCfg(_Base):
    stop_lines: list[StopLineCfg] = Field(default_factory=list)
    solid_lines: list[SolidLineCfg] = Field(default_factory=list)
    counting: list[CountingLineCfg] = Field(default_factory=list)
    speed_traps: list[SpeedTrapCfg] = Field(default_factory=list)


class SpeedCfg(_Base):
    limit_kmh: float = 50.0                       # theo biển báo tại chỗ (TT 38/2024: khu đông dân cư 50/60)
    limits_by_class: dict[str, float] = Field(default_factory=dict)
    tolerance_kmh: float = 3.0                    # biên sai số đo — chỉ báo khi vượt ≥ 5 km/h + biên này
    min_track_s: float = 0.6


class POVCfg(_Base):
    horizon_y_ratio: float = 0.45
    camera_height_m: float = 1.2
    hfov_deg: float = 110.0
    ego_corridor: list[Pt] | None = None          # toạ độ CHUẨN HOÁ 0..1; None = hình thang mặc định
    phone_mount_roi: tuple[float, float, float, float] | None = None   # vùng giá đỡ điện thoại (chuẩn hoá 0..1)
    ego_speed_source: Literal["flow", "none"] = "flow"


class PrivacyCfg(_Base):
    blur_bystanders_in_export: bool = True
    blur_all_faces_in_export: bool = False
    store_plates_of_non_violators: bool = False
    retention_days: int = 90


class EvidenceCfg(_Base):
    pre_roll_s: float = 3.0
    post_roll_s: float = 2.0
    max_plate_wait_s: float = 6.0
    jpeg_quality: int = 92
    save_clip: bool = True
    clip_fps: float | None = 10.0                # clip bằng chứng 10 khung/s là đủ xem lại, ghi nhanh hơn nhiều
    clip_max_width: int = 960


class SiteConfig(_Base):
    camera: CameraCfg = Field(default_factory=CameraCfg)
    calibration: CalibrationCfg = Field(default_factory=CalibrationCfg)
    zones: ZonesCfg = Field(default_factory=ZonesCfg)
    lines: LinesCfg = Field(default_factory=LinesCfg)
    traffic_lights: list[TrafficLightCfg] = Field(default_factory=list)
    speed: SpeedCfg = Field(default_factory=SpeedCfg)
    pov: POVCfg = Field(default_factory=POVCfg)
    privacy: PrivacyCfg = Field(default_factory=PrivacyCfg)
    evidence: EvidenceCfg = Field(default_factory=EvidenceCfg)
    rules: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # ------------------------------------------------------------------ tiện ích
    def rule_params(self, code: str) -> dict[str, Any]:
        return dict(self.rules.get(code, {}) or {})

    def rule_enabled(self, code: str) -> bool:
        return bool(self.rule_params(code).get("enabled", True))

    def speed_limit(self, cls: str) -> float:
        return float(self.speed.limits_by_class.get(cls, self.speed.limit_kmh))

    def light_by_id(self, light_id: str | None) -> TrafficLightCfg | None:
        if not self.traffic_lights:
            return None
        if light_id is None:
            return self.traffic_lights[0]
        return next((lt for lt in self.traffic_lights if lt.id == light_id), None)

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def scaled_to(self, width: int, height: int) -> SiteConfig:
        """Trả bản sao đã co giãn toạ độ pixel sang kích thước khung hình thực tế."""
        out = copy.deepcopy(self)
        ref = self.camera.frame_size
        if ref is None:
            out.camera.frame_size = (int(width), int(height))
            return out
        sx, sy = width / float(ref[0]), height / float(ref[1])
        out.camera.frame_size = (int(width), int(height))
        if abs(sx - 1) < 1e-9 and abs(sy - 1) < 1e-9:
            return out

        def sp(pts: list[Pt] | None) -> list[Pt] | None:
            if pts is None:
                return None
            return [(float(x) * sx, float(y) * sy) for x, y in pts]

        out.calibration.image_points = sp(out.calibration.image_points) or []
        for group in ("sidewalks", "crosswalks", "lanes", "restricted", "no_parking", "sightline", "roadway",
                      "speed_zone", "ego_corridor", "privacy_mask"):
            for z in getattr(out.zones, group):
                z.polygon = sp(z.polygon) or []
                z.direction = sp(z.direction)
        for s in out.lines.stop_lines:
            s.points = sp(s.points) or []
            s.direction = sp(s.direction) or []
            s.confirm_points = sp(s.confirm_points)
            s.exempt_zones = [sp(z) or [] for z in s.exempt_zones]
        for s in out.lines.solid_lines:
            s.points = sp(s.points) or []
        for c in out.lines.counting:
            c.points = sp(c.points) or []
            c.direction = sp(c.direction) or []
        for tr in out.lines.speed_traps:
            tr.line_a = sp(tr.line_a) or []
            tr.line_b = sp(tr.line_b) or []
            tr.direction = sp(tr.direction) or []
        for lt in out.traffic_lights:
            x1, y1, x2, y2 = lt.roi
            lt.roi = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        # đa giác pixel nằm trong tham số rule (lối vào được miễn của RB-04) cũng phải co giãn theo
        for params in out.rules.values():
            if isinstance(params, dict) and isinstance(params.get("exempt_zones"), list):
                params["exempt_zones"] = [[(float(x) * sx, float(y) * sy) for x, y in poly]
                                          for poly in params["exempt_zones"]]
        return out


# ======================================================================== model
class ModelSpec(_Base):
    enabled: bool = True
    weights: str | None = None
    fallback: list[str] = Field(default_factory=list)
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    classes: dict[str, str] | None = None        # tên lớp của model → tên chuẩn của hệ thống
    every_n: int = 1                             # chạy mỗi n frame (model tầng 2)
    max_crops: int = 32
    backend: str | None = None                   # plate_ocr: yolo_chars | fast_plate_ocr | easyocr | none; model detect: opencv (chạy .onnx bằng OpenCV DNN, không cần PyTorch)
    model_name: str | None = None                # tên model của backend ngoài (vd fast-plate-ocr)
    # model tầng 2 trên ảnh cắt: tối đa bao nhiêu lần/giây cho MỖI đối tượng (0 = không giới hạn). Luật chỉ cần
    # ≥ 3 lần nhìn trong 1,2–1,5 s, nên 6 lần/s là thừa; giới hạn theo giây thay vì theo khung để video 30 fps
    # không tốn gấp 3 lần video 10 fps.
    max_hz: float = 6.0
    # chỉ dùng cho model nền: "auto" = chỉ chạy trong vùng bao các vùng/vạch đã khai báo (+ lề), "full" = cả khung,
    # hoặc [x1, y1, x2, y2]. Xem core/regions.py.
    roi: str | list[int] | None = "auto"
    # chỉ dùng cho model nền: chia vùng thành [cột, hàng] ô chồng lấn để bắt vật nhỏ ở xa (kiểu SAHI). Tốn thời
    # gian ≈ (số ô + 1) lần → nên dùng khi có GPU. None = tắt.
    tiles: list[int] | None = None
    tile_overlap: float = 0.2


class ModelsConfig(_Base):
    device: str = "auto"
    half: bool = True
    # ghi đè ngưỡng tracker (ByteTrack) cho hợp với độ tin của model: model nhỏ (yolov8n) cho điểm thấp hơn
    # → hạ high_thresh / new_track_thresh, nếu không nhiều đối tượng sẽ không bao giờ được khởi tạo track
    tracker: dict[str, Any] = Field(default_factory=dict)
    base: ModelSpec = Field(default_factory=lambda: ModelSpec(weights="yolo11s.pt", fallback=["yolov8n.pt"],
                                                              imgsz=1280, conf=0.25))
    pose: ModelSpec = Field(default_factory=lambda: ModelSpec(weights="yolo11n-pose.pt", imgsz=320, every_n=2))
    helmet: ModelSpec = Field(default_factory=lambda: ModelSpec(weights="models/helmet.pt", imgsz=224, every_n=2))
    phone: ModelSpec = Field(default_factory=lambda: ModelSpec(weights=None, imgsz=320, conf=0.25, every_n=2))
    plate: ModelSpec = Field(default_factory=lambda: ModelSpec(weights="models/plate.pt", imgsz=416, conf=0.3,
                                                               every_n=2))
    plate_ocr: ModelSpec = Field(default_factory=lambda: ModelSpec(backend="yolo_chars",
                                                                   weights="models/plate_chars.pt", imgsz=256,
                                                                   conf=0.3))
    cargo: ModelSpec = Field(default_factory=lambda: ModelSpec(weights="models/cargo.pt", imgsz=640))
    lane_seg: ModelSpec = Field(default_factory=lambda: ModelSpec(weights="models/lane_seg.pt", imgsz=640,
                                                                  every_n=2))
    orientation: ModelSpec = Field(default_factory=lambda: ModelSpec(weights="models/vehicle_orientation.pt",
                                                                     imgsz=224))
    turn_signal: ModelSpec = Field(default_factory=lambda: ModelSpec(weights="models/turn_signal.pt", imgsz=224,
                                                                     enabled=False))


# ======================================================================== nạp file
def _format_validation(e: ValidationError, path: Path) -> str:
    lines = [f"Cấu hình sai trong {path}:"]
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        lines.append(f"  - {loc or '(gốc)'}: {err.get('msg')}")
    return "\n".join(lines)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Không tìm thấy file cấu hình: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"File YAML lỗi cú pháp ({path}): {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"File {path} phải là một bảng (mapping) YAML")
    return data


def load_site_config(path: str | Path) -> SiteConfig:
    p = Path(path)
    data = _read_yaml(p)
    try:
        return SiteConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(_format_validation(e, p)) from e


def load_models_config(path: str | Path | None = None) -> ModelsConfig:
    p = Path(path) if path else CONFIG_DIR / "models.yaml"
    if not p.exists():
        return ModelsConfig()
    data = _read_yaml(p)
    try:
        return ModelsConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(_format_validation(e, p)) from e


def save_site_config(cfg: SiteConfig, path: str | Path) -> None:
    data = cfg.model_dump(mode="json", exclude_none=True)
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=110)
    Path(path).write_text(text, encoding="utf-8")


def resolve_weights_path(name: str | None) -> str | None:
    """Đường dẫn tương đối được hiểu theo thư mục gói; tên trơn (yolo11s.pt) để ultralytics tự tải."""
    if not name:
        return None
    p = Path(name)
    if p.is_absolute():
        return str(p)
    for base in (PACKAGE_ROOT, PACKAGE_ROOT.parent, Path.cwd()):
        cand = base / p
        if cand.exists():
            return str(cand)
    if "/" in name or "\\" in name:
        return str(PACKAGE_ROOT / p)     # để báo "không tìm thấy" kèm đường dẫn đầy đủ
    return name


# ======================================================================== đường dẫn chạy
class AppPaths:
    """Thư mục dữ liệu: CSDL, bằng chứng, báo cáo. Mặc định <gói>/data, đổi bằng TSA_DATA_DIR."""

    def __init__(self, data_dir: str | Path | None = None):
        root = data_dir or os.environ.get("TSA_DATA_DIR") or (PACKAGE_ROOT / "data")
        self.root = Path(root).resolve()
        self.db = self.root / "traffic.db"
        self.evidence = self.root / "evidence"
        self.reports = self.root / "reports"
        self.runs = self.root / "runs"

    def ensure(self) -> AppPaths:
        for p in (self.root, self.evidence, self.reports, self.runs):
            p.mkdir(parents=True, exist_ok=True)
        return self
