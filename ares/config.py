"""Cấu hình ARES. Mọi ngưỡng/hệ số trong spec đều nằm ở đây, không hardcode nơi khác.

Đọc từ biến môi trường tiền tố ARES_ hoặc từ `.env` / `ares_config/ares.env`.
Ví dụ:  ARES_ROOM_ID=P101   ARES_KP=0.4   ARES_LED_WATTS=[60,60,60,60]
Cấu trúc (đa giác vùng, lịch lớp) nằm ở ares_config/zones.json và schedule.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ares.clock import get_tz

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "ares_config"


class AresSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARES_",
        env_file=(".env", str(CONFIG_DIR / "ares.env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ chung
    room_id: str = "P101"
    simulation_mode: bool = False  # True: cảm biến/actuator/camera giả, không cần ESP32
    timezone: str = "Asia/Ho_Chi_Minh"
    log_level: str = "INFO"
    storage_root: Path = REPO_ROOT / "var" / "ares"
    db_url: str = ""  # rỗng = sqlite trong storage_root; đổi sang postgresql://... khi cần
    zones_file: Path = CONFIG_DIR / "zones.json"
    schedule_file: Path = CONFIG_DIR / "schedule.json"

    # --------------------------------------------------------------- dashboard
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8100
    dashboard_token: str = ""  # bắt buộc nếu mở ra ngoài 127.0.0.1 (ảnh có người)

    # -------------------------------------------------------------------- MQTT
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_prefix: str = "ares"
    mqtt_client_id: str = "ares-server"
    mqtt_qos: int = Field(1, ge=0, le=2)
    # ---- Cầu nối phần cứng: "mqtt" (ESP32 + broker) hoặc "serial" (Arduino cắm USB)
    bus: str = Field("mqtt", pattern="^(mqtt|serial)$")
    serial_port: str = ""  # ví dụ COM3 (Windows) hoặc /dev/ttyACM0; rỗng = tự dò cổng Arduino
    serial_baud: int = 115200
    sensor_stale_s: float = 120.0  # quá lâu không có gói cảm biến => coi là mất dữ liệu

    # ------------------------------------------------------------------ vision
    camera_source: str = "0"  # chỉ số USB, URL RTSP, file video hoặc thư mục video
    camera_loop: bool = True  # chỉ áp dụng cho file video
    weights: str = "yolov8n.pt"
    device: str = "auto"  # auto | cpu | cuda:0
    half: bool = True
    imgsz: int = 640
    detect_conf: float = Field(0.35, ge=0.01, le=0.99)  # ngưỡng thô của YOLO (>= cận dưới dải AL)
    count_conf_min: float = Field(0.35, ge=0.01, le=0.99)  # chỉ đếm detection >= ngưỡng này
    iou: float = 0.45
    person_class_id: int = 0
    # Điều tiết FPS (workflow): bình thường chạy "full FPS" (giới hạn trên vision_fps, thực tế <= FPS camera và tốc độ YOLO);
    # trong lúc train tự hạ xuống vision_fps_training để nhường GPU/CPU cho tiến trình train. Hết train thì tự trả lại.
    vision_fps: float = Field(15.0, ge=0.2, le=60.0)
    vision_fps_training: float = Field(1.0, ge=0.1, le=60.0)
    vision_interval_s: float = Field(2.0, ge=0.2, le=30.0)  # CHỈ dùng cho chế độ mô phỏng (nhịp vision theo đồng hồ giả)
    frame_stale_s: float = 15.0  # quá lâu không có khung hình => vision "mất"
    overlay_width: int = 640
    overlay_interval_s: float = 0.2  # dựng ảnh minh hoạ cho dashboard tối đa mỗi chừng này giây (0 = mỗi khung)

    # ---------------------------------------------------------------- nhiều camera (dashboard kiểu CCTV)
    cameras_file: Path = CONFIG_DIR / "cameras.json"  # danh sách camera bổ sung; thiếu file thì tự dò camera USB
    camera_name: str = ""  # tên hiển thị của camera điều khiển (rỗng = "Phòng <room_id>")
    camera_auto_detect: bool = True  # tự dò camera USB cắm thêm (bỏ camera điều khiển)
    camera_probe_max: int = Field(6, ge=0, le=32)  # dò các chỉ số 0..N-1
    hub_fps: float = Field(5.0, ge=0.1, le=60.0)  # số vòng/giây qua MỌI camera phụ khi bình thường
    hub_fps_training: float = Field(1.0, ge=0.05, le=60.0)  # ... khi đang train
    hub_interval_s: float = Field(1.0, ge=0.1, le=30.0)  # CHỈ dùng cho chế độ mô phỏng
    hub_jpeg_width: int = 640

    # ---------------------------------------------------------- bộ đệm làm mượt
    lighting_window_s: float = 30.0  # spec: cửa sổ trượt 30 giây cho đèn
    hvac_window_s: float = 600.0  # spec: cửa sổ trượt 10 phút cho điều hòa
    buffer_max_gap_s: float = 20.0  # mẫu giữ tối đa chừng này giây, quá thì coi là thiếu dữ liệu
    min_coverage: float = 0.5  # tỉ lệ cửa sổ có dữ liệu tối thiểu để tin số trung bình

    # ------------------------------------------------------------------- đèn
    lighting_interval_s: float = Field(15.0, ge=1.0, le=60.0)  # spec: 10-30 giây
    lux_occupied: float = 400.0
    lux_dense: float = 450.0  # N_zone > dense_people: bù bóng đổ
    lux_idle: float = 100.0
    dense_people: float = 8.0
    occ_on_threshold: float = 0.35  # trung bình 30 s >= ngưỡng này (~ 1/3 số khung có người) mới coi vùng "có người"; chặn dương tính giả lẻ
    occ_off_threshold: float = 0.15  # TRỄ (hysteresis): đã "có người" thì chỉ coi là hết người khi trung bình < ngưỡng này
    light_on_delay_s: float = Field(10.0, ge=0.0, le=600.0)  # trễ BẬT: phải thấy người liên tục chừng này giây mới bật sáng vùng
    empty_dim_after_s: float = 180.0  # trễ HẠ: vùng trống liên tục > chừng này giây thì hạ về mức nền (spec 3 phút)
    room_off_after_s: float = 180.0  # trễ TẮT: CẢ phòng trống liên tục > chừng này giây thì tắt hẳn (spec 3 phút)
    lux_deadband: float = Field(30.0, ge=0.0, le=300.0)  # VÙNG CHẾT: |lux đích - lux đo| <= chừng này thì giữ nguyên PWM
    pwm_max_step: float = Field(15.0, ge=1.0, le=100.0)  # mỗi nhịp PWM đổi tối đa chừng này % (trừ bước đầu khi vừa có người)
    pwm_occupied_min: float = Field(0.0, ge=0.0, le=100.0)  # dải điều chỉnh PWM khi vùng có người
    pwm_occupied_max: float = Field(100.0, ge=0.0, le=100.0)
    idle_pwm_min: float = 10.0  # spec: 10-15 % khi vùng trống
    idle_pwm_max: float = 15.0
    kp: float = 0.5  # spec: hệ số giảm chấn
    kp_onset: float = 1.0  # bước đầu khi vùng vừa có người (đèn lên nhanh); sau đó dùng kp
    dl_max: float = 450.0  # lux tối đa LED đóng góp ở 100 % PWM
    fade_ms: int = 2000
    sensor_fault_s: float = 300.0  # cảm biến lux lỗi quá lâu => PWM dự phòng
    fallback_pwm_occupied: float = 70.0
    led_publish_min_delta: float = 2.0  # % — PWM đổi ít hơn ngần này thì không gửi lệnh mới (chỉ heartbeat)
    led_heartbeat_s: float = 60.0  # gửi lại trạng thái định kỳ để ESP32 khởi động lại vẫn đồng bộ
    led_watts: list[float] = Field(default_factory=lambda: [60.0, 60.0, 60.0, 60.0])  # ước tính điện năng

    # ---------------------------------------------------------- điều hòa (HVAC)
    hvac_interval_min: float = Field(10.0, ge=1.0, le=60.0)  # spec: 10-15 phút
    hvac_tick_s: float = 30.0  # nhịp kiểm tra kế hoạch pre-cooling/fail-safe
    t_base: float = 26.5
    people_gain: float = 2.5  # ΔT_people = gain * N_total / N_max
    rh_high: float = 70.0
    rh_delta: float = 0.5
    zone_crowd_frac: float = 0.8
    zone_crowd_delta: float = 0.5
    t_min: float = 22.0
    t_max: float = 27.0
    deadband: float = 0.7  # spec: ±0.7 °C trước khi bắn IR
    t_target: float = 25.5  # đích pre-cooling
    precool_ac_temp: int = 25  # setpoint đặt khi pre-cooling (số nguyên, <= t_target)
    gamma: float = 0.8  # phút/°C trễ nhiệt ngoài trời
    t_out_ref: float = 30.0
    r_cool_default: float = 0.10  # °C/phút, giá trị khởi đầu khi chưa học được
    r_cool_min: float = 0.02
    r_cool_alpha: float = 0.2  # EMA: new = 0.8 old + 0.2 today
    r_cool_eval_min: float = 15.0
    lead_max_min: float = 60.0
    precool_horizon_min: float = 30.0  # spec: quyết định trước giờ học 30 phút
    precool_prob: float = 0.75
    schedule_prior_prob: float = 0.9  # xác suất có người nếu lịch có lớp mà chưa đủ lịch sử thói quen
    habit_min_obs: int = 3
    failsafe_min: float = 15.0  # spec: 15 phút sau giờ vào lớp mà trống thì tắt
    hold_before_class_min: float = 20.0  # giờ giải lao: không NỚI setpoint nếu lớp mới bắt đầu trong chừng này phút
    hold_after_class_start_min: float = 15.0  # ... và trong chừng này phút đầu buổi (bộ đệm 10 phút còn nhớ lúc phòng vắng)
    empty_mean_max: float = 0.25  # trung bình 10 phút < ngưỡng này = "phòng trống" (chịu được vài khung YOLO dương tính giả)
    occ_on_people: float = 2.0  # bật điều hòa theo hiện diện khi >= số người này (trung bình 10 phút)
    turn_on_margin: float = 1.0  # bật khi T_in > setpoint + margin
    min_cmd_interval_s: float = 300.0  # giãn cách tối thiểu giữa 2 lệnh IR không khẩn cấp
    ac_min_on_s: float = Field(600.0, ge=0.0, le=7200.0)  # đã BẬT thì chạy ít nhất chừng này giây mới được tắt vì phòng trống
    ac_min_off_s: float = Field(300.0, ge=0.0, le=3600.0)  # đã TẮT thì nghỉ ít nhất chừng này giây mới bật lại (bảo vệ máy nén) — áp dụng cả pre-cool
    ac_on_delay_s: float = Field(120.0, ge=0.0, le=3600.0)  # trễ BẬT theo hiện diện: điều kiện "có người + nóng" phải giữ liên tục chừng này giây
    ac_off_delay_s: float = Field(300.0, ge=0.0, le=3600.0)  # trễ TẮT: "phòng trống" (đã là trung bình 10 phút) phải giữ thêm chừng này giây
    ac_keep_on_before_class_min: float = Field(30.0, ge=0.0, le=120.0)  # phòng trống nhưng lớp kế tiếp bắt đầu trong chừng này phút -> KHÔNG tắt (tránh tắt rồi pre-cool lại ngay)
    ac_setpoint_max_step: int = Field(2, ge=1, le=5)  # mỗi lệnh đổi setpoint tối đa chừng này °C
    fan_min: int = Field(1, ge=1, le=3)  # dải điều chỉnh quạt
    fan_max: int = Field(3, ge=1, le=3)
    fan_hysteresis: float = Field(0.05, ge=0.0, le=0.3)  # vùng chết của mức quạt theo tỉ lệ lấp đầy (0,05 = ±5 % sức chứa)
    ac_mode: str = "cool"

    # ---------------------------------------------------------- học chủ động
    al_enabled: bool = True
    al_conf_low: float = 0.35
    al_conf_high: float = 0.55
    al_daily_cap: int = 0  # 0 = KHÔNG giới hạn (spec cũ: 50/ngày). Việc giảm số lượng giờ do lọc PCA đảm nhiệm
    al_hourly_cap: int = 0  # 0 = không giới hạn
    al_min_gap_s: float = 20.0  # giãn cách tối thiểu giữa 2 ảnh của CÙNG một camera
    al_hash_dist_min: int = 6  # khác nhau < ngần này bit (aHash 64 bit) thì coi là ảnh trùng (lọc thô, trước PCA)
    al_queue_size: int = 16

    # ------------------------------------------------- lọc PCA (trước khi duyệt + trước khi train)
    pca_enabled: bool = True  # False: ảnh vào thẳng hàng chờ duyệt, không lọc (chỉ dùng khi thử nghiệm)
    pca_version: str = "ares_pca_v1"  # PCA fit MỘT lần rồi đóng băng; đổi version = fit lại + tính lại toàn bộ
    pca_components: int = 64
    pca_fit_min: int = 300  # số khung hình mẫu tối thiểu để tự khởi tạo PCA
    pca_fit_min_manual: int = 60  # khi admin bấm "Khởi tạo PCA ngay" bằng tay
    pca_corpus_interval_s: float = 30.0  # lấy 1 khung hình mẫu / camera mỗi chừng này giây để học không gian đặc trưng
    pca_corpus_max: int = 5000
    pca_q_review: float = Field(0.5, gt=0.0, lt=1.0)  # lọc trước khi duyệt: bỏ ảnh gần ảnh đã có hơn mức "khác nhau thường gặp" này
    pca_q_train: float = Field(0.8, gt=0.0, lt=1.0)  # lọc trước khi train: mạnh tay hơn
    pca_min_radius_review: float = Field(0.01, ge=0.0, le=1.0)  # sàn bán kính: kho mẫu toàn khung giống hệt (phòng trống) vẫn lọc được
    pca_min_radius_train: float = Field(0.02, ge=0.0, le=1.0)
    pca_train_min_keep: float = Field(0.25, gt=0.0, le=1.0)  # luôn giữ ít nhất tỉ lệ này số ảnh mới (không cắt quá tay)
    pca_train_skip_below: int = 20  # ít hơn ngần này ảnh mới thì giữ hết, không lọc
    pca_triage_delay_s: float = 60.0  # ảnh ứng viên chờ chừng này giây để gom thành mẻ rồi mới lọc
    duplicate_ttl_days: int = Field(0, ge=0)  # 0 = ảnh bị lọc trùng XOÁ NGAY (workflow); >0 = giữ chừng này ngày, khôi phục được, rồi xoá
    jpeg_quality: int = 95  # ảnh gốc, không resize
    retrain_enabled: bool = True
    retrain_time: str = "00:00"
    nightly_catchup_h: float = 6.0  # lỡ giờ retrain (máy tắt) thì bù trong vòng chừng này giờ, quá thì đợi đêm sau
    retrain_min_new: int = 1  # bỏ rào cản số ảnh tối thiểu: có >=1 ảnh mới là train
    retrain_epochs: int = 30
    retrain_imgsz: int = 640
    retrain_batch: int = 8
    retrain_freeze: int = 10  # số lớp đầu đóng băng khi dữ liệu ít (giảm quên)
    retrain_freeze_until_n: int = 200  # >= ngần này ảnh người duyệt thì mở khóa toàn bộ
    retrain_lr0: float = 0.0005
    retrain_timeout_h: float = 8.0
    replay_dir: Path | None = None  # dataset YOLO gốc (images/, labels/) làm replay buffer
    replay_old_ratio: float = 0.7  # 70 % cũ / 30 % mới
    val_fraction: float = 0.2
    min_holdout: int = 5  # ít hơn thế thì không đủ để làm tập giữ lại
    # always (workflow): train xong là hoán đổi ngay, cổng chất lượng chỉ để tham khảo + cảnh báo (rollback bằng tay).
    # gated: chỉ hoán đổi khi đạt cổng. never: luôn chờ admin bấm Promote.
    auto_promote: Literal["gated", "never", "always"] = "always"
    gate_tolerance: float = 0.01
    rejected_ttl_days: int = 7
    pending_ttl_days: int = 30

    # -------------------------------------------------------------- nhật ký DB
    telemetry_interval_s: float = 60.0
    habit_weeks: int = 8
    habit_slot_min: int = 15
    habit_min_people: float = 2.0

    # ------------------------------------------------------------- mô phỏng
    sim_speed: float = 1.0
    sim_start: str = ""  # ISO, ví dụ 2026-11-02T06:30:00 ; rỗng = bây giờ
    sim_seed: int = 42
    sim_cameras: int = Field(1, ge=1, le=36)  # số camera giả (camera 1 là camera điều khiển)

    # ================================================================ kiểm tra
    @field_validator("storage_root", "zones_file", "schedule_file", "replay_dir", "cameras_file", mode="before")
    @classmethod
    def _expand(cls, v):
        if v in (None, ""):
            return None
        p = Path(str(v)).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p)

    @model_validator(mode="after")
    def _check(self):
        if not self.al_conf_low < self.al_conf_high:
            raise ValueError("al_conf_low phải nhỏ hơn al_conf_high")
        if self.detect_conf > self.al_conf_low:
            raise ValueError(
                "detect_conf phải <= al_conf_low, nếu không YOLO không bao giờ trả ra detection trong dải học chủ động"
            )
        if not 0 <= self.idle_pwm_min <= self.idle_pwm_max <= 100:
            raise ValueError("cần 0 <= idle_pwm_min <= idle_pwm_max <= 100")
        if not 0 <= self.occ_off_threshold <= self.occ_on_threshold:
            raise ValueError("cần 0 <= occ_off_threshold <= occ_on_threshold (ngưỡng tắt phải thấp hơn ngưỡng bật)")
        if not self.pwm_occupied_min <= self.pwm_occupied_max:
            raise ValueError("pwm_occupied_min phải <= pwm_occupied_max")
        if not self.fan_min <= self.fan_max:
            raise ValueError("fan_min phải <= fan_max")
        if not self.t_min < self.t_max:
            raise ValueError("t_min phải nhỏ hơn t_max")
        if self.precool_ac_temp > self.t_target:
            raise ValueError("precool_ac_temp không được lớn hơn t_target")
        if not 0 < self.replay_old_ratio < 1:
            raise ValueError("replay_old_ratio phải trong (0, 1)")
        h, _, m = self.retrain_time.partition(":")
        if not (h.isdigit() and m.isdigit() and 0 <= int(h) < 24 and 0 <= int(m) < 60):
            raise ValueError("retrain_time phải dạng HH:MM")
        if self.simulation_mode:
            # Dữ liệu giả không được lẫn vào DB/ảnh/model thật: mặc định tách hẳn thư mục và không train YOLO trên ảnh tổng hợp.
            if "storage_root" not in self.model_fields_set:
                self.storage_root = REPO_ROOT / "var" / "ares_sim"
            if "retrain_enabled" not in self.model_fields_set:
                self.retrain_enabled = False
        return self

    # ============================================================== đường dẫn
    @property
    def tz(self):
        return get_tz(self.timezone)

    @property
    def database_url(self) -> str:
        return self.db_url or f"sqlite:///{(self.storage_root / 'ares.db').as_posix()}"

    @property
    def pending_dir(self) -> Path:
        return self.storage_root / "storage" / "pending"

    @property
    def approved_dir(self) -> Path:
        return self.storage_root / "storage" / "approved"

    @property
    def rejected_dir(self) -> Path:
        return self.storage_root / "storage" / "rejected"

    @property
    def datasets_dir(self) -> Path:
        return self.storage_root / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.storage_root / "models"

    @property
    def pca_dir(self) -> Path:
        return self.storage_root / "pca"

    @property
    def runs_dir(self) -> Path:
        return self.storage_root / "runs"

    @property
    def tuning_file(self) -> Path:
        """Giá trị delay / vùng điều chỉnh sửa từ dashboard (đè lên ares.env). Mô phỏng có thư mục riêng nên không lẫn."""
        return self.storage_root / "device_tuning.json"

    @property
    def retrain_hm(self) -> tuple[int, int]:
        h, _, m = self.retrain_time.partition(":")
        return int(h), int(m)

    def ensure_dirs(self) -> None:
        for p in (
            self.storage_root,
            self.pending_dir,
            self.approved_dir,
            self.rejected_dir,
            self.datasets_dir,
            self.models_dir,
            self.runs_dir,
            self.pca_dir,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)

    def weights_path(self) -> Path:
        p = Path(self.weights).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p)


# ---------------------------------------------------------------------------
# Cấu trúc vùng (zones.json)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ZoneSpec:
    id: int
    name: str
    polygon: tuple[tuple[float, float], ...]  # toạ độ chuẩn hoá 0..1 theo khung hình
    capacity: int
    k: float  # hệ số hiệu chuẩn lux trần -> lux mặt bàn (spec: ~3.0-3.5)


@dataclass(frozen=True)
class RoomSpec:
    room_id: str
    n_max: int
    zones: tuple[ZoneSpec, ...]

    @property
    def n_zones(self) -> int:
        return len(self.zones)


def default_room(room_id: str = "P101", n_max: int = 40, k: float = 3.25) -> RoomSpec:
    """Lưới 2x2 phủ toàn khung hình: Zone 1 trên-trái, 2 trên-phải, 3 dưới-trái, 4 dưới-phải."""
    cap = max(1, n_max // 4)
    cells = [
        ("Khu 1 (trên-trái)", (0.0, 0.0, 0.5, 0.5)),
        ("Khu 2 (trên-phải)", (0.5, 0.0, 1.0, 0.5)),
        ("Khu 3 (dưới-trái)", (0.0, 0.5, 0.5, 1.0)),
        ("Khu 4 (dưới-phải)", (0.5, 0.5, 1.0, 1.0)),
    ]
    zones = tuple(
        ZoneSpec(i + 1, name, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)), cap, k)
        for i, (name, (x0, y0, x1, y1)) in enumerate(cells)
    )
    return RoomSpec(room_id, n_max, zones)


def load_room(path: Path | str | None, fallback_room_id: str = "P101") -> RoomSpec:
    """Đọc zones.json. Thiếu file thì dùng lưới 2x2 mặc định."""
    if not path or not Path(path).exists():
        return default_room(fallback_room_id)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    zones = []
    for z in raw.get("zones", []):
        poly = tuple((float(x), float(y)) for x, y in z["polygon"])
        if len(poly) < 3:
            raise ValueError(f"Vùng {z.get('id')}: đa giác cần >= 3 điểm")
        for x, y in poly:
            if not (-0.05 <= x <= 1.05 and -0.05 <= y <= 1.05):
                raise ValueError(f"Vùng {z.get('id')}: toạ độ phải chuẩn hoá 0..1, gặp ({x}, {y})")
        k = float(z.get("k", 3.25))
        if not 0.5 <= k <= 10:
            raise ValueError(f"Vùng {z.get('id')}: k={k} bất thường (kỳ vọng ~3.0-3.5)")
        zones.append(ZoneSpec(int(z["id"]), str(z.get("name", f"Khu {z['id']}")), poly, int(z.get("capacity", 10)), k))
    if not zones:
        raise ValueError("zones.json không có vùng nào")
    ids = [z.id for z in zones]
    if len(set(ids)) != len(ids):
        raise ValueError("id vùng bị trùng")
    return RoomSpec(str(raw.get("room_id", fallback_room_id)), int(raw.get("n_max", 40)), tuple(zones))


@lru_cache
def get_settings() -> AresSettings:
    s = AresSettings()
    s.ensure_dirs()
    return s
