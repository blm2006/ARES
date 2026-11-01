"""Cấu hình toàn hệ thống. Mọi ngưỡng đọc từ .env — không hardcode ở nơi khác.

Ràng buộc spec liên quan:
  - Ngưỡng backpressure, quota sampling, theta dedup, gate validation: đều ở đây.
  - PCA version là một giá trị tường minh, không suy ra ngầm từ file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # ---------- chung ----------
    env: Literal["dev", "prod"] = "dev"
    data_root: Path = REPO_ROOT / "var"
    log_level: str = "INFO"

    # ---------- storage ----------
    storage_backend: Literal["local", "minio"] = "local"
    image_root: Path | None = None  # mặc định: data_root/images
    minio_endpoint: str = ""
    minio_bucket: str = "frames"

    # ---------- database ----------
    db_url: str = ""  # mặc định: sqlite ở data_root/meta.db

    # ---------- queue ----------
    queue_backend: Literal["redis", "sqlite"] = "redis"
    redis_url: str = "redis://localhost:6379/0"
    queue_stream: str = "frames"
    queue_group: str = "workers"
    queue_maxlen: int = 200_000

    # ---------- realtime ----------
    cameras: str = "cam01=0"  # "cam01=rtsp://...,cam02=0" ; giá trị số = index USB
    yolo_weights: str = "yolov8n.pt"
    yolo_base_checkpoint: str = "yolov8n.pt"  # LUÔN fine-tune từ đây (spec 3.2)
    device: str = "cuda:0"
    half: bool = True
    imgsz: int = 640
    conf_threshold: float = 0.3  # ngưỡng thô cho inference, KHÔNG dùng để sampling
    iou_threshold: float = 0.45
    person_class_id: int = 0
    detect_class_ids: str = "0"  # COCO id cần detect, phẩy ngăn cách. "0"=người; "0,1,2,3,5,7"=người+xe
    class_names: str = "0:person"  # map id->tên cho nhãn review + data.yaml, vd "0:person,2:car,3:motorcycle"
    ring_buffer_size: int = 8
    target_fps: float = 25.0
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000

    # ---------- tracker ----------
    tracker_type: Literal["bytetrack", "botsort"] = "bytetrack"
    track_history_len: int = 90  # số frame giữ lại để failure mining nhìn ngược

    # ---------- sampling router ----------
    daily_sample_budget: int = 1200  # tổng số frame được nhận / ngày / toàn hệ thống
    quota_uncertainty: float = 0.40
    quota_failure: float = 0.40
    quota_random: float = 0.20
    uncertainty_top_percentile: float = 5.0  # top 5% bất định nhất
    uncertainty_window_hours: int = 24
    uncertainty_min_samples: int = 500  # dưới mức này chưa đủ để tính phân vị
    random_sample_rate: float = 0.015  # 1.5% frame
    sample_min_gap_s: float = 0.0  # giãn cách tối thiểu (giây) giữa 2 frame ĐƯA DUYỆT của 1 camera;
    #                               0 = tắt. Nguồn failure-mining (frame thay đổi) KHÔNG bị giãn.
    short_track_frames: int = 5
    per_hour_cap_ratio: float = 0.12  # 1 giờ không chiếm quá 12% budget ngày
    per_camera_cap_ratio: float = 0.40  # 1 camera không chiếm quá 40% budget ngày

    # ---------- backpressure ----------
    queue_depth_n1: int = 2_000  # > N1: chỉ nhận source B và C
    queue_depth_n2: int = 10_000  # > N2: dừng nhận
    disk_free_min_pct: float = 10.0

    # ---------- frame writer ----------
    jpeg_quality: int = 95
    image_format: Literal["jpg", "webp"] = "jpg"

    # ---------- worker / dedup ----------
    pca_version: str = "pca_v1"
    pca_components: int = 128
    pca_fit_min_samples: int = 2_000
    dedup_theta: float = 0.94  # cosine > theta -> discard
    dedup_theta_failure: float = 0.985  # nguồn B lỏng hơn (spec 2.3)
    faiss_index_path: Path | None = None
    coreset_enabled: bool = True
    coreset_keep_ratio: float = 0.7
    worker_batch_size: int = 32

    # ---------- review ----------
    review_host: str = "127.0.0.1"
    review_port: int = 8010
    agreement_target: float = 0.97

    # ---------- dataset ----------
    golden_frozen: bool = True
    rolling_days: int = 30
    split_ratios: str = "0.8,0.1,0.1"  # train,val,test — chia theo session

    # ---------- training ----------
    train_trigger_min_new: int = 500
    train_trigger_map_drop_pct: float = 2.0
    train_trigger_agreement_drop_pct: float = 3.0
    train_epochs: int = 40
    train_batch: int = 16
    train_old_data_ratio: float = 0.70  # mix 70% cũ / 30% mới
    val_per_camera_max_drop_pct: float = 2.0
    val_primary_metric: Literal["recall50", "map50"] = "recall50"

    # ---------- deploy ----------
    shadow_min_hours: int = 24
    canary_camera: str = ""  # rỗng = lấy camera đầu tiên
    canary_min_hours: int = 24
    rollback_count_drift_pct: float = 15.0
    rollback_id_switch_increase_pct: float = 25.0
    rollback_fps_drop_pct: float = 15.0

    # ---------- governance ----------
    image_ttl_days: int = 90
    audit_log_enabled: bool = True

    # ---------- derived ----------
    @field_validator("data_root", "image_root", "faiss_index_path", mode="before")
    @classmethod
    def _expand(cls, v):
        return Path(v).expanduser() if v else v

    @property
    def images_dir(self) -> Path:
        return self.image_root or (self.data_root / "images")

    @property
    def datasets_dir(self) -> Path:
        return self.data_root / "datasets"

    @property
    def registry_dir(self) -> Path:
        return self.data_root / "registry"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_root / "artifacts"

    @property
    def faiss_path(self) -> Path:
        return self.faiss_index_path or (self.artifacts_dir / f"faiss_{self.pca_version}.index")

    @property
    def pca_path(self) -> Path:
        return self.artifacts_dir / f"{self.pca_version}.pkl"

    @property
    def database_url(self) -> str:
        if self.db_url:
            return self.db_url
        return f"sqlite:///{(self.data_root / 'meta.db').as_posix()}"

    @property
    def split_tuple(self) -> tuple[float, float, float]:
        a, b, c = (float(x) for x in self.split_ratios.split(","))
        return a, b, c

    def camera_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in self.cameras.split(","):
            item = item.strip()
            if not item:
                continue
            cam_id, _, src = item.partition("=")
            out[cam_id.strip()] = src.strip()
        return out

    def class_id_list(self) -> list[int]:
        """Danh sách COCO id sẽ được detect (parse từ DETECT_CLASS_IDS)."""
        return [int(x.strip()) for x in self.detect_class_ids.split(",") if x.strip()]

    def class_name_map(self) -> dict[int, str]:
        """Map {id: tên} cho nhãn review và data.yaml.

        Gộp CLASS_NAMES (.env) với nhãn người dùng tự thêm ở review UI
        (var/labels.json) — file này thắng khi trùng id.
        """
        out: dict[int, str] = {}
        for pair in self.class_names.split(","):
            pair = pair.strip()
            if not pair:
                continue
            i, _, n = pair.partition(":")
            out[int(i.strip())] = n.strip()
        f = self.data_root / "labels.json"
        if f.exists():
            import json
            try:
                for k, v in json.loads(f.read_text(encoding="utf-8")).items():
                    out[int(k)] = str(v)
            except Exception:
                pass
        return out

    def quota_for(self, source: str) -> int:
        ratio = {
            "uncertainty": self.quota_uncertainty,
            "failure_mining": self.quota_failure,
            "random": self.quota_random,
        }[source]
        return int(self.daily_sample_budget * ratio)

    def ensure_dirs(self) -> None:
        for p in (
            self.data_root,
            self.images_dir,
            self.datasets_dir,
            self.registry_dir,
            self.artifacts_dir,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
