"""Chỉnh delay / vùng điều chỉnh của thiết bị NGAY KHI ĐANG CHẠY (tab "Thiết bị" trên dashboard).

* Danh sách tham số được phép chỉnh nằm ở `TUNABLES` — chỉ các tham số chống bật/tắt liên tục, dải điều chỉnh và FPS.
  Những thứ ảnh hưởng an toàn/dữ liệu (đường dẫn, token, MQTT, model) KHÔNG chỉnh được ở đây.
* Mỗi giá trị được kiểm tra bằng đúng ràng buộc của AresSettings (Field ge/le) + các ràng buộc chéo
  (ngưỡng tắt < ngưỡng bật, min <= max...). Sai thì từ chối cả lô, không áp dụng nửa vời.
* Áp dụng bằng cách gán thẳng vào đối tượng cấu hình đang chạy: bộ điều khiển đọc cfg ở mỗi nhịp nên có hiệu lực
  ngay nhịp sau, không cần khởi động lại.
* Lưu vào <storage_root>/device_tuning.json (ghi nguyên tử) và ĐÈ lên ares.env ở lần khởi động sau.
  "Khôi phục mặc định" = xoá các giá trị đè, trở về giá trị lúc khởi động (từ ares.env / mặc định).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Annotated, Any, Callable

from pydantic import TypeAdapter, ValidationError

from ares.config import AresSettings
from ares.state import Event

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tunable:
    key: str
    group: str
    label: str
    unit: str
    help: str
    step: float = 1.0


G_LIGHT_DELAY = "Đèn — độ trễ"
G_LIGHT_BAND = "Đèn — vùng điều chỉnh"
G_AC_DELAY = "Điều hòa — độ trễ"
G_AC_BAND = "Điều hòa — vùng điều chỉnh"
G_FPS = "Camera — FPS"

TUNABLES: tuple[Tunable, ...] = (
    Tunable("light_on_delay_s", G_LIGHT_DELAY, "Trễ bật", "giây", "Phải thấy người liên tục chừng này mới bật sáng vùng. Thực tế còn cộng thêm tối đa 1 nhịp điều khiển đèn."),
    Tunable("empty_dim_after_s", G_LIGHT_DELAY, "Trễ hạ về nền", "giây", "Vùng trống liên tục chừng này thì hạ về mức nền 10–15 %."),
    Tunable("room_off_after_s", G_LIGHT_DELAY, "Trễ tắt cả phòng", "giây", "Cả phòng trống liên tục chừng này thì tắt hẳn đèn."),
    Tunable("lighting_interval_s", G_LIGHT_DELAY, "Nhịp điều khiển đèn", "giây", "Bao lâu tính lại PWM một lần (spec 10–30 s)."),
    Tunable("occ_on_threshold", G_LIGHT_BAND, "Ngưỡng bật (có người)", "0–1", "Trung bình 30 s của số người ≥ ngưỡng này thì coi là có người.", 0.05),
    Tunable("occ_off_threshold", G_LIGHT_BAND, "Ngưỡng tắt (hết người)", "0–1", "Đã có người thì chỉ coi là hết khi < ngưỡng này. Khoảng giữa 2 ngưỡng là vùng trễ.", 0.05),
    Tunable("lux_deadband", G_LIGHT_BAND, "Vùng chết độ rọi", "± lux", "Lux đo lệch lux đích không quá chừng này thì giữ nguyên PWM.", 5),
    Tunable("pwm_max_step", G_LIGHT_BAND, "Bước PWM tối đa", "%/nhịp", "Mỗi nhịp PWM đổi tối đa chừng này (trừ lúc vừa có người).", 1),
    Tunable("pwm_occupied_min", G_LIGHT_BAND, "PWM thấp nhất khi có người", "%", "Dải điều chỉnh khi vùng có người — cận dưới.", 1),
    Tunable("pwm_occupied_max", G_LIGHT_BAND, "PWM cao nhất", "%", "Dải điều chỉnh khi vùng có người — cận trên.", 1),
    Tunable("idle_pwm_min", G_LIGHT_BAND, "PWM nền thấp nhất (vùng trống)", "%", "Vùng trống nhưng phòng còn người.", 1),
    Tunable("idle_pwm_max", G_LIGHT_BAND, "PWM nền cao nhất (vùng trống)", "%", "", 1),
    Tunable("led_publish_min_delta", G_LIGHT_BAND, "Ngưỡng gửi lệnh đèn", "%", "PWM đổi ít hơn chừng này thì không gửi lệnh mới tới ESP32.", 0.5),
    Tunable("ac_on_delay_s", G_AC_DELAY, "Trễ bật theo hiện diện", "giây", "'Có người + nóng' phải giữ liên tục chừng này mới bật (ngoài pre-cool theo lịch)."),
    Tunable("ac_off_delay_s", G_AC_DELAY, "Trễ tắt khi phòng trống", "giây", "Cộng thêm sau cửa sổ trung bình 10 phút."),
    Tunable("ac_min_on_s", G_AC_DELAY, "Thời gian chạy tối thiểu", "giây", "Đã bật thì chạy ít nhất chừng này mới được tắt vì phòng trống (fail-safe lớp trống vẫn tắt được)."),
    Tunable("ac_min_off_s", G_AC_DELAY, "Thời gian nghỉ tối thiểu", "giây", "Đã tắt thì nghỉ ít nhất chừng này mới bật lại — bảo vệ máy nén, áp dụng cả pre-cool."),
    Tunable("ac_keep_on_before_class_min", G_AC_DELAY, "Không tắt nếu sắp vào lớp", "phút", "Phòng trống nhưng lớp kế tiếp bắt đầu trong chừng này phút thì giữ điều hòa chạy (0 = tắt bình thường)."),
    Tunable("min_cmd_interval_s", G_AC_DELAY, "Giãn cách giữa 2 lệnh IR", "giây", "Áp dụng cho lệnh không khẩn cấp (đổi setpoint, quạt, tắt khi trống)."),
    Tunable("hvac_interval_min", G_AC_DELAY, "Nhịp đánh giá setpoint", "phút", "Bao lâu tính lại setpoint một lần trong giờ học (spec 10–15 phút)."),
    Tunable("deadband", G_AC_BAND, "Vùng chết nhiệt độ", "± °C", "Setpoint tính ra lệch setpoint đang đặt không quá chừng này thì không bắn IR.", 0.1),
    Tunable("t_min", G_AC_BAND, "Setpoint thấp nhất", "°C", "", 1),
    Tunable("t_max", G_AC_BAND, "Setpoint cao nhất", "°C", "", 1),
    Tunable("ac_setpoint_max_step", G_AC_BAND, "Bước setpoint tối đa", "°C/lệnh", "Mỗi lệnh đổi setpoint tối đa chừng này.", 1),
    Tunable("turn_on_margin", G_AC_BAND, "Biên bật theo nhiệt độ", "°C", "Bật khi nhiệt độ phòng > setpoint + biên này.", 0.1),
    Tunable("fan_min", G_AC_BAND, "Quạt thấp nhất", "mức 1–3", "", 1),
    Tunable("fan_max", G_AC_BAND, "Quạt cao nhất", "mức 1–3", "", 1),
    Tunable("fan_hysteresis", G_AC_BAND, "Vùng chết mức quạt", "tỉ lệ sức chứa", "0,05 = chỉ đổi mức quạt khi lấp đầy vượt mốc ±5 % sức chứa.", 0.01),
    Tunable("vision_fps", G_FPS, "FPS bình thường", "FPS", "Giới hạn trên — thực tế ≤ FPS camera và tốc độ YOLO.", 1),
    Tunable("vision_fps_training", G_FPS, "FPS khi đang train", "FPS", "Hạ để nhường GPU/CPU cho tiến trình train.", 0.5),
    Tunable("hub_fps", G_FPS, "Camera phụ: vòng/giây", "vòng/s", "Mỗi vòng đọc mọi camera phụ một lần.", 0.5),
    Tunable("hub_fps_training", G_FPS, "Camera phụ khi train", "vòng/s", "", 0.1),
)
KEYS = {t.key for t in TUNABLES}

# (cận dưới, cận trên, thông báo) — phải đúng cả sau khi áp dụng thay đổi
CROSS_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("occ_off_threshold", "occ_on_threshold", "Ngưỡng tắt phải ≤ ngưỡng bật"),
    ("pwm_occupied_min", "pwm_occupied_max", "PWM thấp nhất phải ≤ PWM cao nhất"),
    ("idle_pwm_min", "idle_pwm_max", "PWM nền thấp nhất phải ≤ PWM nền cao nhất"),
    ("t_min", "t_max", "Setpoint thấp nhất phải < cao nhất"),
    ("fan_min", "fan_max", "Quạt thấp nhất phải ≤ cao nhất"),
)


def _adapter(key: str) -> TypeAdapter:
    f = AresSettings.model_fields[key]
    return TypeAdapter(Annotated[(f.annotation, *f.metadata)]) if f.metadata else TypeAdapter(f.annotation)


# Ràng buộc bổ sung cho tham số cũ chưa có ge/le trong config (để dashboard không nhận số vô lý)
EXTRA_RANGE: dict[str, tuple[float, float]] = {
    "empty_dim_after_s": (0, 7200), "room_off_after_s": (0, 7200), "occ_on_threshold": (0.01, 1.0),
    "occ_off_threshold": (0.0, 1.0), "idle_pwm_min": (0, 100), "idle_pwm_max": (0, 100),
    "led_publish_min_delta": (0.1, 50), "min_cmd_interval_s": (0, 3600), "deadband": (0.0, 3.0),
    "t_min": (16, 30), "t_max": (16, 32), "turn_on_margin": (0.0, 5.0),
}


class Tuning:
    def __init__(self, cfg: AresSettings, emit: Callable[[Event], None] | None = None, clock=None):
        self.cfg = cfg
        self.emit = emit if emit is not None else (lambda e: None)
        self.clock = clock
        self._lock = threading.Lock()
        self.baseline: dict[str, Any] = {k: getattr(cfg, k) for k in KEYS}  # giá trị lúc khởi động (ares.env / mặc định)
        self.overrides: dict[str, Any] = {}

    # ---------------------------------------------------------------- kiểm tra
    def _validate(self, changes: dict[str, Any]) -> dict[str, Any]:
        errors: list[str] = []
        clean: dict[str, Any] = {}
        for k, v in changes.items():
            if k not in KEYS:
                errors.append(f"'{k}' không chỉnh được từ dashboard")
                continue
            try:
                val = _adapter(k).validate_python(v)
            except ValidationError as e:
                errors.append(f"{k}: {e.errors()[0].get('msg', 'giá trị không hợp lệ')}")
                continue
            lo_hi = EXTRA_RANGE.get(k)
            if lo_hi and not lo_hi[0] <= float(val) <= lo_hi[1]:
                errors.append(f"{k}: phải trong [{lo_hi[0]:g}, {lo_hi[1]:g}]")
                continue
            clean[k] = val
        merged = {k: clean.get(k, getattr(self.cfg, k)) for k in KEYS}
        for lo, hi, msg in CROSS_CHECKS:
            bad = merged[lo] >= merged[hi] if lo == "t_min" else merged[lo] > merged[hi]
            if bad:
                errors.append(msg)
        if errors:
            raise ValueError("; ".join(errors))
        return clean

    # ---------------------------------------------------------------- áp dụng
    def _apply(self, values: dict[str, Any]) -> None:
        for k, v in values.items():
            setattr(self.cfg, k, v)  # bộ điều khiển đọc cfg mỗi nhịp -> có hiệu lực ngay nhịp sau

    def update(self, changes: dict[str, Any]) -> dict:
        with self._lock:
            clean = self._validate(changes)
            changed = {k: v for k, v in clean.items() if getattr(self.cfg, k) != v}
            self._apply(clean)
            for k, v in clean.items():
                if v == self.baseline[k]:
                    self.overrides.pop(k, None)
                else:
                    self.overrides[k] = v
            self._save()
        if changed and self.clock is not None:
            txt = ", ".join(f"{k}={v}" for k, v in changed.items())
            self.emit(Event(self.clock.time(), "TUNING_CHANGED", "info", f"Đổi cài đặt thiết bị từ dashboard: {txt}.",
                            data={"changed": changed}))
        return self.describe()

    def reset(self) -> dict:
        with self._lock:
            self._apply(self.baseline)
            self.overrides = {}
            self._save()
        if self.clock is not None:
            self.emit(Event(self.clock.time(), "TUNING_RESET", "info", "Khôi phục cài đặt thiết bị về mặc định (ares.env)."))
        return self.describe()

    # ---------------------------------------------------------------- lưu / nạp
    def _save(self) -> None:
        f = self.cfg.tuning_file
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.overrides, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, f)

    def load(self) -> dict:
        """Nạp giá trị đã chỉnh từ lần chạy trước. File hỏng/giá trị sai thì bỏ qua (giữ ares.env), không làm sập khởi động."""
        f = self.cfg.tuning_file
        if not f.exists():
            return {}
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            raw = {k: v for k, v in raw.items() if k in KEYS}
            with self._lock:
                clean = self._validate(raw)
                self._apply(clean)
                self.overrides = {k: v for k, v in clean.items() if v != self.baseline[k]}
            log.info("Đã nạp %d cài đặt thiết bị từ %s", len(clean), f)
            return clean
        except Exception as e:  # noqa: BLE001
            log.warning("Bỏ qua %s (không hợp lệ: %s) — dùng giá trị ares.env", f, e)
            return {}

    # ---------------------------------------------------------------- cho dashboard
    def describe(self) -> dict:
        groups: dict[str, list[dict]] = {}
        for t in TUNABLES:
            f = AresSettings.model_fields[t.key]
            lo = hi = None
            for m in f.metadata:
                lo = getattr(m, "ge", lo) if getattr(m, "ge", None) is not None else lo
                hi = getattr(m, "le", hi) if getattr(m, "le", None) is not None else hi
            if t.key in EXTRA_RANGE:
                lo, hi = EXTRA_RANGE[t.key]
            groups.setdefault(t.group, []).append({
                "key": t.key, "label": t.label, "unit": t.unit, "help": t.help, "step": t.step,
                "value": getattr(self.cfg, t.key), "default": self.baseline[t.key],
                "overridden": t.key in self.overrides, "min": lo, "max": hi,
                "integer": f.annotation is int,
            })
        return {"groups": [{"name": g, "items": items} for g, items in groups.items()],
                "file": str(self.cfg.tuning_file), "n_overrides": len(self.overrides)}
