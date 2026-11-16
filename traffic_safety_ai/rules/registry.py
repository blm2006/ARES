"""Danh sách rule, kiểm tra năng lực (model + cấu hình địa điểm), bảng trạng thái bật/tắt có lý do."""

from __future__ import annotations

from dataclasses import dataclass

from traffic_safety_ai.core.config import SiteConfig
from traffic_safety_ai.rules.base_rule import BaseRule
from traffic_safety_ai.rules.catalog import LegalCatalog
from traffic_safety_ai.rules.cctv_rules import ExcessPassengersRule, NoHelmetRule, OverloadRule, SpeedingRule
from traffic_safety_ai.rules.extra_rules import AbreastRule, JaywalkRule, ParkingRule
from traffic_safety_ai.rules.pov_rules import BlockedSightlineRule, SuddenEmergenceRule, TurnNoSignalRule
from traffic_safety_ai.rules.shared_rules import (FollowingRule, LaneSidewalkRule, PedestrianConflictRule,
                                                  PhoneUseRule, RedLightRule, WrongWayRule)

ALL_RULES: list[type[BaseRule]] = [
    NoHelmetRule, PhoneUseRule, WrongWayRule, LaneSidewalkRule, RedLightRule, ExcessPassengersRule,
    PedestrianConflictRule, FollowingRule, OverloadRule, SpeedingRule, ParkingRule, AbreastRule,
    TurnNoSignalRule, BlockedSightlineRule, SuddenEmergenceRule, JaywalkRule,
]


@dataclass
class RuleStatus:
    code: str
    name: str
    enabled: bool
    reason: str
    channels: tuple[str, ...]


def scene_capabilities(scene: SiteConfig) -> set[str]:
    caps: set[str] = set()
    if scene.calibration.enabled:
        caps.add("calibration")
    if scene.lines.speed_traps:
        caps.add("speed_trap")
    if scene.camera.channel == "cctv":
        caps.add("bgsub")          # camera tĩnh → tách nền được (RB-09 dự phòng)
    return caps


def build_rules(scene: SiteConfig, catalog: LegalCatalog, capabilities: set[str],
                only: set[str] | None = None) -> tuple[list[BaseRule], list[RuleStatus]]:
    channel = scene.camera.channel
    caps = set(capabilities) | scene_capabilities(scene)
    rules: list[BaseRule] = []
    report: list[RuleStatus] = []
    for cls in ALL_RULES:
        r = cls(scene, catalog, scene.rule_params(cls.code))
        if only is not None and cls.code not in only:
            report.append(RuleStatus(cls.code, r.name, False, "không chọn trong lần chạy này", cls.channels))
            continue
        ok, why = r.check_available(caps, channel)
        reason = why or "sẵn sàng"
        if r.unknown_params:
            reason += f" (bỏ qua tham số không hợp lệ: {', '.join(r.unknown_params)})"
        report.append(RuleStatus(cls.code, r.name, ok, reason, cls.channels))
        if ok:
            rules.append(r)
    return rules, report


def needed_models(scene: SiteConfig, lpr: bool = True) -> set[str]:
    """Model cần nạp theo các rule có thể bật (để không tốn VRAM cho model không dùng)."""
    ch = scene.camera.channel
    need = {"base"}
    if scene.rule_enabled("RB-01") and ch == "cctv":
        need.add("helmet")
    if scene.rule_enabled("RB-02"):
        need |= {"pose", "phone"}
    if scene.rule_enabled("RB-09") and ch == "cctv":
        need.add("cargo")
    if ch == "pov":
        if scene.rule_enabled("RB-04"):
            need.add("lane_seg")
        if scene.rule_enabled("RB-03"):
            need.add("orientation")
    if lpr:
        need |= {"plate", "plate_ocr"}
    return need


def format_status_table(report: list[RuleStatus]) -> str:
    lines = [f"{'Mã':6} {'Trạng thái':10} {'Tên':42} Lý do", "-" * 100]
    for s in report:
        lines.append(f"{s.code:6} {'BẬT' if s.enabled else 'tắt':10} {s.name[:42]:42} {s.reason}")
    return "\n".join(lines)
