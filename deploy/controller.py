"""Deploy controller: shadow -> canary (1 camera) -> full, có auto rollback.

Máy trạng thái, mỗi bước có điều kiện thời gian tối thiểu và điều kiện telemetry.
Nó KHÔNG tự nhảy bước khi chưa đủ dữ liệu — "chưa đủ telemetry" khác với "đạt".

Gọi `python -m deploy.controller status|promote|rollback` từ cron, hoặc chạy
`watch` để nó tự kiểm tra định kỳ.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import timedelta

from common.config import Settings, get_settings
from common.db import ModelVersion, init_db, session_scope, utcnow
from deploy.rollback import compare
from training.registry import Registry

log = logging.getLogger(__name__)

NEXT_STAGE = {"registered": "shadow", "shadow": "canary", "canary": "production"}


@dataclass
class PromotionCheck:
    ok: bool
    next_stage: str | None
    reason: str
    signals: dict


class DeployController:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.reg = Registry(self.s)

    # ------------------------------------------------------------------
    def canary_camera(self) -> str:
        return self.s.canary_camera or next(iter(self.s.camera_map()))

    def _hours_in_stage(self, m: ModelVersion) -> float:
        # created_at là mốc thô; đủ dùng vì mỗi lần đổi stage đều ghi note.
        return (utcnow().replace(tzinfo=None) - m.created_at).total_seconds() / 3600

    # ------------------------------------------------------------------
    def check_promotion(self, version: str) -> PromotionCheck:
        m = self.reg.get(version)
        stage = m.stage
        if stage not in NEXT_STAGE:
            return PromotionCheck(False, None, f"stage '{stage}' không promote được", {})

        nxt = NEXT_STAGE[stage]
        hours = self._hours_in_stage(m)

        if stage == "registered":
            passed = (m.metrics or {}).get("validation", {}).get("passed")
            if passed is None:
                return PromotionCheck(False, nxt, "chưa có kết quả validation — chạy validate trước", {})
            if not passed:
                gates = (m.metrics or {}).get("validation", {}).get("gates", {})
                failed = [k for k, v in gates.items() if not v.get("ok")]
                return PromotionCheck(False, None, f"validation trượt gate: {failed}", gates)
            return PromotionCheck(True, nxt, "validation đạt -> vào shadow", {})

        if stage == "shadow":
            if hours < self.s.shadow_min_hours:
                return PromotionCheck(False, nxt,
                                      f"mới shadow {hours:.1f}h / {self.s.shadow_min_hours}h", {})
            chk = compare(self.canary_camera(), role_new="shadow", settings=self.s)
            if chk.should_rollback:
                return PromotionCheck(False, None, f"shadow xấu: {chk.reason}", chk.signals)
            if not chk.signals:
                return PromotionCheck(False, nxt, chk.reason, {})
            return PromotionCheck(True, nxt, "shadow ổn định -> canary 1 camera", chk.signals)

        # canary
        if hours < self.s.canary_min_hours:
            return PromotionCheck(False, nxt, f"mới canary {hours:.1f}h / {self.s.canary_min_hours}h", {})
        chk = compare(self.canary_camera(), role_new="canary", settings=self.s)
        if chk.should_rollback:
            return PromotionCheck(False, None, f"canary xấu: {chk.reason}", chk.signals)
        if not chk.signals:
            return PromotionCheck(False, nxt, chk.reason, {})
        return PromotionCheck(True, nxt, "canary ổn định -> full rollout", chk.signals)

    # ------------------------------------------------------------------
    def promote(self, version: str, dry_run: bool = False) -> PromotionCheck:
        chk = self.check_promotion(version)
        if chk.ok and not dry_run:
            self.reg.set_stage(version, chk.next_stage)
            log.info("model %s -> %s (%s)", version, chk.next_stage, chk.reason)
        return chk

    def auto_rollback(self) -> dict | None:
        """Kiểm tra model đang chạy; xấu thì quay về bản production gần nhất còn tốt."""
        cur = self.reg.current("production")
        if cur is None:
            return None
        cam = self.canary_camera()
        chk = compare(cam, role_new="primary", settings=self.s)
        if not chk.should_rollback:
            return None

        history = [m for m in self.reg.history(20) if m.stage == "rolled_back" or m.version != cur.version]
        target = next((m for m in history if m.version != cur.version), None)
        if target is None:
            log.error("telemetry xấu (%s) nhưng không có bản nào để rollback về", chk.reason)
            return {"rolled_back": False, "reason": chk.reason, "signals": chk.signals}

        self.reg.rollback_to(target.version, chk.reason)
        log.warning("AUTO ROLLBACK %s -> %s: %s", cur.version, target.version, chk.reason)
        return {
            "rolled_back": True,
            "from": cur.version,
            "to": target.version,
            "reason": chk.reason,
            "signals": chk.signals,
        }

    def status(self) -> dict:
        out = {"production": None, "pipeline": []}
        prod = self.reg.current("production")
        if prod:
            out["production"] = {"version": prod.version, "dataset": prod.dataset_version}
        for m in self.reg.history(10):
            out["pipeline"].append(
                {
                    "version": m.version,
                    "stage": m.stage,
                    "dataset": m.dataset_version,
                    "hours_in_stage": round(self._hours_in_stage(m), 1),
                    "primary_metric": (m.metrics or {}).get("validation", {}).get("rolling", {}),
                }
            )
        return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="deploy")
    ap.add_argument("cmd", choices=["status", "promote", "rollback", "watch"])
    ap.add_argument("--version")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    args = ap.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()
    c = DeployController()

    if args.cmd == "status":
        print(json.dumps(c.status(), indent=2, ensure_ascii=False))
    elif args.cmd == "promote":
        if not args.version:
            raise SystemExit("--version bắt buộc")
        chk = c.promote(args.version, args.dry_run)
        print(json.dumps({"ok": chk.ok, "next": chk.next_stage, "reason": chk.reason,
                          "signals": chk.signals}, indent=2, ensure_ascii=False))
    elif args.cmd == "rollback":
        print(json.dumps(c.auto_rollback() or {"rolled_back": False, "reason": "telemetry ổn"},
                         indent=2, ensure_ascii=False))
    else:
        while True:
            r = c.auto_rollback()
            if r:
                print(json.dumps(r, ensure_ascii=False))
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
