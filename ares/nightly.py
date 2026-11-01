"""Việc ban đêm (spec: "mỗi nửa đêm"): gom thói quen -> dọn dẹp -> retrain model.

Thứ tự có chủ ý: gom thói quen (nhanh, quan trọng cho pre-cooling hôm sau) chạy TRƯỚC, retrain (có thể
mất hàng giờ) chạy SAU cùng và nằm ở luồng riêng nên không làm chậm vision/điều khiển.

Chống chạy lặp / bỏ sót khi khởi động lại: dấu "đã chạy ngày X" được ghi vào bảng system_events
(kind=NIGHTLY_DONE) chứ không giữ trong RAM.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable

from ares.config import AresSettings
from ares.state import Event

log = logging.getLogger(__name__)


class NightlyJobs:
    def __init__(self, cfg: AresSettings, clock, db, al=None, retrainer=None,
                 emit: Callable[[Event], None] | None = None, pca=None):
        self.cfg, self.clock, self.db, self.al, self.retrainer = cfg, clock, db, al, retrainer
        self.pca = pca
        self.emit = emit if emit is not None else (lambda e: None)
        self.last_run_local: datetime | None = None  # naive, giờ địa phương
        self.last_summary: dict | None = None

    # ---------------------------------------------------------------- lịch
    def _target(self, now: datetime) -> datetime:
        h, m = self.cfg.retrain_hm
        return now.replace(hour=h, minute=m, second=0, microsecond=0)

    def _last_done(self) -> datetime | None:
        if self.last_run_local is not None:
            return self.last_run_local
        try:
            return self.db.last_event_time("NIGHTLY_DONE")
        except Exception:
            log.exception("Không đọc được mốc NIGHTLY_DONE")
            return None

    def due(self, now: datetime | None = None) -> bool:
        """Đã qua giờ hẹn hôm nay, chưa chạy lần nào kể từ giờ hẹn, và chưa quá hạn bù."""
        now = now or self.clock.now()
        target = self._target(now)
        if now < target:
            return False
        if now > target + timedelta(hours=self.cfg.nightly_catchup_h):
            return False  # lỡ quá lâu: đợi đêm sau, không train giữa giờ học
        last = self._last_done()
        return last is None or last < target.replace(tzinfo=None)

    # ---------------------------------------------------------------- chạy
    def run(self, now: datetime | None = None, with_retrain: bool = True) -> dict:
        now = now or self.clock.now()
        summary: dict = {"at": now.isoformat(timespec="seconds")}
        t = self.clock.time()

        def step(name: str, fn: Callable[[], object]) -> None:
            try:
                summary[name] = fn()
            except Exception as e:  # noqa: BLE001 - một bước hỏng không được chặn các bước sau
                log.exception("Việc ban đêm '%s' lỗi", name)
                summary[name] = {"error": str(e)}
                self.emit(Event(t, "NIGHTLY_STEP_FAILED", "error", f"Việc ban đêm '{name}' lỗi: {e}"))

        step("habits", lambda: self.db.aggregate_habits(now.replace(tzinfo=None)))
        step("telemetry_purge", lambda: self.db.purge_old_telemetry())
        if self.pca is not None and self.cfg.pca_enabled:
            step("pca_triage", lambda: self.pca.triage(force=True))  # không để ứng viên nào kẹt ở hàng chờ lọc
        if self.al is not None:
            step("al_purge", self.al.purge)
        if with_retrain and self.retrainer is not None and self.cfg.retrain_enabled:
            step("retrain", lambda: self.retrainer.run("midnight"))
        else:
            summary["retrain"] = {"status": "skipped", "reason": "tắt hoặc không có Retrainer"}

        self.last_run_local = self.clock.naive_now()
        self.last_summary = summary
        h = summary.get("habits", {})
        rt = summary.get("retrain", {})
        self.emit(Event(self.clock.time(), "NIGHTLY_DONE", "info",
                        f"Việc ban đêm xong: cập nhật {h.get('slots_updated', '?')} slot thói quen; "
                        f"retrain: {rt.get('status', '?')}.", data={"summary": summary}))
        return summary

    def tick(self) -> dict | None:
        """Gọi định kỳ (ví dụ mỗi 30 giây). Chạy nếu đến hạn."""
        if self.due():
            return self.run()
        return None
