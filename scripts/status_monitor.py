#!/usr/bin/env python3
"""Bảng theo dõi toàn hệ thống — đọc TRẠNG THÁI THẬT từ DB + queue + artifacts.

Khác bản cũ: bản cũ đếm file trong `var/images/cam01/*.jpg` và `var/batches/`.
Cả hai đều luôn bằng 0 — ảnh nằm ở `cam01/YYYY-MM-DD/HH/*.jpg`, còn `var/batches/`
là tàn dư của một thiết kế đã bỏ. Nên bảng cũ lúc nào cũng báo 0 dù hệ thống chạy tốt.

    python -m scripts.status_monitor            # làm mới mỗi 5 giây
    python -m scripts.status_monitor --once     # in một lần rồi thoát
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from common.config import get_settings
from common.db import (
    DatasetVersion,
    Frame,
    ModelVersion,
    ReviewItem,
    SampleBudget,
    session_scope,
)

# Console Windows cũ (cp1252/cp437) không encode được ký tự vẽ khung và emoji.
# Ghi ra màn hình thì Python tự xử lý, nhưng `> out.txt` sẽ ném UnicodeEncodeError.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BAR_W = 34


def bar(cur: int, cap: int, width: int = BAR_W) -> str:
    if cap <= 0:
        return "─" * width
    filled = min(width, int(width * cur / cap))
    return "█" * filled + "░" * (width - filled)


def pct(cur: int, cap: int) -> float:
    return (cur / cap * 100.0) if cap > 0 else 0.0


class StatusMonitor:
    def __init__(self) -> None:
        self.s = get_settings()
        self._prev_frames: int | None = None
        self._prev_t: float | None = None

    # ---------------- thu thập ----------------
    def collect(self) -> dict:
        s = self.s
        # Frame.ts và SampleBudget.day đều tính theo UTC (collector/writer.py dùng
        # ts UTC rồi bỏ tzinfo). Dùng giờ máy ở đây là tra nhầm ngày: máy UTC+7 lúc
        # 5 giờ sáng thì UTC vẫn là hôm trước -> bảng báo 0/5000 trong khi quota đã cạn.
        now_utc = datetime.now(timezone.utc)
        today = now_utc.strftime("%Y-%m-%d")
        utc_midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        d: dict = {"today": today}
        # mốc reset hạn mức, quy về giờ máy để người đọc khỏi tự cộng trừ
        d["reset_local"] = (utc_midnight + timedelta(days=1)).astimezone().strftime("%H:%M %d/%m")

        with session_scope() as sess:
            d["frames_total"] = sess.execute(
                select(func.count()).select_from(Frame)
            ).scalar_one()
            d["frames_today"] = sess.execute(
                select(func.count()).select_from(Frame)
                .where(Frame.ts >= utc_midnight.replace(tzinfo=None))
            ).scalar_one()
            d["last_frame_ts"] = sess.execute(select(func.max(Frame.ts))).scalar_one()
            d["embedded"] = sess.execute(
                select(func.count()).select_from(Frame).where(Frame.embedded.is_(True))
            ).scalar_one()
            d["discarded"] = sess.execute(
                select(func.count()).select_from(Frame).where(Frame.dedup_discarded.is_(True))
            ).scalar_one()

            d["quota"] = {}
            for row in sess.execute(
                select(SampleBudget).where(
                    SampleBudget.day == today,
                    SampleBudget.cam_id == "*",
                    SampleBudget.hour == -1,
                )
            ).scalars():
                d["quota"][row.source] = row.used

            d["review"] = dict(
                sess.execute(
                    select(ReviewItem.status, func.count()).group_by(ReviewItem.status)
                ).all()
            )
            d["review_pending_by_source"] = dict(
                sess.execute(
                    select(ReviewItem.source, func.count())
                    .where(ReviewItem.status == "pending")
                    .group_by(ReviewItem.source)
                ).all()
            )
            d["datasets"] = [
                (v.version, v.n_train, v.n_val, v.n_test)
                for v in sess.execute(
                    select(DatasetVersion).order_by(DatasetVersion.created_at.desc()).limit(3)
                ).scalars()
            ]
            d["models"] = [
                (m.version, m.stage)
                for m in sess.execute(
                    select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(3)
                ).scalars()
            ]

        # queue: mở đọc-only, không tạo consumer group của Redis
        d["queue_depth"] = self._queue_depth()

        d["pca_ok"] = s.pca_path.exists()
        d["pca_info"] = ""
        if d["pca_ok"]:
            import json

            meta = s.pca_path.with_suffix(".json")
            if meta.exists():
                m = json.loads(meta.read_text(encoding="utf-8"))
                d["pca_info"] = (
                    f"{m['version']} · embedder={m['embedder']} · "
                    f"{m['input_dim']}→{m['n_components']} chiều · {m['n_fit_samples']} mẫu"
                )
            else:
                d["pca_info"] = s.pca_version

        usage = shutil.disk_usage(s.data_root)
        d["disk_free_pct"] = usage.free / usage.total * 100.0
        d["images_bytes"] = self._dir_size(s.images_dir)

        # tốc độ thu
        now = time.monotonic()
        rate = 0.0
        if self._prev_frames is not None and self._prev_t is not None:
            dt = now - self._prev_t
            if dt > 0:
                rate = (d["frames_total"] - self._prev_frames) / dt * 60.0
        self._prev_frames, self._prev_t = d["frames_total"], now
        d["rate_per_min"] = rate
        return d

    def _queue_depth(self) -> int | str:
        s = self.s
        if s.queue_backend == "sqlite":
            import sqlite3

            p = s.data_root / "queue.db"
            if not p.exists():
                return 0
            try:
                con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5)
                try:
                    return int(
                        con.execute(
                            "SELECT COUNT(*) FROM q WHERE state IN ('ready','inflight')"
                        ).fetchone()[0]
                    )
                finally:
                    con.close()
            except Exception as e:  # noqa: BLE001
                return f"lỗi đọc ({type(e).__name__})"
        try:
            from common.queue import get_queue

            return get_queue(s).depth()
        except Exception as e:  # noqa: BLE001
            return f"không kết nối được ({type(e).__name__})"

    @staticmethod
    def _dir_size(path) -> int:
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    # ---------------- hiển thị ----------------
    def render(self, d: dict) -> str:
        s = self.s
        L: list[str] = []
        W = 74
        L.append("═" * W)
        L.append(f"  YOLO-AL — trạng thái hệ thống   {datetime.now():%Y-%m-%d %H:%M:%S}")
        L.append("═" * W)

        # ---- Phase 2 ----
        age = None
        if d.get("last_frame_ts") is not None:
            age = (datetime.now(timezone.utc).replace(tzinfo=None) - d["last_frame_ts"]).total_seconds()
        if age is None:
            status, ago = "⚪ chưa có frame nào", "—"
        elif age < 120:
            status, ago = "🟢 đang thu", f"{age:.0f} giây trước"
        elif age < 3600:
            status, ago = "🟡 thưa (xem hạn mức bên dưới)", f"{age / 60:.0f} phút trước"
        else:
            status, ago = "⚪ đã dừng", f"{age / 3600:.1f} giờ trước"
        L.append("")
        L.append(f"[2] THU MẪU          {status}")
        L.append(f"    Tổng frame đã ghi : {d['frames_total']:,}   (hôm nay {d['frames_today']:,})")
        L.append(f"    Frame gần nhất    : {ago}")
        L.append(f"    Tốc độ            : {d['rate_per_min']:.0f} frame/phút")
        L.append(f"    Dung lượng ảnh    : {d['images_bytes'] / 1e9:.2f} GB   "
                 f"(đĩa trống {d['disk_free_pct']:.1f}%)")
        L.append("")
        L.append(f"    Hạn mức ngày {d['today']} (theo UTC — reset lúc {d['reset_local']} giờ máy).")
        L.append("    Hết hạn mức thì Phase 2 NGỪNG nhận. Đó là thiết kế, không phải lỗi:")
        total_used = total_cap = 0
        for src in ("uncertainty", "failure_mining", "random"):
            used = d["quota"].get(src, 0)
            cap = s.quota_for(src)
            total_used += used
            total_cap += cap
            flag = "HẾT" if used >= cap else "   "
            L.append(f"      {src:<15} [{bar(used, cap, 26)}] {used:>5,}/{cap:<5,} "
                     f"{pct(used, cap):5.1f}% {flag}")
        L.append(f"      {'TỔNG':<15} [{bar(total_used, total_cap, 26)}] "
                 f"{total_used:>5,}/{total_cap:<5,} {pct(total_used, total_cap):5.1f}%")

        # ---- PCA ----
        L.append("")
        L.append(f"[PCA] {'✅ đã freeze' if d['pca_ok'] else '❌ CHƯA CÓ'}")
        if d["pca_ok"]:
            L.append(f"      {d['pca_info']}")
        else:
            L.append(f"      Chạy: python -m scripts.fit_pca --images-from-storage --limit 5000")

        # ---- Phase 3 ----
        qd = d["queue_depth"]
        L.append("")
        pending_embed = qd if isinstance(qd, int) else "?"
        note = " ← Phase 3 chưa xử lý" if isinstance(qd, int) and qd > 0 else " (đã xử lý hết)"
        L.append(f"[3] WORKER (embed → dedup → review)")
        L.append(f"    Hàng đợi còn     : {pending_embed}{note}")
        L.append(f"    Đã embed         : {d['embedded']:,}")
        L.append(f"    Bị loại (trùng)  : {d['discarded']:,}")
        if isinstance(qd, int) and qd > 0 and d["pca_ok"]:
            L.append("    → Chạy: python scripts/startup.py --phase 3")

        # ---- Phase 4 ----
        rv = d["review"]
        L.append("")
        L.append(f"[4] REVIEW (cửa duy nhất vào dataset)")
        L.append(f"    Chờ duyệt        : {rv.get('pending', 0):,}")
        L.append(f"    Đã duyệt         : {rv.get('done', 0):,}")
        if d["review_pending_by_source"]:
            detail = "  ".join(f"{k}={v}" for k, v in sorted(d["review_pending_by_source"].items()))
            L.append(f"    Theo nguồn       : {detail}")
        if rv.get("pending", 0) > 0:
            L.append("    → Chạy: python scripts/startup.py --phase 4   (http://127.0.0.1:8010)")

        # ---- Phase 5-7 ----
        L.append("")
        L.append(f"[5-7] DATASET / TRAIN / DEPLOY")
        if d["datasets"]:
            for v, tr, va, te in d["datasets"]:
                L.append(f"    dataset {v}: train={tr} val={va} test={te}")
        else:
            L.append(f"    Chưa có dataset version nào "
                     f"(cần ≥1 frame đã duyệt rồi chạy --phase 5)")
        if d["models"]:
            for v, stage in d["models"]:
                L.append(f"    model {v}: {stage}")

        L.append("")
        L.append("─" * W)
        L.append(f"  {self.next_action(d)}")
        L.append("═" * W)
        return "\n".join(L)

    def next_action(self, d: dict) -> str:
        qd = d["queue_depth"]
        cạn = [src for src in ("uncertainty", "failure_mining", "random")
               if d["quota"].get(src, 0) >= self.s.quota_for(src)]
        if len(cạn) == 3 and not d["review"].get("pending"):
            return f"Hết hạn mức ngày, Phase 2 nghỉ tới {d['reset_local']} giờ máy."
        if not d["pca_ok"]:
            return "VIỆC TIẾP THEO: fit PCA → python -m scripts.fit_pca --images-from-storage"
        if isinstance(qd, int) and qd > 0:
            return f"VIỆC TIẾP THEO: chạy Phase 3 để xử lý {qd:,} message đang chờ"
        if d["review"].get("pending", 0) > 0:
            return (f"VIỆC TIẾP THEO: duyệt {d['review']['pending']:,} mẫu → "
                    "Phase 4, http://127.0.0.1:8010")
        if d["review"].get("done", 0) > 0 and not d["datasets"]:
            return "VIỆC TIẾP THEO: dựng dataset → python scripts/startup.py --phase 5"
        if d["datasets"]:
            return "VIỆC TIẾP THEO: train → python scripts/startup.py --phase 6"
        return "VIỆC TIẾP THEO: thu thêm mẫu (Phase 2)"

    def run(self, interval: float = 5.0, once: bool = False) -> None:
        try:
            while True:
                out = self.render(self.collect())
                if not once:
                    os.system("cls" if os.name == "nt" else "clear")
                print(out, flush=True)
                if once:
                    return
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nĐã dừng theo dõi.")


def main() -> int:
    ap = argparse.ArgumentParser(prog="status_monitor")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", help="in một lần rồi thoát")
    args = ap.parse_args()
    StatusMonitor().run(interval=args.interval, once=args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
