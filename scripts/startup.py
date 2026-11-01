#!/usr/bin/env python3
"""Bộ khởi động theo phase. Mỗi phase kiểm tra điều kiện trước rồi mới chạy.

    python scripts/startup.py --phase 2 --collector
    python scripts/startup.py --phase 3
    python scripts/startup.py --check          # chỉ kiểm tra, không chạy gì

Ba lỗi của bản cũ đã sửa:
  1. Phase 3 gọi `scripts.fit_pca` mỗi lần chạy — nhưng PCA đã freeze thì
     `FrozenPCA.save()` ném PCAFrozenError, và bản cũ vẫn chạy tiếp như không có gì.
     Giờ: chỉ fit khi CHƯA có file, và fit xong mới sang worker.
  2. Phase 3 ép `--embedder yolo` trong khi PCA đã freeze bằng `hist_v1` (512 chiều).
     Worker sẽ ném `ValueError: input_dim ... != 512` ngay message đầu tiên.
     Giờ: đọc embedder từ chính file PCA đã freeze.
  3. Tiến trình detach dùng stdout=PIPE nhưng không ai đọc → đầy buffer ống (~64 KB)
     → tiến trình con TREO. Giờ: kế thừa stdout/stderr của terminal.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Console Windows cũ (cp1252/cp437) không encode được ký tự vẽ khung và emoji.
# Ghi ra màn hình thì Python tự xử lý, nhưng `> out.txt` sẽ ném UnicodeEncodeError.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

log = logging.getLogger("startup")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EMBEDDER_OF = {"hist_v1": "hist", "yolo_backbone_v1": "yolo"}


def banner(title: str) -> None:
    print()
    print("=" * 66)
    print(f"  {title}")
    print("=" * 66)


def run(cmd: list[str], detach: bool = False) -> bool:
    """Chạy lệnh con. stdout/stderr LUÔN kế thừa terminal — không bao giờ PIPE,
    vì PIPE không ai đọc sẽ làm tiến trình con treo khi in đủ ~64 KB."""
    log.info("→ %s", " ".join(cmd))
    try:
        if detach:
            subprocess.Popen(cmd, cwd=PROJECT_ROOT)
            return True
        return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode == 0
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001
        log.error("không chạy được: %s", e)
        return False


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


# ----------------------------------------------------------------------
def preflight(settings) -> dict:
    """Kiểm tra điều kiện, không sửa gì. Trả về dict trạng thái."""
    st: dict = {}
    st["queue_backend"] = settings.queue_backend
    st["pca_exists"] = settings.pca_path.exists()
    st["pca_embedder"] = None
    if st["pca_exists"]:
        meta = settings.pca_path.with_suffix(".json")
        if meta.exists():
            st["pca_embedder"] = json.loads(meta.read_text(encoding="utf-8")).get("embedder")

    st["weights_exists"] = (PROJECT_ROOT / settings.yolo_weights).exists() or Path(
        settings.yolo_weights
    ).exists()

    st["redis_needed"] = settings.queue_backend == "redis"
    st["redis_up"] = False
    if st["redis_needed"]:
        import socket

        with socket.socket() as s:
            s.settimeout(1.0)
            st["redis_up"] = s.connect_ex(("127.0.0.1", 6379)) == 0

    from sqlalchemy import func, select

    from common.db import Frame, ReviewItem, session_scope

    with session_scope() as sess:
        st["n_frames"] = sess.execute(select(func.count()).select_from(Frame)).scalar_one()
        st["n_review_pending"] = sess.execute(
            select(func.count()).select_from(ReviewItem).where(ReviewItem.status == "pending")
        ).scalar_one()
        st["n_review_done"] = sess.execute(
            select(func.count()).select_from(ReviewItem).where(ReviewItem.status == "done")
        ).scalar_one()
    return st


def print_check(settings, st: dict) -> None:
    banner("KIỂM TRA ĐIỀU KIỆN")
    ok = "✓"
    bad = "✗"
    print(f"  {ok} data_root        : {settings.data_root}")
    print(f"  {ok} database         : {settings.database_url}")
    print(f"  {ok} queue backend    : {st['queue_backend']}")
    if st["redis_needed"]:
        print(f"  {ok if st['redis_up'] else bad} Redis :6379       : "
              f"{'đang chạy' if st['redis_up'] else 'KHÔNG kết nối được → docker compose up -d'}")
    else:
        print(f"  {ok} Redis            : không cần (QUEUE_BACKEND=sqlite)")
    print(f"  {ok if st['weights_exists'] else bad} YOLO weights     : {settings.yolo_weights}")
    if st["pca_exists"]:
        print(f"  {ok} PCA              : đã freeze, embedder={st['pca_embedder']}")
    else:
        print(f"  {bad} PCA              : chưa có → sẽ fit tự động ở Phase 3")
    print(f"  {ok} frame đã ghi     : {st['n_frames']:,}")
    print(f"  {ok} chờ duyệt / đã duyệt: {st['n_review_pending']:,} / {st['n_review_done']:,}")
    print()


# ----------------------------------------------------------------------
def phase1(args) -> int:
    banner("PHASE 1 — realtime, KHÔNG thu mẫu")
    print("  Dashboard: http://127.0.0.1:8000")
    print("  Nghiệm thu: FPS thật ≥ target_fps. Chưa đạt thì đừng sang Phase 2.")
    print("  Ctrl+C để dừng.\n")
    cmd = py("-m", "realtime", "--no-collector")
    _add_video_args(cmd, args)
    return 0 if run(cmd) else 1


def phase2(args) -> int:
    banner("PHASE 2 — realtime + collector (ghi ảnh + đẩy queue)")
    print("  Dashboard: http://127.0.0.1:8000")
    print("  Ảnh → var/images/<cam>/<ngày>/<giờ>/<uuid>.jpg")
    print("  Hết hạn mức ngày thì collector NGỪNG nhận — đó là thiết kế, không phải lỗi.")
    print("  Theo dõi: python -m scripts.status_monitor (cửa sổ khác)")
    print("  Ctrl+C để dừng.\n")
    cmd = py("-m", "realtime")
    if args.no_dashboard:
        cmd.append("--no-dashboard")
    _add_video_args(cmd, args)
    return 0 if run(cmd) else 1


def phase3(args, settings, st: dict) -> int:
    banner("PHASE 3 — worker: embedding → PCA → dedup → pre-annotation → review_queue")

    if not st["pca_exists"]:
        log.info("Chưa có PCA. Fit một lần rồi freeze (cần ≥ %d ảnh đã thu).",
                 settings.pca_fit_min_samples)
        if st["n_frames"] < settings.pca_fit_min_samples:
            log.error(
                "Mới có %d frame, cần ≥ %d. Chạy Phase 2 thêm, hoặc hạ PCA_FIT_MIN_SAMPLES.",
                st["n_frames"], settings.pca_fit_min_samples,
            )
            return 1
        if not run(py("-m", "scripts.fit_pca", "--images-from-storage",
                      "--limit", str(max(5000, settings.pca_fit_min_samples)),
                      "--embedder", args.embedder or "hist")):
            log.error("Fit PCA thất bại — dừng, không chạy worker với PCA hỏng.")
            return 1
        st = preflight(settings)

    embedder = args.embedder or EMBEDDER_OF.get(st["pca_embedder"] or "", "hist")
    if st["pca_embedder"] and EMBEDDER_OF.get(st["pca_embedder"]) != embedder:
        log.error(
            "PCA freeze bằng '%s' nhưng bạn chọn --embedder %s. Số chiều sẽ lệch và "
            "worker ném ValueError. Muốn đổi embedder: bump PCA_VERSION rồi reindex.",
            st["pca_embedder"], embedder,
        )
        return 1

    log.info("Dùng embedder=%s (khớp PCA đã freeze: %s)", embedder, st["pca_embedder"])
    cmd = py("-m", "worker.consumer", "--embedder", embedder)
    if args.no_preannot_model:
        cmd.append("--no-preannot-model")
    if args.max_messages:
        cmd += ["--max-messages", str(args.max_messages)]
    print("  Ctrl+C để dừng.\n")
    return 0 if run(cmd) else 1


def phase4(args, st: dict) -> int:
    banner("PHASE 4 — review UI (cửa DUY NHẤT vào dataset)")
    if st["n_review_pending"] == 0:
        log.warning("Không có mẫu nào chờ duyệt. Chạy Phase 3 trước để đổ mẫu vào review_queue.")
    print("  UI: http://127.0.0.1:8010")
    print("  A = duyệt nhanh · S = sửa box rồi lưu · R = loại (phải chọn lý do)")
    print("  Ctrl+C để dừng.\n")
    return 0 if run(py("-m", "review.app")) else 1


def phase5(args, st: dict) -> int:
    banner("PHASE 5 — dựng dataset từ mẫu ĐÃ DUYỆT")
    if st["n_review_done"] == 0:
        log.error("Chưa có mẫu nào duyệt xong. Chạy Phase 4 trước.")
        return 1
    log.info("%d mẫu đã duyệt sẽ được đưa vào dataset.", st["n_review_done"])
    cmd = py("-m", "dataset.builder", "--notes", args.notes)
    if getattr(args, "frame_split", False):
        cmd.append("--frame-split")
    return 0 if run(cmd) else 1


def phase6(args) -> int:
    banner("PHASE 6 — fine-tune từ checkpoint GỐC")
    dataset = args.dataset
    if dataset == "latest":
        from common.config import get_settings
        ddir = get_settings().datasets_dir
        vers = sorted(p.name for p in ddir.glob("v*") if p.is_dir())
        if not vers:
            log.error("Chưa có dataset nào. Chạy Phase 5 trước.")
            return 1
        dataset = vers[-1]
        log.info("Dùng dataset mới nhất: %s", dataset)
    if not dataset:
        log.error("Thiếu --dataset. Ví dụ: --phase 6 --dataset v001 (hoặc --dataset latest)")
        return 1
    cmd = py("-m", "training.train", "--dataset", dataset)
    if args.previous:
        cmd += ["--previous", args.previous]
    if args.force:
        cmd.append("--force")
    if args.ablation:
        cmd.append("--ablation")
    return 0 if run(cmd) else 1


def phase7(args) -> int:
    banner("PHASE 7 — deploy: shadow → canary → production")
    if args.promote:
        return 0 if run(py("-m", "deploy.controller", "promote", "--version", args.promote)) else 1
    if args.rollback:
        return 0 if run(py("-m", "deploy.controller", "rollback")) else 1
    return 0 if run(py("-m", "deploy.controller", "status")) else 1


# ----------------------------------------------------------------------
def _add_video_args(cmd: list[str], args) -> None:
    """Nối cờ test bằng video (Phase 1/2) vào lệnh `python -m realtime`."""
    if getattr(args, "video", None):
        cmd += ["--video", args.video]
        if getattr(args, "fps", None):
            cmd += ["--fps", str(args.fps)]
        if getattr(args, "loop", False):
            cmd.append("--loop")


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(prog="startup", description="Khởi động yolo-al theo phase")
    ap.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra điều kiện rồi thoát")
    ap.add_argument("--skip-db", action="store_true")
    ap.add_argument("--collector", action="store_true",
                    help="(giữ cho tương thích — Phase 2 vốn đã bật collector)")
    ap.add_argument("--skip-redis", action="store_true",
                    help="(giữ cho tương thích — tự bỏ qua khi QUEUE_BACKEND=sqlite)")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--video", default=None,
                    help="Phase 1/2: dùng file video thay camera để test, vd --video samples/test.mp4")
    ap.add_argument("--fps", type=float, default=None,
                    help="Phase 1/2: giới hạn tốc độ đọc video (fps), dùng với --video")
    ap.add_argument("--loop", action="store_true",
                    help="Phase 1/2: lặp lại video/danh sách khi hết (mặc định hết thì DỪNG)")
    ap.add_argument("--embedder", choices=["hist", "yolo"], default=None,
                    help="mặc định: lấy đúng embedder mà PCA đã freeze")
    ap.add_argument("--no-preannot-model", action="store_true",
                    help="Phase 3: không nạp YOLO cho pre-annotation (nhanh hơn, gợi ý thô hơn)")
    ap.add_argument("--max-messages", type=int, default=None)
    ap.add_argument("--notes", default="", help="Phase 5: ghi chú cho dataset version")
    ap.add_argument("--frame-split", action="store_true",
                    help="Phase 5 (TEST): chia dataset theo frame khi chỉ có 1 video/session")
    ap.add_argument("--dataset", default=None, help="Phase 6: dataset version, vd v001 hoặc 'latest'")
    ap.add_argument("--previous", default=None, help="Phase 6: dataset version trước đó")
    ap.add_argument("--force", action="store_true", help="Phase 6: bỏ qua điều kiện trigger")
    ap.add_argument("--ablation", action="store_true", help="Phase 6: train thêm baseline random")
    ap.add_argument("--promote", default=None, help="Phase 7: model version cần đẩy lên")
    ap.add_argument("--rollback", action="store_true",
                    help="Phase 7: kiểm tra telemetry và tự quay về bản trước nếu tụt")
    args = ap.parse_args()

    if not args.phase and not args.check:
        ap.error("cần --phase N hoặc --check")

    from common.config import get_settings
    from common.db import init_db

    settings = get_settings()
    if not args.skip_db:
        init_db()

    if settings.queue_backend == "redis":
        import socket

        with socket.socket() as sk:
            sk.settimeout(1.0)
            up = sk.connect_ex(("127.0.0.1", 6379)) == 0
        if not up and not args.skip_redis:
            log.info("Redis chưa chạy, thử `docker compose up -d`…")
            run(["docker", "compose", "up", "-d"])
            time.sleep(2)

    st = preflight(settings)
    print_check(settings, st)
    if args.check:
        return 0

    if st["redis_needed"] and not st["redis_up"]:
        log.error("QUEUE_BACKEND=redis nhưng Redis không chạy. Bật Redis, hoặc đặt "
                  "QUEUE_BACKEND=sqlite trong .env.")
        return 1

    try:
        if args.phase == 1:
            return phase1(args)
        if args.phase == 2:
            return phase2(args)
        if args.phase == 3:
            return phase3(args, settings, st)
        if args.phase == 4:
            return phase4(args, st)
        if args.phase == 5:
            return phase5(args, st)
        if args.phase == 6:
            return phase6(args)
        if args.phase == 7:
            return phase7(args)
    except KeyboardInterrupt:
        print("\nĐã dừng.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
