"""Fit PCA MỘT LẦN trên tập khởi tạo rồi freeze.

Chạy TRƯỚC khi bật worker. Worker cố tình không tự fit: nếu nó tự fit, mỗi lần
khởi động lại với dữ liệu khác sẽ ra một phép chiếu khác, và faiss index cũ trở
thành rác mà không có lỗi nào nổ ra.

    python -m scripts.fit_pca --images-from-storage --limit 5000
    python -m scripts.fit_pca --dir /duong/dan/anh/khoi/tao
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from common.config import get_settings
from common.db import Frame as FrameRow
from common.db import init_db, session_scope
from common.storage import get_storage
from worker.embedding import build_embedder
from worker.pca import fit_pca

log = logging.getLogger(__name__)


def collect_images(args, s) -> list:
    storage = get_storage(s)
    imgs = []
    if args.dir:
        paths = sorted(Path(args.dir).rglob("*.jpg")) + sorted(Path(args.dir).rglob("*.png"))
        import cv2

        for p in paths[: args.limit]:
            im = cv2.imread(str(p))
            if im is not None:
                imgs.append(im)
    else:
        with session_scope() as sess:
            rows = sess.query(FrameRow).limit(args.limit).all()
            paths = [r.path for r in rows]
        for rp in paths:
            try:
                imgs.append(storage.read_image(rp))
            except FileNotFoundError:
                continue
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser(prog="fit_pca")
    ap.add_argument("--dir", help="thư mục ảnh khởi tạo")
    ap.add_argument("--images-from-storage", action="store_true")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--embedder", choices=["hist", "yolo"], default="hist")
    ap.add_argument("--min-samples", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    s = get_settings()
    init_db()

    imgs = collect_images(args, s)
    if not imgs:
        raise SystemExit("Không tìm thấy ảnh nào. Thu dữ liệu trước (Phase 2), hoặc dùng --dir.")

    emb = build_embedder(args.embedder, s.yolo_weights, s.device, s.imgsz)
    log.info("embedding %d ảnh bằng %s", len(imgs), emb.name)
    vecs = emb.embed(imgs)

    pca = fit_pca(
        vecs,
        version=s.pca_version,
        embedder=emb.name,
        n_components=s.pca_components,
        min_samples=args.min_samples if args.min_samples is not None else s.pca_fit_min_samples,
    )
    path = pca.save(s.pca_path)
    print(f"PCA {pca.version} đã freeze: {path}")
    print(f"  embedder={pca.embedder} input_dim={pca.input_dim} -> {pca.n_components} chiều")
    print(f"  fit trên {pca.n_fit_samples} mẫu")
    print("Từ giờ KHÔNG fit lại. Muốn đổi: bump PCA_VERSION rồi reindex toàn bộ.")


if __name__ == "__main__":
    main()
