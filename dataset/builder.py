"""Dataset Builder — nơi DUY NHẤT sinh ra dữ liệu training.

Nó chỉ đọc review_queue với status='done' và action in (approve, fix). Không có
hàm nào ở đây nhận nhãn do model tự sinh. Đó là cách ràng buộc "không đường tắt
vào dataset" được thực thi bằng code chứ không bằng lời hứa.

Lưu gì (spec 2.6a):
  - ẢNH GỐC (symlink/copy, không đụng vào pixel) + label YOLO .txt
  - KHÔNG resize / normalize / denoise. Nướng 640×640 vào dữ liệu thì sau này
    đổi sang 960 là mất thông tin vĩnh viễn.
  - ROI = mask vùng không quan tâm, KHÔNG phải crop bbox. Crop bbox rồi lưu thì
    không còn dataset train detector, chỉ còn classifier.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from common.config import Settings, get_settings
from common.db import DatasetVersion, Frame as FrameRow, ReviewItem, session_scope
from common.schemas import BBox
from common.storage import get_storage
from dataset.manifest import Manifest
from dataset.split import GOLDEN_FILE, GoldenSet, Split, assert_no_leakage, split_sessions

log = logging.getLogger(__name__)

MIN_SESSIONS_FOR_GOLDEN = 10  # dưới mức này thì chưa đóng băng bộ chuẩn

DATA_YAML = """# sinh tự động bởi dataset/builder.py — đừng sửa tay
path: {root}
train: images/train
val: images/val
test: images/test
names:
{names}
"""


@dataclass
class ReviewedFrame:
    frame_id: str
    path: str
    cam_id: str
    session_key: str
    ts: datetime
    width: int
    height: int
    source: str
    action: str
    reviewer: str | None
    preannot_model: str | None
    boxes: list[BBox]


def load_reviewed(since: datetime | None = None) -> list[ReviewedFrame]:
    with session_scope() as s:
        q = (
            select(ReviewItem, FrameRow)
            .join(FrameRow, FrameRow.id == ReviewItem.frame_id)
            .where(
                ReviewItem.status == "done",
                ReviewItem.action.in_(["approve", "fix"]),
                FrameRow.deleted_at.is_(None),
            )
        )
        if since:
            q = q.where(ReviewItem.reviewed_at >= since)
        rows = s.execute(q).all()

    out: list[ReviewedFrame] = []
    for item, frame in rows:
        boxes = [BBox(**b) for b in (item.final_boxes or [])]
        # Frame approve/fix mà không còn box nào = ảnh nền không có người.
        # Vẫn giữ: negative sample là dữ liệu hợp lệ và hiếm khi được thu đủ.
        out.append(
            ReviewedFrame(
                frame_id=frame.id,
                path=frame.path,
                cam_id=frame.cam_id,
                session_key=frame.session_key,
                ts=frame.ts,
                width=frame.width,
                height=frame.height,
                source=frame.source,
                action=item.action,
                reviewer=item.reviewer,
                preannot_model=frame.preannot_model,
                boxes=boxes,
            )
        )
    return out


def next_version(root: Path) -> str:
    """Số version lấy từ CẢ thư mục lẫn DB.

    Chỉ nhìn thư mục là sai: xoá `var/datasets/v001` đi thì hàm trả về v001 lần nữa,
    trong khi bảng `dataset_versions` vẫn còn v001 -> INSERT vi phạm UNIQUE và
    Phase 5 chết bằng một IntegrityError không nói lên điều gì.
    """
    nums: list[int] = []
    for p in Path(root).glob("v*"):
        if p.is_dir() and p.name[1:].isdigit():
            nums.append(int(p.name[1:]))
    try:
        with session_scope() as s:
            for (v,) in s.execute(select(DatasetVersion.version)).all():
                if v.startswith("v") and v[1:].isdigit():
                    nums.append(int(v[1:]))
    except Exception:  # noqa: BLE001 - DB chưa có bảng thì cứ dùng thư mục
        pass
    return f"v{max(nums) + 1 if nums else 1:03d}"


class DatasetBuilder:
    def __init__(self, settings: Settings | None = None, link_mode: str = "auto"):
        self.s = settings or get_settings()
        self.storage = get_storage(self.s)
        self.link_mode = link_mode  # auto | symlink | copy

    def _place(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return
        mode = self.link_mode
        if mode == "auto":
            mode = "symlink" if os.name != "nt" else "copy"
        if mode == "symlink":
            try:
                dst.symlink_to(src)
                return
            except OSError:
                pass  # Windows không có quyền tạo symlink -> copy
        shutil.copy2(src, dst)

    def build(self, notes: str = "", freeze_golden_if_empty: bool = True,
              frame_split: bool = False) -> tuple[str, Path, Manifest]:
        frames = load_reviewed()
        if not frames:
            raise RuntimeError("Chưa có frame nào được duyệt. Chạy review UI trước.")

        sessions = sorted({f.session_key for f in frames})
        golden = GoldenSet(self.s.datasets_dir / GOLDEN_FILE)
        if not golden.frozen and freeze_golden_if_empty:
            # Golden ĐÓNG BĂNG VĨNH VIỄN — freeze() sẽ từ chối mọi lần gọi sau.
            # Nên KHÔNG được đóng băng khi mới có vài session: với 1 session,
            # 10% vẫn làm tròn lên 1, session duy nhất đó rơi vào golden -> vào test,
            # train rỗng, và sai lầm đó không bao giờ sửa được nữa.
            if len(sessions) < MIN_SESSIONS_FOR_GOLDEN:
                log.warning(
                    "Mới có %d session (%s) — CHƯA đóng băng golden set. "
                    "Cần >= %d session (camera + ngày khác nhau), vì golden đã đóng băng "
                    "là vĩnh viễn, đóng băng sớm sẽ khoá luôn dữ liệu duy nhất vào test.",
                    len(sessions), ", ".join(sessions), MIN_SESSIONS_FOR_GOLDEN,
                )
            else:
                n = max(1, len(sessions) // 10)
                golden.freeze(sessions[:n])
                log.info("Golden set đóng băng với %d/%d session", n, len(sessions))

        split = split_sessions(
            sessions,
            self.s.split_tuple,
            golden.sessions,
            self.s.rolling_days,
        )
        assert_no_leakage(split)

        # Split là THEO SESSION, nên một session không thể vừa train vừa val.
        # Dữ liệu 1 camera trong 1 ngày = 1 session = chỉ vào được đúng một tập.
        # Dựng ra dataset train rỗng rồi để Phase 6 chết ở đó thì khó lần ra nguyên nhân.
        if not frame_split and (not split.train or not split.val):
            raise RuntimeError(
                f"Chỉ có {len(sessions)} session ({', '.join(sessions)}), chia ra "
                f"train/val/test = {len(split.train)}/{len(split.val)}/{len(split.test)} "
                "— thiếu train hoặc val, dataset này không train được.\n\n"
                "Nguyên nhân: split chia theo SESSION (camera + ngày), cố ý như vậy. "
                "Frame liền nhau trong một track giống nhau gần như tuyệt đối; chia theo "
                "frame thì mọi ảnh val đều có anh em sinh đôi trong train, mAP đẹp nhưng "
                "vô nghĩa. Một session chỉ vào được đúng MỘT tập.\n\n"
                "Cách xử lý: thu và duyệt thêm mẫu ở NHIỀU NGÀY khác nhau (hoặc thêm "
                f"camera). Thực tế cần khoảng {MIN_SESSIONS_FOR_GOLDEN} cặp (camera, ngày) "
                "thì tỉ lệ 0.8/0.1/0.1 mới chia ra được cả ba tập."
            )

        frame_part: dict[str, str] = {}
        if frame_split:
            log.warning(
                "FRAME-SPLIT (chế độ TEST): chia train/val/test theo FRAME, không theo "
                "session. Frame liền nhau rất giống nhau nên val/test sẽ 'lọt' — chỉ dùng "
                "để chạy thử pipeline, KHÔNG dùng đánh giá thật."
            )
            tr, va, te = self.s.split_tuple
            order = sorted(frames, key=lambda f: f.frame_id)
            n = len(order)
            n_val = max(1, round(n * va)) if n >= 3 else 0
            n_test = max(1, round(n * te)) if n >= 3 else 0
            n_train = n - n_val - n_test
            for i, f in enumerate(order):
                frame_part[f.frame_id] = (
                    "train" if i < n_train else "val" if i < n_train + n_val else "test"
                )

        version = next_version(self.s.datasets_dir)
        root = self.s.datasets_dir / version
        counts = {"train": 0, "val": 0, "test": 0}
        man = Manifest(version=version, pca_version=self.s.pca_version, notes=notes)

        # YOLO cần class id LIỀN MẠCH 0..N-1, nhưng nhãn của ta dùng COCO id (vd 0,2,3,5,7).
        # Gộp lớp có tên (CLASS_NAMES + labels.json) với lớp thực xuất hiện trong nhãn,
        # sắp xếp rồi đánh số lại về 0..N-1. data.yaml sẽ liệt kê theo số mới này.
        name_map = self.s.class_name_map()
        used_ids = set(name_map)
        for f in frames:
            for b in f.boxes:
                used_ids.add(b.cls)
        ordered_ids = sorted(used_ids)
        id_remap = {orig: i for i, orig in enumerate(ordered_ids)}
        names_block = "\n".join(
            f"  {i}: {name_map.get(orig, f'class{orig}')}" for i, orig in enumerate(ordered_ids)
        )

        for f in frames:
            part = frame_part.get(f.frame_id) if frame_split else split.of(f.session_key)
            if part is None:
                continue
            src = self.storage.local_path(f.path)
            if src is None or not src.exists():
                log.warning("thiếu ảnh %s, bỏ qua", f.path)
                continue

            stem = f.frame_id
            self._place(src, root / "images" / part / f"{stem}{src.suffix}")
            label = "\n".join(
                f"{id_remap[b.cls]} " + b.to_yolo(f.width, f.height).split(" ", 1)[1]
                for b in f.boxes
            )
            lp = root / "labels" / part / f"{stem}.txt"
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text(label + ("\n" if label else ""), encoding="utf-8")

            counts[part] += 1
            man.source_distribution[f.source] = man.source_distribution.get(f.source, 0) + 1
            man.action_distribution[f.action] = man.action_distribution.get(f.action, 0) + 1
            man.cameras[f.cam_id] = man.cameras.get(f.cam_id, 0) + 1
            if f.reviewer:
                man.reviewers[f.reviewer] = man.reviewers.get(f.reviewer, 0) + 1
            if f.preannot_model:
                man.preannot_models[f.preannot_model] = man.preannot_models.get(f.preannot_model, 0) + 1

        (root / "data.yaml").write_text(
            DATA_YAML.format(root=Path(root).resolve().as_posix(), names=names_block),
            encoding="utf-8",
        )
        man.counts = counts
        man.sessions = {"train": split.train, "val": split.val, "test": split.test,
                        "golden": split.golden, "rolling": split.rolling}
        man.golden_frozen_at = (
            json.loads((self.s.datasets_dir / GOLDEN_FILE).read_text(encoding="utf-8"))["frozen_at"]
            if golden.frozen else None
        )
        man.save(root / "manifest.json")

        # Danh sách file cho từng bộ eval — validate.py đọc thẳng từ đây.
        for name, keys in (("golden", split.golden), ("rolling", split.rolling)):
            ids = [f.frame_id for f in frames if f.session_key in set(keys)]
            (root / f"eval_{name}.json").write_text(json.dumps(ids), encoding="utf-8")

        with session_scope() as s:
            s.add(
                DatasetVersion(
                    version=version,
                    path=str(root),
                    manifest=json.loads((root / "manifest.json").read_text(encoding="utf-8")),
                    n_train=counts["train"],
                    n_val=counts["val"],
                    n_test=counts["test"],
                )
            )

        log.info("dataset %s: %s", version, counts)
        return version, root, man


def apply_roi_mask(image, roi_polygon) -> "object":
    """ROI = che vùng KHÔNG quan tâm, giữ nguyên kích thước ảnh và toạ độ bbox.

    Không crop. Crop là thứ làm dataset detector biến thành dataset classifier.
    Dùng lúc train (dataloader) hoặc lúc infer, KHÔNG dùng lúc lưu.
    """
    import cv2
    import numpy as np

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_polygon, dtype=np.int32)], 255)
    return cv2.bitwise_and(image, image, mask=mask)


def main() -> None:
    import argparse

    from common.db import init_db

    ap = argparse.ArgumentParser(prog="dataset")
    ap.add_argument("--notes", default="")
    ap.add_argument("--link-mode", choices=["auto", "symlink", "copy"], default="auto")
    ap.add_argument("--frame-split", action="store_true",
                    help="TEST: chia train/val/test theo frame (khi chỉ có 1 session/1 video)")
    args = ap.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    init_db()
    v, root, man = DatasetBuilder(link_mode=args.link_mode).build(args.notes, frame_split=args.frame_split)
    print(f"dataset {v} -> {root}")
    print(json.dumps(man.counts, indent=2))
    print("nguồn:", json.dumps(man.source_distribution, indent=2))


if __name__ == "__main__":
    main()
