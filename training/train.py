"""Fine-tune. LUÔN từ checkpoint GỐC, không chồng lên model đã fine-tune trước.

Vì sao: fine-tune chồng liên tiếp là công thức của catastrophic forgetting — model
lần thứ 5 quên gần hết những gì lần 1 học, mà không có metric nào của bạn báo động
cho tới khi một camera cũ hỏng. Mỗi lần train là: checkpoint gốc + TOÀN BỘ dataset
đã duyệt (mix 70% cũ / 30% mới).

Augmentation, resize, normalize: ở dataloader. Dữ liệu trên đĩa vẫn là ảnh gốc.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from common.config import Settings, get_settings
from dataset.manifest import Manifest

log = logging.getLogger(__name__)

# Augmentation ở dataloader — ultralytics nhận thẳng các key này.
AUG = dict(
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
    degrees=0.0,        # camera cố định, xoay ảnh là dạy model thứ không tồn tại
    translate=0.08, scale=0.4, shear=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, mixup=0.0, erasing=0.2,
    close_mosaic=10,
)


@dataclass
class TrainResult:
    weights: Path
    run_dir: Path
    metrics: dict
    dataset_version: str
    base_checkpoint: str
    mix: dict


def build_mix_yaml(
    dataset_root: Path,
    old_ratio: float,
    previous_dataset: Path | None,
    out: Path,
    seed: int = 0,
) -> dict:
    """Tạo danh sách file train theo tỉ lệ cũ/mới.

    Nếu chưa có dataset trước đó thì dùng toàn bộ — không có gì để 'giữ lại'.
    """
    new_imgs = sorted((dataset_root / "images" / "train").glob("*"))
    if previous_dataset is None:
        chosen = new_imgs
        info = {"old": 0, "new": len(new_imgs), "ratio_applied": None}
    else:
        old_imgs = sorted((previous_dataset / "images" / "train").glob("*"))
        # Giữ 70% cũ / 30% mới tính theo số ảnh cuối cùng đưa vào train.
        n_new = len(new_imgs)
        n_old_target = int(n_new * old_ratio / max(1e-9, 1 - old_ratio))
        rng = random.Random(seed)
        old_sample = old_imgs if n_old_target >= len(old_imgs) else rng.sample(old_imgs, n_old_target)
        chosen = list(new_imgs) + list(old_sample)
        info = {"old": len(old_sample), "new": n_new, "ratio_applied": old_ratio}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in chosen), encoding="utf-8")
    info["list_file"] = str(out)
    return info


class Trainer:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    def train(
        self,
        dataset_version: str,
        previous_dataset_version: str | None = None,
        epochs: int | None = None,
        tag: str = "main",
        source_filter: str | None = None,
    ) -> TrainResult:
        """source_filter='random' -> train baseline đối chứng cho ablation."""
        from ultralytics import YOLO

        root = self.s.datasets_dir / dataset_version
        data_yaml = root / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(data_yaml)

        prev_root = (
            self.s.datasets_dir / previous_dataset_version if previous_dataset_version else None
        )
        mix = build_mix_yaml(
            root,
            self.s.train_old_data_ratio,
            prev_root,
            root / f"train_list_{tag}.txt",
        )

        if source_filter:
            mix = self._filter_by_source(root, mix, source_filter, tag)

        # RÀNG BUỘC: base checkpoint GỐC, không phải model production hiện tại.
        base = self.s.yolo_base_checkpoint
        model = YOLO(base)
        run_dir = self.s.data_root / "runs" / f"{dataset_version}_{tag}"
        if run_dir.exists():
            shutil.rmtree(run_dir)

        # Không có GPU thì tự chuyển sang CPU (giống realtime), khỏi crash vì device=cuda:0.
        device = self.s.device
        try:
            import torch
            if str(device).startswith("cuda") and not torch.cuda.is_available():
                log.warning("Không thấy GPU (CUDA) — train bằng CPU, sẽ chậm hơn.")
                device = "cpu"
        except Exception:
            device = "cpu"

        train_kwargs = dict(
            data=str(data_yaml),
            epochs=epochs or self.s.train_epochs,
            imgsz=self.s.imgsz,
            batch=self.s.train_batch,
            project=str(Path(run_dir.parent).resolve()),
            name=run_dir.name,
            exist_ok=True,
            pretrained=True,
            seed=0,
            **AUG,
        )
        log.info("train %s từ base=%s, device=%s, mix=%s", dataset_version, base, device, mix)
        try:
            model.train(device=device, **train_kwargs)
        except Exception as e:
            # GPU không tương thích (thiếu kernel cho kiến trúc card, hết VRAM, driver lệch...)
            # -> train lại bằng CPU để vẫn ra model, thay vì chết hẳn.
            if str(device) != "cpu":
                log.warning("Train trên GPU lỗi (%s). Chuyển sang CPU train lại.", e)
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                device = "cpu"
                model = YOLO(base)  # model mới, tránh trạng thái hỏng sau lỗi CUDA
                model.train(device="cpu", **train_kwargs)
            else:
                raise
        # ultralytics tự quyết thư mục lưu (đôi khi khác project/name) -> lấy đúng từ trainer.
        save_dir = Path(getattr(model.trainer, "save_dir", run_dir))
        best = getattr(model.trainer, "best", None)
        weights = Path(best) if best else (save_dir / "weights" / "best.pt")
        if not weights.exists():  # phòng khi chỉ có last.pt
            alt = save_dir / "weights" / "last.pt"
            if alt.exists():
                weights = alt
        metrics_file = save_dir / "results.csv"
        metrics = {"results_csv": str(metrics_file) if metrics_file.exists() else None}

        return TrainResult(
            weights=weights,
            run_dir=save_dir,
            metrics=metrics,
            dataset_version=dataset_version,
            base_checkpoint=base,
            mix=mix,
        )

    def _filter_by_source(self, root: Path, mix: dict, source: str, tag: str) -> dict:
        """Ablation: giữ lại đúng các ảnh có source mong muốn, theo manifest."""
        from sqlalchemy import select

        from common.db import Frame as FrameRow
        from common.db import session_scope

        with session_scope() as s:
            ids = set(
                s.execute(select(FrameRow.id).where(FrameRow.source == source)).scalars().all()
            )
        list_file = Path(mix["list_file"])
        kept = [ln for ln in list_file.read_text(encoding="utf-8").splitlines() if Path(ln).stem in ids]
        new_file = root / f"train_list_{tag}_{source}.txt"
        new_file.write_text("\n".join(kept), encoding="utf-8")
        return {**mix, "list_file": str(new_file), "source_filter": source, "n_after_filter": len(kept)}


def main() -> None:
    import argparse

    from common.db import init_db
    from training.registry import Registry
    from training.trigger import evaluate

    ap = argparse.ArgumentParser(prog="training")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--previous", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="bỏ qua trigger")
    ap.add_argument("--ablation", action="store_true", help="train thêm baseline chỉ-random")
    args = ap.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    init_db()

    if not args.force:
        d = evaluate()
        if not d.should_train:
            print("Chưa đủ điều kiện train:", json.dumps(d.__dict__, default=str, indent=2))
            return
        print("Trigger:", "; ".join(d.reasons))

    t = Trainer()
    res = t.train(args.dataset, args.previous, args.epochs)
    print("weights:", res.weights)

    if args.ablation:
        base = t.train(args.dataset, args.previous, args.epochs, tag="random_only",
                       source_filter="random")
        print("baseline random-only:", base.weights)

    reg = Registry()
    m = reg.register(res.weights, args.dataset, res.metrics, res.base_checkpoint,
                     notes=f"mix={res.mix}")
    print("đã đăng ký model", m.version, "— chạy validate rồi mới deploy")


if __name__ == "__main__":
    main()
