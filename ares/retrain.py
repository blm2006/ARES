"""Retrain ban đêm (00:00): Approved_Images + Replay_Buffer_From_Base_Dataset -> .pt mới.

Quy tắc giữ nguyên từ workflow v2 vì chúng đúng:
  * Luôn fine-tune từ CHECKPOINT GỐC, không chồng lên model đã fine-tune (tránh quên dần).
  * Mix ~70 % dữ liệu cũ (replay) / 30 % mới.
  * Tập giữ lại (holdout) chia theo NHÓM thời gian (ngày + khối 3 giờ), không chia theo khung hình:
    các khung hình liền nhau gần như y hệt nhau, chia ngẫu nhiên là rò rỉ và làm mAP đẹp giả.
  * Chính sách lên production (ARES_AUTO_PROMOTE):
      - always (mặc định theo workflow): train xong là hoán đổi ngay, không ngắt nhận diện. Cổng chất lượng vẫn được
        TÍNH và lưu vào sổ model; nếu model mới kém hơn model cũ thì phát cảnh báo MODEL_GATE_WARN để admin Rollback.
      - gated: chỉ hoán đổi khi KHÔNG tệ hơn model đang chạy trên tập giữ lại; không có tập giữ lại thì chờ Promote tay.
      - never: luôn chờ admin bấm Promote.

Việc train chạy ở TIẾN TRÌNH CON (ares.train_job): không tranh GIL với luồng vision và
lỗi CUDA/hết VRAM không làm chết cả hệ thống.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from ares.active_learner import ActiveLearner
from ares.config import REPO_ROOT, AresSettings
from ares.db_manager import ActiveLearningItem
from ares.model_registry import ModelRegistry
from ares.state import Event

log = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class DatasetInfo:
    run_dir: Path
    train_list: Path
    val_list: Path
    val_human_list: Path | None
    val_replay_list: Path | None
    n_human_train: int = 0
    n_human_val: int = 0
    n_replay_train: int = 0
    n_replay_val: int = 0
    has_holdout: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "run_dir": str(self.run_dir), "train_list": str(self.train_list), "val_list": str(self.val_list),
            "val_human_list": str(self.val_human_list) if self.val_human_list else None,
            "val_replay_list": str(self.val_replay_list) if self.val_replay_list else None,
            "n_human_train": self.n_human_train, "n_human_val": self.n_human_val,
            "n_replay_train": self.n_replay_train, "n_replay_val": self.n_replay_val,
            "has_holdout": self.has_holdout, "warnings": self.warnings,
        }


def group_key(captured_at: datetime) -> str:
    """Nhóm (ngày, khối 3 giờ): mọi khung hình trong cùng nhóm luôn cùng phía train hoặc val."""
    return f"{captured_at:%Y-%m-%d}_{captured_at.hour // 3}"


def in_val_group(key: str, fraction: float) -> bool:
    return (zlib.crc32(key.encode("utf-8")) % 1000) < int(fraction * 1000)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _find_replay_pairs(replay_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    img_root = replay_dir / "images"
    if not img_root.exists():
        return pairs
    for p in sorted(img_root.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        rel = p.relative_to(img_root)
        lab = (replay_dir / "labels" / rel).with_suffix(".txt")
        if lab.exists():  # thiếu file nhãn = chưa gán nhãn, không phải 'không có người' -> bỏ
            pairs.append((p, lab))
    return pairs


def build_dataset(cfg: AresSettings, items: list[ActiveLearningItem], run_dir: Path, seed: int = 0) -> DatasetInfo:
    """Dựng thư mục YOLO tự chứa (images/<split>, labels/<split>) + các file danh sách."""
    run_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    splits: dict[str, list[Path]] = {"train": [], "val_human": [], "val_replay": []}
    for sp in splits:
        (run_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (run_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    # ---- dữ liệu người đã duyệt ----------------------------------------------------
    humans = []
    for it in sorted(items, key=lambda i: i.captured_at):
        img = cfg.storage_root / it.img_path
        lab = img.with_suffix(".txt")
        if not img.exists() or not lab.exists():
            warnings.append(f"Bỏ {it.image_id}: thiếu file ảnh/nhãn")
            continue
        humans.append((it, img, lab))
    val_h = [(it, a, b) for it, a, b in humans if in_val_group(group_key(it.captured_at), cfg.val_fraction)]
    if len(val_h) < cfg.min_holdout:
        if val_h:
            warnings.append(f"Tập giữ lại chỉ có {len(val_h)} ảnh (< {cfg.min_holdout}) — không đủ tin cậy, dồn vào train.")
        val_h = []
    val_ids = {it.image_id for it, _, _ in val_h}
    n_ht = n_hv = 0
    for it, img, lab in humans:
        sp = "val_human" if it.image_id in val_ids else "train"
        _link_or_copy(img, run_dir / "images" / sp / img.name)
        _link_or_copy(lab, run_dir / "labels" / sp / lab.name)
        splits[sp].append(run_dir / "images" / sp / img.name)
        if sp == "train":
            n_ht += 1
        else:
            n_hv += 1

    # ---- replay buffer ----------------------------------------------------------------
    n_rt = n_rv = 0
    if cfg.replay_dir and Path(cfg.replay_dir).exists():
        pairs = _find_replay_pairs(Path(cfg.replay_dir))
        if not pairs:
            warnings.append(f"replay_dir {cfg.replay_dir} không có cặp ảnh/nhãn nào.")
        # tập replay giữ lại CỐ ĐỊNH theo tên file => ổn định giữa các đêm
        rv = [p for p in pairs if (zlib.crc32(p[0].name.encode()) % 1000) < int(cfg.val_fraction * 1000)][:200]
        rv_set = {p[0] for p in rv}
        rtrain = [p for p in pairs if p[0] not in rv_set]
        ratio = cfg.replay_old_ratio
        target = math.ceil(max(1, n_ht) * ratio / (1.0 - ratio))
        rng = random.Random(seed)
        if len(rtrain) > target:
            rtrain = rng.sample(rtrain, target)
        elif rtrain and len(rtrain) < target:
            warnings.append(f"Replay chỉ có {len(rtrain)}/{target} ảnh để đạt tỉ lệ {int(ratio * 100)}/{100 - int(ratio * 100)}.")
        for sp, group in (("train", rtrain), ("val_replay", rv)):
            for img, lab in group:
                name = f"rp_{img.parent.name}_{img.name}"
                _link_or_copy(img, run_dir / "images" / sp / name)
                _link_or_copy(lab, run_dir / "labels" / sp / Path(name).with_suffix(".txt").name)
                splits[sp].append(run_dir / "images" / sp / name)
        n_rt, n_rv = len(rtrain), len(rv)
    else:
        warnings.append("Chưa cấu hình replay buffer (ARES_REPLAY_DIR) — chỉ train trên ảnh người duyệt, "
                        "dùng freeze lớp đầu và lr nhỏ để giảm quên.")

    def write_list(name: str, paths: list[Path]) -> Path | None:
        if not paths:
            return None
        f = run_dir / name
        f.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")  # đường dẫn theo hệ điều hành
        return f

    train_list = write_list("train.txt", splits["train"])
    vh, vr = write_list("val_human.txt", splits["val_human"]), write_list("val_replay.txt", splits["val_replay"])
    has_holdout = bool(vh or vr)
    val_all = splits["val_human"] + splits["val_replay"]
    if not val_all:  # ultralytics bắt buộc có val; không có holdout thì dùng vài ảnh train (chỉ để chọn epoch)
        val_all = splits["train"][:20]
        warnings.append("Không có tập giữ lại độc lập: model mới sẽ KHÔNG được tự động promote.")
    val_list = write_list("val.txt", val_all)
    if train_list is None:
        raise ValueError("Không có ảnh train nào")
    return DatasetInfo(run_dir, train_list, val_list, vh, vr, n_ht, n_hv, n_rt, n_rv, has_holdout, warnings)


# ---------------------------------------------------------------------------
# Cổng chất lượng
# ---------------------------------------------------------------------------
def evaluate_gate(metrics: dict, tol: float) -> dict:
    """So model ứng viên với model đang chạy trên mọi tập giữ lại có sẵn.

    passed: True = không tệ hơn (trong dung sai) trên mọi tập; False = tệ hơn ở ít nhất một tập;
            None = không có tập giữ lại để so (cần người quyết định).
    """
    cand, cur = metrics.get("candidate", {}), metrics.get("current", {})
    sets = [k for k in ("human", "replay") if k in cand and k in cur]
    if not sets:
        return {"passed": None, "reason": "Không có tập giữ lại để so sánh.", "checks": {}}
    checks, ok = {}, True
    for k in sets:
        c, p = cand[k], cur[k]
        rec_ok = c["recall"] >= p["recall"] - tol
        map_ok = c["map50"] >= p["map50"] - tol
        checks[k] = {"recall": [p["recall"], c["recall"]], "map50": [p["map50"], c["map50"]], "passed": bool(rec_ok and map_ok)}
        ok = ok and rec_ok and map_ok
    bad = [k for k, v in checks.items() if not v["passed"]]
    return {"passed": ok, "checks": checks,
            "reason": "Đạt trên mọi tập giữ lại." if ok else f"Kém hơn model hiện tại trên tập: {', '.join(bad)}."}


# ---------------------------------------------------------------------------
# Chạy train trong tiến trình con
# ---------------------------------------------------------------------------
class SubprocessTrainer:
    def __init__(self, cfg: AresSettings):
        self.cfg = cfg

    def __call__(self, job: dict) -> dict:
        run_dir = Path(job["run_dir"])
        job_file = run_dir / "job.json"
        job_file.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        result_file = run_dir / "result.json"
        result_file.unlink(missing_ok=True)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS: nhường CPU cho vision
        else:
            kwargs["preexec_fn"] = lambda: os.nice(10)
        with open(run_dir / "train.log", "w", encoding="utf-8") as logf:
            proc = subprocess.Popen([sys.executable, "-m", "ares.train_job", str(job_file)], cwd=str(REPO_ROOT),
                                    stdout=logf, stderr=subprocess.STDOUT, env=env, **kwargs)
            try:
                proc.wait(timeout=self.cfg.retrain_timeout_h * 3600)
            except subprocess.TimeoutExpired:
                proc.kill()
                return {"ok": False, "error": f"Quá {self.cfg.retrain_timeout_h:g} giờ, đã dừng tiến trình train."}
        if not result_file.exists():
            return {"ok": False, "error": f"Tiến trình train thoát (mã {proc.returncode}) mà không có kết quả — xem {run_dir / 'train.log'}"}
        return json.loads(result_file.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Điều phối
# ---------------------------------------------------------------------------
class Retrainer:
    def __init__(self, cfg: AresSettings, clock, al: ActiveLearner, registry: ModelRegistry,
                 trainer: Callable[[dict], dict] | None = None, emit: Callable[[Event], None] | None = None, pca=None):
        self.cfg, self.clock, self.al, self.registry = cfg, clock, al, registry
        self.pca = pca  # ares.pca_filter.PcaFilter | None: lọc PCA lần 2 (chọn ảnh đại diện) trước khi train
        self.trainer = trainer or SubprocessTrainer(cfg)
        self.emit = emit if emit is not None else (lambda e: None)
        self._lock = threading.Lock()
        self.running = False
        self.last_result: dict | None = None

    def _ev(self, kind: str, sev: str, msg: str, **data) -> None:
        self.emit(Event(t=self.clock.time(), kind=kind, severity=sev, message=msg, data=data))

    def run(self, trigger: str = "midnight", force: bool = False) -> dict:
        if not self._lock.acquire(blocking=False):
            return {"status": "busy", "reason": "Một lần train khác đang chạy."}
        self.running = True
        try:
            res = self._run(trigger, force)
        except Exception as e:
            log.exception("Retrain lỗi")
            self._ev("RETRAIN_FAILED", "error", f"Retrain lỗi: {e}")
            res = {"status": "failed", "error": str(e)}
        finally:
            self.running = False
            self._lock.release()
        self.last_result = {**res, "at": self.clock.now().isoformat(timespec="seconds"), "trigger": trigger}
        return res

    def _run(self, trigger: str, force: bool) -> dict:
        cfg = self.cfg
        rows = self.al.items_for_training()
        if not rows:
            return {"status": "skipped", "reason": "Chưa có ảnh nào được duyệt."}
        # LỌC PCA LẦN 2: ảnh APPROVED nào thật sự đáng train. Ảnh bị coi là thừa (gần giống ảnh đã train/đã chọn)
        # bị loại khỏi CẢ train lẫn tập giữ lại (để khỏi rò rỉ), nhưng vẫn giữ trạng thái APPROVED.
        sel = None
        if self.pca is not None:
            sel = self.pca.select_for_training(rows, commit=True)
            dropped = set(sel["dropped_ids"])
            rows = [r for r in rows if r.image_id not in dropped]
        new = [r for r in rows if r.status == "APPROVED"]
        if not force and (not new or len(new) < cfg.retrain_min_new):
            why = "đều trùng với ảnh đã train" if sel and sel["n_dropped"] and not new else "Chưa có ảnh APPROVED mới"
            return {"status": "skipped", "reason": f"{why} ({len(new)}).", "pca": _slim(sel)}
        now = self.clock.now()
        run_id = f"run_{now:%Y%m%d_%H%M%S}"
        run_dir = cfg.datasets_dir / run_id
        extra = f", PCA đã loại {sel['n_dropped']} ảnh thừa" if sel and sel["n_dropped"] else ""
        self._ev("RETRAIN_START", "info", f"Bắt đầu retrain ({trigger}): {len(new)} ảnh mới, {len(rows)} ảnh người đã duyệt tổng{extra}.")
        info = build_dataset(cfg, rows, run_dir, seed=now.toordinal())
        for w in info.warnings:
            self._ev("RETRAIN_WARN", "warn", w)
        prod_ver, prod_path = self.registry.production()
        n_human = info.n_human_train + info.n_human_val
        job = {
            "run_dir": str(run_dir), "train_list": str(info.train_list), "val_list": str(info.val_list),
            "val_human_list": str(info.val_human_list) if info.val_human_list else None,
            "val_replay_list": str(info.val_replay_list) if info.val_replay_list else None,
            "base_weights": str(cfg.weights_path()), "current_weights": str(prod_path),
            "epochs": cfg.retrain_epochs, "imgsz": cfg.retrain_imgsz, "batch": cfg.retrain_batch,
            "freeze": cfg.retrain_freeze if n_human < cfg.retrain_freeze_until_n else 0,
            "lr0": cfg.retrain_lr0, "device": cfg.device, "workers": 2,
        }
        t0 = time.time()
        res = self.trainer(job)
        dur = time.time() - t0
        if not res.get("ok"):
            self._ev("RETRAIN_FAILED", "error", f"Retrain thất bại: {res.get('error', '?')}")
            return {"status": "failed", "error": res.get("error"), "run_dir": str(run_dir)}

        gate = evaluate_gate(res.get("metrics", {}), cfg.gate_tolerance)
        if not info.has_holdout:
            gate = {"passed": None, "reason": "Không có tập giữ lại độc lập.", "checks": {}}
        policy = cfg.auto_promote
        if policy == "always":
            promote, status = True, "production"
        elif policy == "never":
            promote, status = False, "candidate"
        elif gate["passed"] is True:
            promote, status = True, "production"
        elif gate["passed"] is False:
            promote, status = False, "rejected"
        else:
            promote, status = False, "candidate"

        if promote and policy == "always" and gate["passed"] is False:
            self._ev("MODEL_GATE_WARN", "warn",
                     f"Model mới KÉM HƠN model cũ trên tập giữ lại ({gate['reason']}) nhưng vẫn được hoán đổi do "
                     f"ARES_AUTO_PROMOTE=always. Nếu nhận diện tệ đi, bấm 'Quay lui phiên bản trước' ở tab Model.", gate=gate)
        mv = self.registry.add(
            Path(res["weights"]), parent=prod_ver, trigger=trigger, n_new=len(new), n_total=n_human,
            n_replay=info.n_replay_train, metrics=res.get("metrics", {}), gate=gate,
            status="candidate" if promote else status,
            notes=f"epochs={cfg.retrain_epochs} freeze={job['freeze']} thời gian={dur / 60:.1f} phút; " + "; ".join(info.warnings),
        )
        if promote:
            self.registry.promote(mv.version)
        self.al.mark_trained([r.image_id for r in new], mv.version)
        msg = {
            "production": (f"Model {mv.version} đã lên production ngay (chính sách always; cổng chất lượng: {gate['reason']})."
                           if policy == "always" else
                           f"Model {mv.version} đạt cổng chất lượng ({gate['reason']}) -> đã chuyển thành production."),
            "rejected": f"Model {mv.version} KHÔNG đạt cổng chất lượng ({gate['reason']}) -> giữ model cũ.",
            "candidate": f"Model {mv.version} sẵn sàng nhưng cần admin duyệt ({gate['reason']}).",
        }[status if not promote else "production"]
        self._ev("RETRAIN_DONE", "info" if status != "rejected" else "warn", msg, version=mv.version, gate=gate)
        return {"status": "done", "version": mv.version, "promoted": promote, "model_status": status if not promote else "production",
                "gate": gate, "n_new": len(new), "n_total": n_human, "n_replay": info.n_replay_train,
                "duration_s": round(dur, 1), "pca": _slim(sel)}


def _slim(sel: dict | None) -> dict | None:
    return None if sel is None else {k: v for k, v in sel.items() if k not in ("picked_ids", "dropped_ids")}
