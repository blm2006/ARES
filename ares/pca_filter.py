"""Lọc ảnh bằng PCA — hai giai đoạn, thay cho trần "50 ảnh/ngày".

    camera --(band conf 0.35-0.55)--> CANDIDATE
        │
        ▼  GIAI ĐOẠN 1 — trước khi người duyệt
        │  vector đặc trưng -> PCA (đóng băng) -> bỏ ảnh quá giống ảnh ĐÃ CÓ (đang chờ duyệt, đã duyệt, đã loại, đã train)
        │  và quá giống nhau trong cùng mẻ. Giữ: PENDING (người duyệt). Bỏ: DUPLICATE (giữ vài ngày, khôi phục được).
        ▼
    người duyệt: Approve / Edit BBox / Reject  ->  APPROVED
        │
        ▼  GIAI ĐOẠN 2 — trước khi train (chạy ngay trước retrain ban đêm)
        │  PCA lần nữa trên ảnh đã duyệt: luôn giữ ảnh người đã SỬA bbox (mẫu khó), phần còn lại chọn tập đại diện
        │  (xa nhất trước) và bỏ ảnh thừa so với tập đã train / đã chọn.  train_pick = True | False
        ▼
    train chỉ trên ảnh train_pick != False

Tái sử dụng từ hệ thống cũ (`worker/`): `HistogramEmbedder` (vector thô 512 chiều, chỉ cần OpenCV), `FrozenPCA`/`fit_pca`
(PCA fit MỘT lần rồi đóng băng, có version, không ghi đè file) và ý tưởng k-center (chọn điểm xa nhất).

Vì sao vector thô được lưu cùng ảnh: nếu sau này đổi version PCA thì chiếu lại từ vector thô, không phải giải mã lại ảnh.

Ngưỡng không đặt cứng mà TỰ HIỆU CHUẨN lúc fit: đo khoảng cách tới láng giềng gần nhất giữa các khung hình mẫu
(cách nhau ~30 giây, tức "mức khác nhau thường thấy"), rồi lấy phân vị `pca_q_review` (lọc trước duyệt) và
`pca_q_train` (lọc trước train) làm bán kính. Ảnh gần ảnh đã có hơn bán kính = ảnh thừa.

Giới hạn cần biết: vector thô mô tả BỐ CỤC/MÀU/CẠNH của cả khung hình, không hiểu "có người hay không". Hai khung giống
hệt nhau chỉ khác một người đứng ở góc có thể bị coi là trùng. Vì vậy ảnh người đã sửa bbox luôn được giữ.
Ảnh bị lọc ở giai đoạn 1: `duplicate_ttl_days = 0` (mặc định theo workflow) thì XOÁ NGAY cả file lẫn dòng CSDL —
không khôi phục được; đặt > 0 thì giữ chừng đó ngày (khôi phục được trên dashboard) rồi mới xoá.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
from sqlalchemy import func, select

from ares.config import AresSettings
from ares.db_manager import ActiveLearningItem, DBManager
from ares.state import Event

log = logging.getLogger(__name__)

EMBEDDER_NAME = "hist_v1"
_EMB: object | None = None
_EMB_LOCK = threading.Lock()


def _embedder():
    global _EMB
    with _EMB_LOCK:
        if _EMB is None:
            from worker.embedding import HistogramEmbedder  # hệ thống cũ; chỉ cần cv2 + numpy

            _EMB = HistogramEmbedder()
        return _EMB


def embed_image(image: np.ndarray) -> np.ndarray:
    """Vector thô 512 chiều (đã chuẩn hoá L2), float32."""
    return _embedder().embed([image])[0]


def emb_to_bytes(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float16).tobytes()


def emb_from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)


def _read_image(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    return img


# ---------------------------------------------------------------------------
# Thuật toán chọn tập đại diện
# ---------------------------------------------------------------------------
def select_diverse(z: np.ndarray, seeds: np.ndarray | None, radius: float, min_keep: int = 0,
                   forced: list[int] | None = None) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Chọn tập con đại diện của `z` (các hàng đã chuẩn hoá L2, khoảng cách = 1 - cosine).

    Lặp: chọn điểm xa nhất so với (seeds ∪ điểm đã chọn); dừng khi điểm xa nhất cách tập hiện có < `radius`
    VÀ đã đủ `min_keep`. `forced` luôn được chọn trước (vd ảnh người đã sửa bbox).

    Trả (chỉ số đã chọn theo thứ tự chọn, khoảng cách tới điểm gần nhất, tham chiếu của điểm gần nhất:
    >= 0 là chỉ số trong z, < 0 là seed thứ -(r+1)).
    """
    n = len(z)
    dist = np.full(n, np.inf, dtype=np.float32)
    ref = np.zeros(n, dtype=np.int64)
    if n == 0:
        return [], dist, ref
    if seeds is not None and len(seeds):
        sims = z @ seeds.T
        j = sims.argmax(axis=1)
        dist = (1.0 - sims[np.arange(n), j]).astype(np.float32)
        ref = -(j + 1)
    picked: list[int] = []
    chosen = np.zeros(n, dtype=bool)

    def add(i: int) -> None:
        picked.append(i)
        chosen[i] = True
        d = (1.0 - z @ z[i]).astype(np.float32)
        better = d < dist
        better[i] = False
        dist[better] = d[better]
        ref[better] = i

    for i in forced or []:
        if not chosen[i]:
            add(i)
    while len(picked) < n:
        masked = np.where(chosen, -1.0, dist)
        i = int(masked.argmax())
        if masked[i] < radius and len(picked) >= min_keep:
            break
        add(i)
    return picked, dist, ref


# ---------------------------------------------------------------------------
class PcaFilter:
    def __init__(self, cfg: AresSettings, clock, db: DBManager, emit: Callable[[Event], None] | None = None):
        self.cfg, self.clock, self.db = cfg, clock, db
        self.emit = emit if emit is not None else (lambda e: None)
        self._lock = threading.RLock()
        self.dir = cfg.pca_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pca = None  # FrozenPCA | None
        self.thr: dict[str, float] | None = None
        self.corpus = np.zeros((0, _embedder().dim), dtype=np.float16)
        self._dirty = 0
        self._last_corpus: dict[str, float] = {}
        self._last_tick = -math.inf
        self.last_stage1: dict | None = None
        self.last_stage2: dict | None = None
        self.deleted_total = 0  # ảnh trùng đã xoá ngay (duplicate_ttl_days = 0) kể từ lúc khởi động
        self._load()

    # ------------------------------------------------------------------ lưu / nạp
    @property
    def _pkl(self) -> Path:
        return self.dir / f"{self.cfg.pca_version}.pkl"

    @property
    def _meta(self) -> Path:
        return self.dir / f"{self.cfg.pca_version}.thr.json"

    @property
    def _corpus_file(self) -> Path:
        return self.dir / "corpus.npy"

    def _load(self) -> None:
        if self._corpus_file.exists():
            try:
                c = np.load(self._corpus_file)
                if c.ndim == 2 and c.shape[1] == self.corpus.shape[1]:
                    self.corpus = c.astype(np.float16)
            except Exception:
                log.warning("Không đọc được %s, bắt đầu lại tập mẫu PCA", self._corpus_file, exc_info=True)
        if self._pkl.exists():
            from worker.pca import FrozenPCA

            try:
                pca = FrozenPCA.load(self._pkl)
                if pca.version != self.cfg.pca_version or pca.embedder != EMBEDDER_NAME:
                    raise ValueError(f"{pca.version}/{pca.embedder} không khớp cấu hình")
                self.pca = pca
                self.thr = json.loads(self._meta.read_text(encoding="utf-8")) if self._meta.exists() else None
                if self.thr is None:
                    self.thr = self._calibrate(pca, self.corpus.astype(np.float32))
                    self._meta.write_text(json.dumps(self.thr), encoding="utf-8")
                log.info("PCA %s đã nạp (%d chiều, %d mẫu fit)", pca.version, pca.n_components, pca.n_fit_samples)
            except Exception:
                log.exception("Không nạp được PCA %s — sẽ coi như chưa khởi tạo", self._pkl)
                self.pca = None

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            tmp = self._corpus_file.with_suffix(".tmp.npy")
            np.save(tmp, self.corpus)
            tmp.replace(self._corpus_file)
            self._dirty = 0

    def close(self) -> None:
        try:
            self.flush()
        except Exception:
            log.exception("Không lưu được tập mẫu PCA")

    # ------------------------------------------------------------------ tập mẫu (trước khi fit)
    def offer_corpus(self, cam_id: str, image: np.ndarray) -> bool:
        """Gọi từ luồng vision/hub. Mỗi camera tối đa 1 khung / pca_corpus_interval_s; ~vài ms. Ngừng thu khi PCA đã đóng băng."""
        if not self.cfg.pca_enabled or self.pca is not None:
            return False
        t = self.clock.time()
        if t - self._last_corpus.get(cam_id, -math.inf) < self.cfg.pca_corpus_interval_s:
            return False
        self._last_corpus[cam_id] = t
        v = embed_image(image).astype(np.float16)
        with self._lock:
            self.corpus = np.vstack([self.corpus, v[None]])[-self.cfg.pca_corpus_max:]
            self._dirty += 1
            if self._dirty >= 25:
                self.flush()
        return True

    # ------------------------------------------------------------------ fit (một lần)
    def need(self, manual: bool = False) -> int:
        return self.cfg.pca_fit_min_manual if manual else self.cfg.pca_fit_min

    def fit(self, manual: bool = False) -> dict | None:
        """Fit PCA một lần rồi đóng băng. None nếu chưa đủ mẫu hoặc đã có PCA."""
        from worker.pca import fit_pca

        with self._lock:
            if self.pca is not None:
                return None
            n = len(self.corpus)
            need = self.need(manual)
            if n < need:
                return None
            vecs = self.corpus.astype(np.float32)
            pca = fit_pca(vecs, self.cfg.pca_version, EMBEDDER_NAME, n_components=max(2, min(self.cfg.pca_components, n - 1)),
                          min_samples=need)
            pca.save(self._pkl)  # từ chối ghi đè: đã đóng băng
            self.thr = self._calibrate(pca, vecs)
            self._meta.write_text(json.dumps(self.thr), encoding="utf-8")
            self.pca = pca
            self.flush()
        msg = (f"Đã khởi tạo PCA {pca.version} từ {n} khung hình mẫu ({pca.n_components} chiều) và đóng băng. "
               f"Bán kính lọc: trước duyệt {self.thr['review']:.3f}, trước train {self.thr['train']:.3f}.")
        self.emit(Event(self.clock.time(), "PCA_FITTED", "info", msg, {"n": n, "thr": self.thr}))
        return {"version": pca.version, "n": n, "thr": self.thr}

    def _calibrate(self, pca, vecs: np.ndarray) -> dict[str, float]:
        z = pca.transform(vecs)
        n = len(z)
        nn = np.empty(n, dtype=np.float32)
        for a in range(0, n, 500):  # theo mẻ để không dựng ma trận n x n
            s = z[a:a + 500] @ z.T
            for r in range(s.shape[0]):
                s[r, a + r] = -np.inf
            nn[a:a + 500] = 1.0 - s.max(axis=1)
        nn = np.clip(nn, 0.0, 2.0)
        pos = nn[nn > 1e-4]  # khung trùng khít (phòng trống) kéo phân vị về 0 -> chỉ tính các cặp thật sự khác nhau
        ref = pos if len(pos) >= 10 else nn
        return {"review": max(float(np.quantile(ref, self.cfg.pca_q_review)), self.cfg.pca_min_radius_review),
                "train": max(float(np.quantile(ref, self.cfg.pca_q_train)), self.cfg.pca_min_radius_train),
                "nn_median": float(np.median(nn)), "n": int(n)}

    # ------------------------------------------------------------------ tiện ích vector
    def _vec(self, it: ActiveLearningItem) -> np.ndarray | None:
        if it.embedding:
            v = emb_from_bytes(it.embedding)
            if v.shape[0] == self.corpus.shape[1]:
                return v
        try:
            v = embed_image(_read_image(self.cfg.storage_root / it.img_path))
        except Exception:
            log.warning("Không tính được vector cho %s", it.image_id, exc_info=True)
            return None
        it.embedding = emb_to_bytes(v)
        return v

    def _project(self, vs: list[np.ndarray]) -> np.ndarray:
        if not vs:
            return np.zeros((0, self.pca.n_components), dtype=np.float32)
        return self.pca.transform(np.stack(vs))

    # ------------------------------------------------------------------ GIAI ĐOẠN 1
    def triage(self, force: bool = False) -> dict:
        """CANDIDATE -> PENDING (giữ) | DUPLICATE (bỏ). Không làm gì nếu PCA chưa sẵn sàng."""
        cfg = self.cfg
        if self.pca is None:
            return {"status": "waiting_pca", "have": len(self.corpus), "need": cfg.pca_fit_min}
        now = self.clock.naive_now()
        delete_now = cfg.duplicate_ttl_days <= 0
        to_unlink: list[Path] = []
        with self._lock, self.db.session() as s:
            q = select(ActiveLearningItem).where(ActiveLearningItem.room_id == cfg.room_id,
                                                 ActiveLearningItem.status == "CANDIDATE").order_by(ActiveLearningItem.captured_at)
            cands = list(s.execute(q).scalars().all())
            if not force:
                cands = [c for c in cands if (now - c.captured_at).total_seconds() >= cfg.pca_triage_delay_s]
            if not cands:
                return {"status": "idle", "n": 0}
            seeds_rows = list(s.execute(
                select(ActiveLearningItem).where(
                    ActiveLearningItem.room_id == cfg.room_id,
                    ActiveLearningItem.status.in_(("PENDING", "APPROVED", "TRAINED", "REJECTED")))
                .order_by(ActiveLearningItem.captured_at.desc()).limit(4000)).scalars().all())
            cv = [self._vec(c) for c in cands]
            good = [(c, v) for c, v in zip(cands, cv) if v is not None]
            sv = [(r, self._vec(r)) for r in seeds_rows]
            sv = [(r, v) for r, v in sv if v is not None]
            for c, v in zip(cands, cv):
                if v is None:  # ảnh hỏng/mất file: không đưa người duyệt
                    c.status = "DUPLICATE"
            if not good:
                return {"status": "idle", "n": 0}
            z = self._project([v for _, v in good])
            zs = self._project([v for _, v in sv])
            picked, dist, ref = select_diverse(z, zs, self.thr["review"])
            keep = set(picked)
            n_keep = n_drop = 0
            for i, (c, _) in enumerate(good):
                if i in keep:
                    c.status = "PENDING"
                    n_keep += 1
                else:
                    r = int(ref[i])
                    c.status = "DUPLICATE"
                    c.dup_of = good[r][0].image_id if r >= 0 else sv[-r - 1][0].image_id
                    c.dup_dist = float(dist[i])
                    n_drop += 1
            if delete_now:
                for c in cands:
                    if c.status == "DUPLICATE":
                        to_unlink += [cfg.storage_root / Path(c.img_path).with_suffix(suf) for suf in (".jpg", ".txt")]
                        s.delete(c)
        # xoá file SAU khi CSDL đã ghi xong (lỗi CSDL thì file vẫn còn, không có dòng trỏ vào file đã mất)
        for f in to_unlink:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                log.warning("Không xoá được %s", f)
        if delete_now:
            self.deleted_total += n_drop
        res = {"status": "done", "n": n_keep + n_drop, "kept": n_keep, "dropped": n_drop,
               "reduction_pct": round(100.0 * n_drop / max(1, n_keep + n_drop), 1), "radius": round(self.thr["review"], 4),
               "at": self.clock.now().isoformat(timespec="seconds")}
        self.last_stage1 = res
        self.emit(Event(self.clock.time(), "PCA_TRIAGE", "info",
                        f"Lọc PCA trước khi duyệt: {res['n']} ảnh -> giữ {n_keep} cho người duyệt, "
                        f"{'xoá' if delete_now else 'loại'} {n_drop} ảnh trùng "
                        f"(giảm {res['reduction_pct']}%).", res))
        return res

    def release_all(self) -> int:
        """Admin chấp nhận bỏ qua lọc: CANDIDATE -> PENDING hết."""
        with self.db.session() as s:
            rows = s.execute(select(ActiveLearningItem).where(
                ActiveLearningItem.room_id == self.cfg.room_id, ActiveLearningItem.status == "CANDIDATE")).scalars().all()
            for r in rows:
                r.status = "PENDING"
            return len(rows)

    # ------------------------------------------------------------------ GIAI ĐOẠN 2
    def select_for_training(self, rows: list[ActiveLearningItem], commit: bool = True) -> dict:
        """Chọn ảnh đã duyệt nào thật sự đưa vào train. `rows` = APPROVED + TRAINED (như ActiveLearner.items_for_training).

        Với commit=True ghi train_pick lên DB. Trả báo cáo; `picked_ids` / `dropped_ids` là các APPROVED mới.
        """
        cfg = self.cfg
        new = [r for r in rows if r.status == "APPROVED"]
        trained = [r for r in rows if r.status == "TRAINED"]
        rep: dict = {"n_new": len(new), "n_trained_before": len(trained), "pca_version": cfg.pca_version if self.pca else None}
        picked_ids = [r.image_id for r in new]
        if not cfg.pca_enabled:
            rep.update(note="Lọc PCA đang tắt — train trên toàn bộ ảnh mới.")
        elif self.pca is None:
            rep.update(note="PCA chưa khởi tạo — train trên toàn bộ ảnh mới.")
        elif len(new) < cfg.pca_train_skip_below:
            rep.update(note=f"Chỉ {len(new)} ảnh mới (< {cfg.pca_train_skip_below}) — giữ hết, không lọc.")
        else:
            with self._lock:
                vn = [self._vec(r) for r in new]
                vt = [self._vec(r) for r in trained]
                ok = [(r, v) for r, v in zip(new, vn) if v is not None]
                zt = self._project([v for v in vt if v is not None])
                z = self._project([v for _, v in ok])
                forced = [i for i, (r, _) in enumerate(ok) if r.edited]
                min_keep = max(len(forced), math.ceil(cfg.pca_train_min_keep * len(ok)))
                pick_idx, _, _ = select_diverse(z, zt, self.thr["train"], min_keep=min_keep, forced=forced)
            keep = {ok[i][0].image_id for i in pick_idx}
            picked_ids = [r.image_id for r, _ in ok if r.image_id in keep]
            rep.update(forced_edited=len(forced), radius=round(self.thr["train"], 4))
        dropped_ids = [r.image_id for r in new if r.image_id not in set(picked_ids)]
        rep.update(n_picked=len(picked_ids), n_dropped=len(dropped_ids),
                   reduction_pct=round(100.0 * len(dropped_ids) / max(1, len(new)), 1),
                   picked_ids=picked_ids, dropped_ids=dropped_ids)
        if commit:
            self._commit_picks(picked_ids, dropped_ids)
            slim = {k: v for k, v in rep.items() if k not in ("picked_ids", "dropped_ids")}
            slim["at"] = self.clock.now().isoformat(timespec="seconds")
            self.last_stage2 = slim
            if new:
                self.emit(Event(self.clock.time(), "PCA_TRAIN_SELECT", "info",
                                f"Lọc PCA trước khi train: {len(new)} ảnh đã duyệt -> đưa {len(picked_ids)} ảnh vào train, "
                                f"bỏ {len(dropped_ids)} ảnh thừa (giảm {rep['reduction_pct']}%).", slim))
        return rep

    def _commit_picks(self, picked: list[str], dropped: list[str]) -> None:
        with self.db.session() as s:
            for iid, val in [(i, True) for i in picked] + [(i, False) for i in dropped]:
                it = s.get(ActiveLearningItem, iid)
                if it is not None and it.status == "APPROVED":
                    it.train_pick = val

    # ------------------------------------------------------------------ nhịp định kỳ
    def tick(self) -> None:
        """Gọi từ luồng housekeeping (vài giây một lần): tự fit khi đủ mẫu, lọc mẻ ứng viên đã chờ đủ lâu."""
        if not self.cfg.pca_enabled:
            return
        t = self.clock.time()
        if t - self._last_tick < 20.0:
            return
        self._last_tick = t
        if self.pca is None:
            self.fit()
        if self.pca is not None:
            self.triage()

    # ------------------------------------------------------------------ cho dashboard
    def status(self) -> dict:
        cfg = self.cfg
        with self.db.session() as s:
            rows = dict(s.execute(select(ActiveLearningItem.status, func.count()).where(
                ActiveLearningItem.room_id == cfg.room_id).group_by(ActiveLearningItem.status)).all())
        kept = sum(int(rows.get(k, 0)) for k in ("PENDING", "APPROVED", "REJECTED", "TRAINED"))
        dup = int(rows.get("DUPLICATE", 0)) + self.deleted_total
        return {
            "enabled": cfg.pca_enabled, "fitted": self.pca is not None, "version": cfg.pca_version,
            "n_components": self.pca.n_components if self.pca else None,
            "corpus": int(len(self.corpus)), "need": cfg.pca_fit_min, "need_manual": cfg.pca_fit_min_manual,
            "thr": self.thr, "candidates": int(rows.get("CANDIDATE", 0)), "kept_total": kept, "duplicates_total": dup,
            "reduction_total_pct": round(100.0 * dup / max(1, kept + dup), 1),
            "last_stage1": self.last_stage1, "last_stage2": self.last_stage2,
            "duplicates_deleted_now": cfg.duplicate_ttl_days <= 0, "duplicates_deleted_since_start": self.deleted_total,
        }
