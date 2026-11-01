import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ares.active_learner import ActiveLearner
from ares.db_manager import ActiveLearningItem
from ares.model_registry import BASE, ModelRegistry
from ares.retrain import Retrainer, build_dataset, evaluate_gate, group_key, in_val_group

W, H = 160, 120


# ------------------------------------------------------------------------------ registry
@pytest.fixture
def reg(tmp_path):
    base = tmp_path / "base.pt"
    base.write_bytes(b"base")
    return ModelRegistry(tmp_path / "models", base)


def weights(tmp_path, name="w.pt", data=b"x"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_registry_starts_on_base(reg):
    assert reg.production()[0] == BASE and reg.next_version() == "v0001" and reg.previous_production() is None


def test_rollback_walks_back_through_history(reg, tmp_path):
    for i in (1, 2):
        reg.add(weights(tmp_path, f"w{i}.pt", b"%d" % i), BASE, "x", 1, 1, 0, {}, {})
        reg.promote(f"v000{i}")
    assert reg.previous_production() == "v0001"
    assert reg.rollback() == "v0001" and reg.production()[0] == "v0001"
    assert reg.rollback() == BASE and reg.production()[0] == BASE
    assert reg.rollback() is None and reg.previous_production() is None
    reg.promote("v0002")  # đi tiếp sau khi lùi
    assert reg.previous_production() == BASE


def test_rollback_skips_versions_whose_file_vanished(reg, tmp_path):
    for i in (1, 2, 3):
        reg.add(weights(tmp_path, f"w{i}.pt", b"%d" % i), BASE, "x", 1, 1, 0, {}, {})
        reg.promote(f"v000{i}")
    Path(reg.get("v0002")["path"]).unlink()
    assert reg.rollback() == "v0001"


def test_add_promote_rollback(reg, tmp_path):
    v1 = reg.add(weights(tmp_path, data=b"1"), BASE, "midnight", 5, 10, 20, {}, {})
    assert v1.version == "v0001" and reg.production()[0] == BASE  # thêm ≠ promote
    reg.promote("v0001")
    assert reg.production()[0] == "v0001" and reg.production()[1].read_bytes() == b"1"
    v2 = reg.add(weights(tmp_path, "w2.pt", b"2"), "v0001", "midnight", 3, 13, 20, {}, {})
    reg.promote(v2.version)
    assert reg.previous_production() == "v0001"
    reg.promote("v0001")
    assert reg.production()[0] == "v0001"
    assert {v["version"]: v["status"] for v in reg.versions()}["v0002"] == "archived"


def test_production_falls_back_to_base_if_file_missing(reg, tmp_path):
    reg.add(weights(tmp_path), BASE, "x", 1, 1, 0, {}, {})
    reg.promote("v0001")
    Path(reg.production()[1]).unlink()
    assert reg.production()[0] == BASE


def test_promote_unknown_version_raises(reg):
    with pytest.raises(KeyError):
        reg.promote("v9999")


def test_registry_shared_across_processes_via_file(reg, tmp_path):
    other = ModelRegistry(reg.dir, reg.base_weights)
    reg.add(weights(tmp_path), BASE, "x", 1, 1, 0, {}, {})
    reg.promote("v0001")
    assert other.production()[0] == "v0001"  # dashboard (tiến trình khác) thấy ngay


# ------------------------------------------------------------------------------ cổng chất lượng
def m(rec, ap):
    return {"recall": rec, "map50": ap, "precision": 0.9}


def test_gate_passes_when_not_worse():
    g = evaluate_gate({"candidate": {"human": m(0.90, 0.80)}, "current": {"human": m(0.88, 0.80)}}, 0.01)
    assert g["passed"] is True


def test_gate_fails_if_worse_on_any_set():
    g = evaluate_gate({"candidate": {"human": m(0.95, 0.9), "replay": m(0.70, 0.9)},
                       "current": {"human": m(0.90, 0.8), "replay": m(0.90, 0.9)}}, 0.01)
    assert g["passed"] is False and "replay" in g["reason"]


def test_gate_tolerance_boundary():
    cur, cand = {"human": m(0.90, 0.80)}, {"human": m(0.89, 0.80)}
    assert evaluate_gate({"candidate": cand, "current": cur}, 0.01)["passed"] is True
    assert evaluate_gate({"candidate": {"human": m(0.88, 0.80)}, "current": cur}, 0.01)["passed"] is False


def test_gate_without_holdout_is_undecided():
    assert evaluate_gate({"candidate": {}, "current": {}}, 0.01)["passed"] is None


# ------------------------------------------------------------------------------ dataset
def make_approved(cfg, db, when, idx, status="APPROVED"):
    iid = f"P101_{when:%Y%m%d_%H%M%S}_{idx:02d}"
    jpg = cfg.approved_dir / f"{iid}.jpg"
    jpg.write_bytes(b"\xff\xd8fake")
    jpg.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.4\n")
    with db.session() as s:
        s.add(ActiveLearningItem(image_id=iid, captured_at=when, img_path=jpg.relative_to(cfg.storage_root).as_posix(),
                                 pred_bboxes=[], final_bboxes=[], status=status, room_id=cfg.room_id, width=W, height=H))
    return iid


def days_with_val(frac, want_val, n):
    out, d = [], datetime(2026, 10, 1, 9, 0)
    while len(out) < n:
        if in_val_group(group_key(d), frac) == want_val:
            out.append(d)
        d += timedelta(days=1)
    return out


def rows(al):
    return al.items_for_training()


def test_group_split_keeps_same_block_together(cfg):
    a, b = datetime(2026, 11, 2, 9, 5), datetime(2026, 11, 2, 11, 55)  # cùng khối 9-12h
    assert group_key(a) == group_key(b) and group_key(a) != group_key(datetime(2026, 11, 2, 12, 0))


def test_holdout_by_time_group_has_no_leakage(cfg, db, al, clock):
    for i, d in enumerate(days_with_val(cfg.val_fraction, True, 6)):
        make_approved(cfg, db, d, i)
    for i, d in enumerate(days_with_val(cfg.val_fraction, False, 6)):
        make_approved(cfg, db, d, 10 + i)
    info = build_dataset(cfg, rows(al), cfg.datasets_dir / "r1")
    assert info.has_holdout and info.n_human_val == 6 and info.n_human_train == 6
    tr = set((cfg.datasets_dir / "r1" / "train.txt").read_text().split())
    va = set((cfg.datasets_dir / "r1" / "val_human.txt").read_text().split())
    assert not (tr & va)
    assert all("/labels/" not in p for p in tr)  # đường dẫn ảnh, và nhãn suy ra bằng thay /images/ -> /labels/
    for p in tr | va:
        lab = Path(p.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        assert lab.exists()


def test_too_few_holdout_images_are_folded_into_train(cfg, db, al):
    for i, d in enumerate(days_with_val(cfg.val_fraction, True, 2)):
        make_approved(cfg, db, d, i)
    for i, d in enumerate(days_with_val(cfg.val_fraction, False, 6)):
        make_approved(cfg, db, d, 10 + i)
    info = build_dataset(cfg, rows(al), cfg.datasets_dir / "r2")
    assert not info.has_holdout and info.n_human_train == 8 and any("giữ lại" in w for w in info.warnings)


def test_replay_ratio_70_30(mk_cfg, db, tmp_path):
    replay = tmp_path / "replay"
    for i in range(300):
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            (replay / sub).mkdir(parents=True, exist_ok=True)
            (replay / sub / f"r{i}{ext}").write_text("x")
    cfg = mk_cfg(replay_dir=replay, replay_old_ratio=0.7)
    from ares.clock import ManualClock
    al = ActiveLearner(cfg, ManualClock(datetime(2026, 11, 2)), db)
    for i, d in enumerate(days_with_val(cfg.val_fraction, False, 30)):
        make_approved(cfg, db, d, i)
    info = build_dataset(cfg, rows(al), cfg.datasets_dir / "r3")
    frac_old = info.n_replay_train / (info.n_replay_train + info.n_human_train)
    assert info.n_human_train == 30 and frac_old == pytest.approx(0.7, abs=0.03)
    assert info.val_replay_list is not None  # tập replay giữ lại cố định để so sánh giữa các đêm


# ------------------------------------------------------------------------------ Retrainer
def fake_trainer(tmp_path, cand, cur, ok=True):
    calls = []

    def run(job):
        calls.append(job)
        if not ok:
            return {"ok": False, "error": "CUDA hết bộ nhớ"}
        w = tmp_path / f"cand_{len(calls)}.pt"
        w.write_bytes(b"cand")
        return {"ok": True, "weights": str(w), "metrics": {"candidate": cand, "current": cur}}
    run.calls = calls
    return run


def seed_items(cfg, db, n_val=6, n_train=6):
    for i, d in enumerate(days_with_val(cfg.val_fraction, True, n_val)):
        make_approved(cfg, db, d, i)
    for i, d in enumerate(days_with_val(cfg.val_fraction, False, n_train)):
        make_approved(cfg, db, d, 10 + i)


GOOD = ({"human": m(0.92, 0.85)}, {"human": m(0.90, 0.84)})
BAD = ({"human": m(0.70, 0.60)}, {"human": m(0.90, 0.84)})


def retrainer(cfg, clock, al, reg, trainer, events):
    return Retrainer(cfg, clock, al, reg, trainer=trainer, emit=events)


def test_skips_when_nothing_approved(cfg, clock, al, reg, events, tmp_path):
    r = retrainer(cfg, clock, al, reg, fake_trainer(tmp_path, *GOOD), events).run()
    assert r["status"] == "skipped"


def test_good_candidate_is_promoted_and_items_marked_trained(cfg, clock, db, al, reg, events, tmp_path):
    seed_items(cfg, db)
    tr = fake_trainer(tmp_path, *GOOD)
    r = retrainer(cfg, clock, al, reg, tr, events).run()
    assert r["status"] == "done" and r["promoted"] and reg.production()[0] == "v0001"
    job = tr.calls[0]
    assert job["base_weights"].endswith(cfg.weights) or Path(job["base_weights"]).name == Path(cfg.weights).name  # luôn từ checkpoint GỐC
    assert al.counts()["TRAINED"] == 12 and al.counts()["APPROVED"] == 0
    assert "RETRAIN_DONE" in events.kinds()


def test_worse_candidate_is_rejected_and_production_unchanged(cfg, clock, db, al, reg, events, tmp_path):
    cfg.auto_promote = "gated"
    seed_items(cfg, db)
    r = retrainer(cfg, clock, al, reg, fake_trainer(tmp_path, *BAD), events).run()
    assert r["status"] == "done" and not r["promoted"] and r["model_status"] == "rejected"
    assert reg.production()[0] == BASE
    assert {v["version"]: v["status"] for v in reg.versions()}["v0001"] == "rejected"


def test_policy_never_keeps_candidate_for_manual_approval(mk_cfg, clock, db, reg, events, tmp_path):
    cfg = mk_cfg(auto_promote="never", al_min_gap_s=0.0)
    al = ActiveLearner(cfg, clock, db)
    seed_items(cfg, db)
    retrainer(cfg, clock, al, reg, fake_trainer(tmp_path, *GOOD), events).run()
    assert reg.production()[0] == BASE and {v["version"]: v["status"] for v in reg.versions()}["v0001"] == "candidate"


def test_default_policy_always_swaps_even_worse_model_but_warns(cfg, clock, db, al, reg, events, tmp_path):
    assert cfg.auto_promote == "always"  # workflow: train xong hoán đổi ngay
    seed_items(cfg, db)
    r = retrainer(cfg, clock, al, reg, fake_trainer(tmp_path, *BAD), events).run()
    assert r["status"] == "done" and r["promoted"] and reg.production()[0] == "v0001"
    assert r["gate"]["passed"] is False and "MODEL_GATE_WARN" in events.kinds()  # vẫn đo + cảnh báo để rollback
    assert reg.rollback() == BASE and reg.production()[0] == BASE  # quay lui bằng tay vẫn được


def test_default_policy_always_swaps_without_holdout(cfg, clock, db, al, reg, events, tmp_path):
    for i, d in enumerate(days_with_val(cfg.val_fraction, False, 4)):
        make_approved(cfg, db, d, i)
    r = retrainer(cfg, clock, al, reg, fake_trainer(tmp_path, {}, {}), events).run()
    assert r["status"] == "done" and r["promoted"] and reg.production()[0] == "v0001"
    assert "MODEL_GATE_WARN" not in events.kinds()  # không có tập để so thì không kết luận "kém hơn"


def test_no_holdout_means_candidate_never_auto_promoted(cfg, clock, db, al, reg, events, tmp_path):
    cfg.auto_promote = "gated"
    for i, d in enumerate(days_with_val(cfg.val_fraction, False, 4)):
        make_approved(cfg, db, d, i)
    r = retrainer(cfg, clock, al, reg, fake_trainer(tmp_path, {}, {}), events).run()
    assert r["status"] == "done" and not r["promoted"] and reg.production()[0] == BASE


def test_trainer_failure_changes_nothing(cfg, clock, db, al, reg, events, tmp_path):
    seed_items(cfg, db)
    r = retrainer(cfg, clock, al, reg, fake_trainer(tmp_path, *GOOD, ok=False), events).run()
    assert r["status"] == "failed" and reg.production()[0] == BASE and al.counts()["APPROVED"] == 12
    assert "RETRAIN_FAILED" in events.kinds()


def test_force_retrains_even_without_new_images(cfg, clock, db, al, reg, events, tmp_path):
    seed_items(cfg, db)
    rt = retrainer(cfg, clock, al, reg, fake_trainer(tmp_path, *BAD), events)
    rt.run()
    assert rt.run()["status"] == "skipped"  # đêm sau: không có ảnh mới
    assert rt.run(force=True)["status"] == "done"  # admin bấm "Train ngay"


def test_concurrent_runs_are_refused(cfg, clock, al, reg, events, tmp_path):
    rt = retrainer(cfg, clock, al, reg, fake_trainer(tmp_path, *GOOD), events)
    rt._lock.acquire()
    assert rt.run()["status"] == "busy"
