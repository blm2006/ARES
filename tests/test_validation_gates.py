"""Gate validation: per-camera, golden, và ablation với baseline random."""

from __future__ import annotations

from common.schemas import BBox
from training.validate import ValidationReport, check_gates, evaluate_set


def b(x, conf=1.0):
    return BBox(x1=x, y1=0, x2=x + 10, y2=20, conf=conf)


def test_recall_counts_missed_people():
    gts = {"a": [b(0), b(50)], "b": [b(0)]}
    preds = {"a": [b(0)], "b": []}  # bỏ sót 2 người
    r = evaluate_set(preds, gts)
    assert r.tp == 1 and r.fn == 2
    assert abs(r.recall50 - 1 / 3) < 1e-6


def test_per_camera_gate_catches_regression_hidden_by_average(settings):
    """mAP tổng tăng nhưng một camera tụt 5% -> phải trượt."""
    prev = {"per_camera": {"cam01": {"recall50": 0.90}, "cam02": {"recall50": 0.80}}}
    new = ValidationReport(
        golden={"recall50": 0.88},
        rolling={"recall50": 0.90},
        per_camera={"cam01": {"recall50": 0.95}, "cam02": {"recall50": 0.75}},
    )
    out = check_gates(new, prev, settings)
    assert not out.gates["per_camera"]["ok"]
    assert out.gates["per_camera"]["worst_camera"] == "cam02"
    assert not out.passed


def test_golden_gate_blocks_drop(settings):
    prev = {"golden": {"recall50": 0.90}, "per_camera": {}}
    new = ValidationReport(golden={"recall50": 0.85}, rolling={"recall50": 0.95}, per_camera={})
    out = check_gates(new, prev, settings)
    assert not out.gates["golden"]["ok"]


def test_ablation_gate_flags_useless_active_learning(settings):
    """Nếu không thắng nổi random thì nguồn A đang là chi phí vô ích."""
    new = ValidationReport(
        golden={"recall50": 0.90},
        rolling={"recall50": 0.86},
        per_camera={},
        baseline_random={"recall50": 0.88},
    )
    out = check_gates(new, None, settings)
    g = out.gates["beats_random_baseline"]
    assert not g["ok"]
    assert "KHÔNG thắng random" in g["note"]
    assert not out.passed


def test_all_gates_pass(settings):
    prev = {"golden": {"recall50": 0.85}, "per_camera": {"cam01": {"recall50": 0.85}}}
    new = ValidationReport(
        golden={"recall50": 0.88},
        rolling={"recall50": 0.91},
        per_camera={"cam01": {"recall50": 0.87}},
        baseline_random={"recall50": 0.84},
    )
    out = check_gates(new, prev, settings)
    assert out.passed, out.gates


def test_primary_metric_is_recall_by_default(settings):
    assert settings.val_primary_metric == "recall50"
