"""Một xe bị model COCO cho 2 hộp khác lớp (car + truck) phải được gộp còn 1 — tránh lập 2 hồ sơ cho 1 xe."""

import numpy as np

from traffic_safety_ai.core.detector import ModelOutput, merge_cross_class
from traffic_safety_ai.core.types import Detection


def _d(box, conf, cls):
    return Detection(np.array(box, float), conf, cls)


def test_car_truck_duplicate_merged_keeps_higher_conf():
    out = ModelOutput(boxes=[_d([10, 10, 100, 60], 0.53, "car"), _d([11, 10, 101, 61], 0.64, "truck")])
    merge_cross_class(out)
    assert [b.cls for b in out.boxes] == ["truck"]


def test_different_vehicles_and_person_not_merged():
    out = ModelOutput(boxes=[_d([10, 10, 100, 60], 0.6, "car"), _d([200, 10, 290, 60], 0.6, "truck"),
                             _d([20, 0, 40, 50], 0.7, "person"), _d([30, 20, 60, 70], 0.5, "motorcycle")])
    merge_cross_class(out)
    assert len(out.boxes) == 4


def test_same_class_overlap_left_to_nms():
    out = ModelOutput(boxes=[_d([10, 10, 100, 60], 0.6, "car"), _d([12, 10, 100, 60], 0.5, "car")])
    merge_cross_class(out)
    assert len(out.boxes) == 2
