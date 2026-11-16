import cv2
import numpy as np
import pytest

from traffic_safety_ai.core.types import Detection
from traffic_safety_ai.lpr.fusion import PlateFusion, PlateRead
from traffic_safety_ai.lpr.plate_color import classify_plate_color
from traffic_safety_ai.lpr.reader import classic_plate_candidates, decode_char_boxes, plate_quality
from traffic_safety_ai.lpr.vn_plate import parse_plate, province_label


@pytest.mark.parametrize("raw,cls,display", [
    ("51F12345", "car", "51F-123.45"),
    ("51F-123.45", "car", "51F-123.45"),
    ("51F1234", "car", "51F-1234"),
    ("30LD12345", "car", "30LD-123.45"),
    ("59X112345", "motorcycle", "59-X1 123.45"),
    ("59 X1 123.45", "motorcycle", "59-X1 123.45"),
    ("29AB12345", "motorcycle", "29-AB 123.45"),
    ("29MD112345", "motorcycle", "29-MĐ1 123.45"),
])
def test_parse_and_format(raw, cls, display):
    p = parse_plate(raw, cls)
    assert p is not None and p.display == display and p.valid


def test_positional_correction_of_ocr_confusions():
    # O→0 ở vị trí số, 8→B ở vị trí chữ, S→5 ở vị trí số
    p = parse_plate("5O8-1S3.45", "car")
    assert p is not None and p.display == "50B-153.45" and p.corrections == 3
    p2 = parse_plate("59X1I2345", "motorcycle")
    assert p2 is not None and p2.display == "59-X1 123.45"


def test_invalid_province_flagged():
    p = parse_plate("13A12345", "car")
    assert p is not None and not p.valid and "13" in p.reasons[0]
    assert "Hưng Yên" in province_label("17") and "Thái Bình" in province_label("17")
    assert province_label("51") == "TP. Hồ Chí Minh"


def test_unknown_vehicle_uses_layout_hint():
    p1 = parse_plate("59X112345", None, "2line")
    assert p1.kind == "moto"
    p2 = parse_plate("51F12345", None, "1line")
    assert p2.kind == "car"


def test_fusion_votes_out_single_frame_errors():
    fu = PlateFusion("motorcycle")
    truth = "59X112345"
    noisy = ["59X112345", "59X112845", "59X112345", "S9X112345", "59X1I2345", "59X112345", "59X172345"]
    for i, s in enumerate(noisy):
        fu.add(PlateRead(s, 0.8, 0.7, i, i / 30))
    r = fu.result()
    assert r.text == truth and r.display == "59-X1 123.45" and r.valid
    assert r.confidence > 0.6 and r.n_reads == len(noisy)


def test_fusion_single_read_low_confidence():
    fu = PlateFusion("car")
    fu.add(PlateRead("51F12345", 0.95, 1.0, 0, 0.0))
    r = fu.result()
    assert r.text == "51F12345" and r.confidence < 0.5


def _plate(bg, fg):
    img = np.full((60, 200, 3), bg, np.uint8)
    cv2.putText(img, "51F-12345", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, fg, 3)
    return img


def test_plate_color():
    assert classify_plate_color(_plate((245, 245, 245), (10, 10, 10)))[0] == "white"
    assert classify_plate_color(_plate((0, 210, 240), (10, 10, 10)))[0] == "yellow"
    assert classify_plate_color(_plate((160, 60, 10), (250, 250, 250)))[0] == "blue"
    k, label = classify_plate_color(_plate((245, 245, 245), (10, 10, 10)))
    assert label.startswith("Nền màu trắng")


def test_decode_two_line_char_boxes():
    dets = []
    top = "59X1"
    bottom = "12345"
    for i, ch in enumerate(top):
        dets.append(Detection([20 + i * 30, 10, 45 + i * 30, 45], 0.9, ch))
    for i, ch in enumerate(bottom):
        dets.append(Detection([10 + i * 28, 60, 34 + i * 28, 95], 0.9, ch))
    dets.append(Detection([21, 11, 44, 44], 0.4, "S"))   # hộp trùng lặp khác lớp → bị NMS
    r = decode_char_boxes(dets)
    assert r.text == "59X112345" and r.layout == "2line"
    p = parse_plate(r.text, "motorcycle", r.layout)
    assert p.display == "59-X1 123.45"


def test_classic_detector_finds_plate_like_region():
    car = np.full((240, 320, 3), 90, np.uint8)
    plate = _plate((245, 245, 245), (10, 10, 10))
    car[150:210, 60:260] = plate
    boxes = classic_plate_candidates(car)
    assert boxes, "phải tìm được ít nhất 1 vùng giống biển"
    from traffic_safety_ai.utils.geometry import iou
    assert max(iou(b.xyxy, [60, 150, 260, 210]) for b in boxes) > 0.3
    assert plate_quality(plate) > 0.5
