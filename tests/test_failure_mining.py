"""Nguồn B phải bắt được false negative — thứ v1 không bao giờ bắt được."""

from __future__ import annotations

from collector.failure_mining import FailureMiner
from collector.uncertainty import UncertaintySampler, frame_uncertainty
from common.schemas import BBox, FailureReason
from realtime.tracker import TrackHistory


def box(x, tid, conf=0.9):
    return BBox(x1=x, y1=100, x2=x + 50, y2=250, conf=conf, track_id=tid)


def test_track_gap_detected_and_interpolated():
    h = TrackHistory(maxlen=30, short_track_frames=5)
    h.add(1, [box(100, 1)])
    h.add(2, [])              # người biến mất 1 frame -> false negative
    events = h.add(3, [box(120, 1)])

    gaps = [e for e in events if e.reason is FailureReason.TRACK_GAP]
    assert gaps, "không bắt được track gap"
    e = gaps[0]
    assert e.frame_idx == 2
    assert e.track_ids == [1]
    nb = e.neighbor_boxes[0]
    assert nb.interpolated and 100 < nb.x1 < 120, "bbox nội suy phải nằm giữa t-1 và t+1"


def test_missed_person_produces_no_uncertainty_signal():
    """Lý do nguồn B tồn tại: frame bỏ sót người KHÔNG sinh tín hiệu bất định.

    Frame 2 ở test trên hoàn toàn rỗng -> điểm bất định 0 -> nguồn A không bao giờ
    lấy nó. Nếu chỉ có nguồn A, lỗi này không bao giờ vào dataset.
    """
    assert frame_uncertainty([]) == 0.0
    s = UncertaintySampler(top_percent=5, min_samples=10)
    for _ in range(100):
        s.observe([box(10, 1, conf=0.95)])
    assert not s.should_sample(frame_uncertainty([]))


def test_id_switch_detected():
    h = TrackHistory(maxlen=30)
    h.add(1, [box(100, 7)])
    events = h.add(2, [box(102, 9)])  # gần như cùng chỗ nhưng ID đổi
    assert any(e.reason is FailureReason.ID_SWITCH for e in events)


def test_short_track_detected():
    h = TrackHistory(maxlen=60, short_track_frames=5)
    h.add(1, [box(100, 3)])
    h.add(2, [box(103, 3)])  # track chỉ sống 2 frame, dưới ngưỡng 5
    events = []
    for i in range(3, 30):
        events += h.add(i, [])
    short = [e for e in events if e.reason is FailureReason.SHORT_TRACK]
    assert short, "không bắt được track chết sớm"
    assert short[0].track_ids == [3]
    assert len(short) == 1, "một track chỉ được báo một lần"


def test_cooldown_prevents_burst_of_duplicates():
    """Một sự cố kéo dài không được sinh 30 mẫu giống nhau."""
    from realtime.tracker import FailureEvent

    m = FailureMiner(cooldown_frames=30, max_per_burst=3)
    emitted = 0
    for i in range(1, 25):
        ev = [FailureEvent(frame_idx=i, reason=FailureReason.TRACK_GAP, track_ids=[1])]
        emitted += len(m.select(ev))
    assert emitted == 1, f"cooldown không chặn burst: {emitted} mẫu"


def test_priority_track_gap_first():
    from realtime.tracker import FailureEvent

    m = FailureMiner(cooldown_frames=0, max_per_burst=1)
    out = m.select(
        [
            FailureEvent(frame_idx=5, reason=FailureReason.SHORT_TRACK, track_ids=[2]),
            FailureEvent(frame_idx=5, reason=FailureReason.TRACK_GAP, track_ids=[1]),
        ]
    )
    assert out[0].reason is FailureReason.TRACK_GAP


def test_uncertainty_uses_percentile_not_fixed_band():
    """Phân phối dịch đi thì ngưỡng cũng phải dịch theo."""
    s = UncertaintySampler(top_percent=10, window_hours=24, min_samples=50)
    for c in [0.95] * 200:  # model rất tự tin
        s.observe([box(0, 1, conf=c)])
    thr_confident = s.window.threshold_for_top(10)

    s2 = UncertaintySampler(top_percent=10, window_hours=24, min_samples=50)
    for c in [0.55] * 200:  # model lưỡng lự
        s2.observe([box(0, 1, conf=c)])
    thr_unsure = s2.window.threshold_for_top(10)

    assert thr_unsure > thr_confident, "ngưỡng không tự co giãn theo phân phối"
