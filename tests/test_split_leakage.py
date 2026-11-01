"""Phase 5 nghiệm thu: KHÔNG frame nào của cùng một session nằm ở cả train và test.

Đây là test quan trọng nhất trong repo. Nếu nó fail, mọi con số mAP về sau là rác.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from dataset.split import GoldenSet, assert_no_leakage, split_sessions


def make_sessions(n_cam: int = 4, n_days: int = 30) -> list[str]:
    base = datetime(2026, 1, 1)
    return [
        f"cam{c:02d}:{(base + timedelta(days=d)):%Y-%m-%d}"
        for c in range(1, n_cam + 1)
        for d in range(n_days)
    ]


def test_session_never_in_two_splits():
    split = split_sessions(make_sessions(), (0.8, 0.1, 0.1))
    assert_no_leakage(split)


def test_frames_of_same_session_stay_together():
    """Mô phỏng frame: mọi frame của một session phải rơi vào ĐÚNG MỘT tập."""
    sessions = make_sessions(3, 10)
    split = split_sessions(sessions, (0.8, 0.1, 0.1))

    frames = [(f"{s}#{i}", s) for s in sessions for i in range(25)]
    placement: dict[str, set[str]] = {}
    for _frame_id, sess in frames:
        placement.setdefault(sess, set()).add(split.of(sess))

    for sess, parts in placement.items():
        assert len(parts) == 1, f"session {sess} rơi vào nhiều tập: {parts}"


def test_split_is_deterministic_when_data_grows():
    """Thêm dữ liệu mới KHÔNG được làm session cũ nhảy tập."""
    old = make_sessions(2, 10)
    s1 = split_sessions(old, (0.8, 0.1, 0.1))
    s2 = split_sessions(old + make_sessions(4, 40), (0.8, 0.1, 0.1))
    for sess in old:
        assert s1.of(sess) == s2.of(sess), f"{sess} đổi tập khi thêm dữ liệu"


def test_golden_always_in_test_and_frozen(tmp_path):
    sessions = make_sessions(3, 12)
    g = GoldenSet(tmp_path / "golden.json")
    g.freeze(sessions[:5])

    split = split_sessions(sessions, (0.8, 0.1, 0.1), golden=g.sessions)
    for s in g.sessions:
        assert s in split.test
        assert s not in split.train and s not in split.val

    g2 = GoldenSet(tmp_path / "golden.json")
    with pytest.raises(RuntimeError, match="đã đóng băng"):
        g2.freeze(sessions)  # thêm vào golden phải bị từ chối


def test_rolling_only_recent_days():
    today = datetime(2026, 3, 1)
    sessions = [f"cam01:{(today - timedelta(days=d)):%Y-%m-%d}" for d in range(90)]
    split = split_sessions(sessions, (0.8, 0.1, 0.1), rolling_days=30, today=today)
    assert len(split.rolling) == 31  # 0..30 ngày
    assert all(s.split(":")[1] >= "2026-01-30" for s in split.rolling)


def test_ratios_roughly_respected():
    split = split_sessions(make_sessions(10, 60), (0.8, 0.1, 0.1))
    total = len(split.train) + len(split.val) + len(split.test)
    assert 0.74 < len(split.train) / total < 0.86


# ---------------------------------------------------------------------------
# Bẫy của dữ liệu 1 camera / 1 ngày: đúng 1 session.
# Split theo session nên session đó chỉ vào được MỘT tập. Nếu đồng thời đóng băng
# golden thì nó vào test vĩnh viễn, train rỗng, Phase 6 chết mà không rõ vì sao.
# ---------------------------------------------------------------------------
def test_golden_not_frozen_when_too_few_sessions(settings, tmp_path):
    from dataset.builder import MIN_SESSIONS_FOR_GOLDEN
    from dataset.split import GOLDEN_FILE, GoldenSet

    assert MIN_SESSIONS_FOR_GOLDEN >= 3
    g = GoldenSet(settings.datasets_dir / GOLDEN_FILE)
    assert not g.frozen, "chưa dựng dataset lần nào thì không được có golden"


def test_builder_refuses_dataset_with_empty_train(settings):
    """Thà dừng với thông báo rõ còn hơn đẻ ra dataset train rỗng."""
    import pytest

    from dataset.builder import DatasetBuilder

    b = DatasetBuilder(settings)
    with pytest.raises(RuntimeError) as e:
        b.build(notes="không có mẫu nào đã duyệt")
    assert "duyệt" in str(e.value).lower() or "review" in str(e.value).lower()
