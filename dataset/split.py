"""Chia train/val/test THEO SESSION (cam_id + ngày), tuyệt đối không theo frame.

Vì sao: frame liền nhau trong cùng một track giống nhau gần như tuyệt đối. Chia
ngẫu nhiên theo frame nghĩa là gần như mọi frame test đều có anh em sinh đôi trong
train — mAP đẹp nhưng vô nghĩa, và bạn sẽ không phát hiện được model đang xấu đi.

Gán session bằng hash tất định, không bằng random.shuffle: thêm dữ liệu mới KHÔNG
được làm một session đang ở test nhảy sang train.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

GOLDEN_FILE = "golden_sessions.json"


def _hash01(key: str, salt: str = "split-v1") -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


@dataclass
class Split:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    golden: list[str] = field(default_factory=list)
    rolling: list[str] = field(default_factory=list)

    def of(self, session_key: str) -> str | None:
        for name in ("train", "val", "test"):
            if session_key in getattr(self, name):
                return name
        return None

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "golden": len(self.golden),
            "rolling": len(self.rolling),
        }


class GoldenSet:
    """Đóng băng vĩnh viễn. Một session đã vào golden thì không bao giờ ra, và
    golden không bao giờ được thêm mẫu mới — nếu không, "so với lần trước" mất nghĩa."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.sessions: list[str] = []
        if self.path.exists():
            self.sessions = json.loads(self.path.read_text(encoding="utf-8"))["sessions"]

    @property
    def frozen(self) -> bool:
        return self.path.exists()

    def freeze(self, sessions: list[str]) -> None:
        if self.frozen:
            raise RuntimeError(
                f"Golden set đã đóng băng ở {self.path} với {len(self.sessions)} session. "
                "Không thêm, không sửa. Muốn bộ chuẩn khác thì tạo file mới có tên khác "
                "và ghi rõ trong manifest là đổi baseline."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions = sorted(set(sessions))
        self.path.write_text(
            json.dumps(
                {"frozen_at": datetime.now(timezone.utc).isoformat(), "sessions": self.sessions},
                indent=2,
            ),
            encoding="utf-8",
        )


def split_sessions(
    sessions: list[str],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    golden: list[str] | None = None,
    rolling_days: int = 30,
    today: datetime | None = None,
) -> Split:
    """golden luôn nằm ở test. rolling = session trong N ngày gần nhất (để bắt drift)."""
    golden = list(golden or [])
    tr, va, _te = ratios
    today = today or datetime.now(timezone.utc)
    cutoff = (today - timedelta(days=rolling_days)).strftime("%Y-%m-%d")

    out = Split(golden=sorted(set(golden)))
    for key in sorted(set(sessions)):
        if key in out.golden:
            out.test.append(key)
        else:
            r = _hash01(key)
            if r < tr:
                out.train.append(key)
            elif r < tr + va:
                out.val.append(key)
            else:
                out.test.append(key)
        day = key.split(":")[-1]
        if day >= cutoff:
            out.rolling.append(key)

    moved = _repair_empty_splits(out)
    if moved:
        log.warning(
            "Ít session nên có tập bị rỗng, đã cân lại: %s. "
            "Khi thu thêm được nhiều ngày thì split trở lại thuần theo hash.",
            "; ".join(moved),
        )
    return out


def _repair_empty_splits(out: Split) -> list[str]:
    """Gán theo hash là tất định và ổn định, nhưng với ÍT session thì một dải có thể
    rỗng: 12 session mà val chiếm dải rộng 10% thì rất dễ ra 10/0/2.

    Vá lại bằng cách chuyển session có hash GẦN BIÊN của dải rỗng nhất, lấy từ tập
    đông nhất. Vẫn tất định (chỉ phụ thuộc tập session), và chỉ kích hoạt khi còn ít
    session — có đủ dữ liệu rồi thì hàm này không bao giờ chạm vào gì.

    Golden không bao giờ bị chuyển: nó đã đóng băng ở test.
    """
    moved: list[str] = []
    golden = set(out.golden)
    if len(out.train) + len(out.val) + len(out.test) < 3:
        return moved

    for name in ("val", "test", "train"):
        while not getattr(out, name):
            donor = max(("train", "val", "test"), key=lambda n: len(getattr(out, n)))
            pool = [k for k in getattr(out, donor) if k not in golden]
            if donor == name or len(pool) <= 1:
                return moved
            # hash lớn nhất = gần biên val/test nhất; nhỏ nhất = gần biên train nhất
            pool.sort(key=_hash01, reverse=(name != "train"))
            key = pool[0]
            getattr(out, donor).remove(key)
            getattr(out, name).append(key)
            getattr(out, name).sort()
            moved.append(f"{key}: {donor} -> {name}")
    return moved


def assert_no_leakage(split: Split) -> None:
    """Một session không được nằm ở hai tập. Test này chạy trong CI, không phải để trang trí."""
    s_tr, s_va, s_te = set(split.train), set(split.val), set(split.test)
    for a, b, na, nb in ((s_tr, s_va, "train", "val"), (s_tr, s_te, "train", "test"), (s_va, s_te, "val", "test")):
        overlap = a & b
        if overlap:
            raise AssertionError(f"leakage {na}/{nb}: {sorted(overlap)[:5]}")
