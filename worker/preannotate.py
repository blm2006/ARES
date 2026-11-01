"""Pre-annotation — GỢI Ý, KHÔNG PHẢI GROUND TRUTH. Thay đổi quan trọng nhất so với v1.

v1: conf > 0.9 + tracker ổn định -> auto-label -> vào thẳng dataset training.
Đó chính xác là những mẫu model ĐÃ LÀM ĐÚNG: gradient ≈ 0, học lại không đổi gì,
mà còn đóng băng sai lệch sẵn có — nếu model luôn bỏ sót người ở góc tối thì
không có nhãn nào trong dataset sửa được điều đó.

Ở đây pre-annotation có đúng một vai trò: giảm thời gian gán nhãn của người.
KHÔNG tồn tại đường nào đi thẳng vào dataset. Hàm duy nhất ghi ra dataset nằm ở
dataset/builder.py và nó chỉ đọc bản ghi đã review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from common.schemas import BBox, QueueMessage, Source

log = logging.getLogger(__name__)


@dataclass
class PreAnnotation:
    boxes: list[BBox]
    model_version: str
    n_from_model: int
    n_interpolated: int

    def as_json(self) -> list[dict]:
        return [b.model_dump() for b in self.boxes]


def merge_suggestions(
    model_boxes: list[BBox], neighbor_boxes: list[BBox], iou_threshold: float = 0.45
) -> list[BBox]:
    """Thêm bbox nội suy vào chỗ model KHÔNG có box nào trùng.

    Không ghi đè box của model — người duyệt cần thấy cả hai để biết model sai ở đâu.
    """
    out = list(model_boxes)
    for nb in neighbor_boxes:
        if any(nb.iou(mb) > iou_threshold for mb in model_boxes):
            continue
        out.append(nb.model_copy(update={"interpolated": True, "conf": 0.0}))
    return out


class PreAnnotator:
    def __init__(self, detector, model_version: str = "unknown"):
        self.detector = detector
        self.model_version = model_version

    def annotate(self, image: np.ndarray, msg: QueueMessage) -> PreAnnotation:
        if self.detector is not None:
            try:
                model_boxes = self.detector.infer(image).boxes
            except Exception:
                log.exception("infer lỗi, dùng box lúc bắt mẫu làm gợi ý")
                model_boxes = list(msg.boxes)
        else:
            model_boxes = list(msg.boxes)

        neighbors: list[BBox] = []
        if msg.source == Source.FAILURE_MINING:
            # Với nguồn B, nội suy từ track t-1/t+1 là gợi ý cho đúng vùng model bỏ sót.
            neighbors = list(msg.neighbor_boxes)

        merged = merge_suggestions(model_boxes, neighbors)
        return PreAnnotation(
            boxes=merged,
            model_version=self.model_version,
            n_from_model=len(model_boxes),
            n_interpolated=len(merged) - len(model_boxes),
        )


# Ưu tiên hiển thị trong review UI: failure_mining > uncertainty > random (spec).
PRIORITY = {
    Source.FAILURE_MINING.value: 0,
    Source.UNCERTAINTY.value: 1,
    Source.RANDOM.value: 2,
}


def review_priority(source: Source | str) -> int:
    src = source.value if isinstance(source, Source) else source
    return PRIORITY.get(src, 9)
