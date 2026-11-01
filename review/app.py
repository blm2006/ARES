"""Review UI — CỬA DUY NHẤT vào dataset training.

Ba làn (spec 2.5):
  Fast lane : pre-annotation đúng -> 1 phím (A). Mẫu dễ vẫn vào dataset, ~1 giây/frame.
  Fix lane  : vẽ/sửa/xoá bbox rồi lưu (S). Ưu tiên hiển thị B -> A -> C.
  Reject    : frame vô dụng/hỏng/trùng, có ghi lý do -> vòng lại cải thiện 2.1.

Mọi quyết định ghi vào review_queue.final_boxes. dataset/builder.py chỉ đọc từ đó.
"""

from __future__ import annotations

import io
from datetime import datetime

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy import func, select

from common.config import get_settings
from common.db import Frame as FrameRow
from common.db import ReviewItem, audit, session_scope, utcnow
from common.schemas import BBox, ReviewAction
from common.storage import get_storage
from review.agreement import summary as agreement_summary

REJECT_REASONS = [
    "khong_co_nguoi",
    "anh_hong_blur",
    "trung_lap",
    "che_khuat_hoan_toan",
    "ngoai_vung_quan_tam",
    "khac",
]


class Decision(BaseModel):
    action: ReviewAction
    boxes: list[BBox] | None = None
    reject_reason: str | None = None
    reviewer: str = "local"


class LabelReq(BaseModel):
    name: str


def create_app() -> FastAPI:
    app = FastAPI(title="review-ui")
    s = get_settings()
    storage = get_storage(s)

    # ---------- hàng chờ ----------
    @app.get("/api/next")
    def next_item(skip: int = 0):
        """Ưu tiên: priority tăng dần (B=0, A=1, C=2), rồi cũ trước."""
        with session_scope() as sess:
            item = sess.execute(
                select(ReviewItem)
                .where(ReviewItem.status == "pending")
                .order_by(ReviewItem.priority, ReviewItem.created_at)
                .offset(skip)
                .limit(1)
            ).scalar_one_or_none()
            if item is None:
                return JSONResponse({"empty": True, "pending": 0})

            frame = sess.get(FrameRow, item.frame_id)
            pending = sess.execute(
                select(func.count()).select_from(ReviewItem).where(ReviewItem.status == "pending")
            ).scalar_one()
            return JSONResponse(
                {
                    "empty": False,
                    "pending": int(pending),
                    "item_id": item.id,
                    "frame_id": item.frame_id,
                    "source": item.source,
                    "failure_reason": frame.failure_reason,
                    "cam_id": frame.cam_id,
                    "ts": frame.ts.isoformat(),
                    "width": frame.width,
                    "height": frame.height,
                    "uncertainty": frame.uncertainty_score,
                    "boxes": frame.preannot_boxes or [],
                    "preannot_model": frame.preannot_model,
                    "class_names": s.class_name_map(),
                }
            )

    @app.post("/api/labels")
    def add_label(req: LabelReq):
        """Thêm/đặt TÊN cho một lớp. Trùng tên (không phân biệt hoa thường) thì
        trả lại id cũ; tên mới thì cấp id kế tiếp và lưu vào var/labels.json."""
        import json

        name = (req.name or "").strip()
        if not name:
            raise HTTPException(400, "tên rỗng")
        m = s.class_name_map()
        for i, n in m.items():
            if n.lower() == name.lower():
                return {"id": i, "class_names": {str(k): v for k, v in m.items()}}
        new_id = (max(m) + 1) if m else 0
        f = s.data_root / "labels.json"
        cur = {}
        if f.exists():
            try:
                cur = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                cur = {}
        cur[str(new_id)] = name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        audit("review-ui", "add_label", str(new_id), {"name": name})
        m[new_id] = name
        return {"id": new_id, "class_names": {str(k): v for k, v in m.items()}}

    @app.get("/api/image/{frame_id}")
    def image(frame_id: str):
        with session_scope() as sess:
            frame = sess.get(FrameRow, frame_id)
            if frame is None:
                raise HTTPException(404)
            path = frame.path
        audit("review-ui", "read_image", path)
        img = storage.read_image(path)
        # Chỉ nén để truyền qua web — ảnh trên đĩa vẫn là bản gốc.
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(500)
        return Response(buf.tobytes(), media_type="image/jpeg")

    # ---------- quyết định ----------
    @app.post("/api/decide/{item_id}")
    def decide(item_id: int, d: Decision):
        with session_scope() as sess:
            item = sess.get(ReviewItem, item_id)
            if item is None or item.status == "done":
                raise HTTPException(409, "item đã xử lý")
            frame = sess.get(FrameRow, item.frame_id)
            pre = [BBox(**b) for b in (frame.preannot_boxes or [])]

            if d.action is ReviewAction.APPROVE:
                # Approve = chấp nhận nguyên pre-annotation, bỏ các box nội suy chưa xác nhận.
                final = [b for b in pre if not b.interpolated]
                edited = False
            elif d.action is ReviewAction.FIX:
                if d.boxes is None:
                    raise HTTPException(400, "fix phải kèm boxes")
                final = [b.model_copy(update={"interpolated": False}) for b in d.boxes]
                edited = True
            else:
                final = []
                edited = True

            item.status = "done"
            item.action = d.action.value
            item.reject_reason = d.reject_reason
            item.final_boxes = [b.model_dump() for b in final]
            item.n_boxes_final = len(final)
            item.edited = edited
            item.reviewer = d.reviewer
            item.reviewed_at = utcnow()

        audit(d.reviewer, f"review_{d.action.value}", str(item_id), {"n_boxes": len(final)})
        return {"ok": True, "n_boxes": len(final), "edited": edited}

    @app.post("/api/skip/{item_id}")
    def skip(item_id: int):
        return {"ok": True}  # chỉ là con trỏ phía client, không đổi DB

    # ---------- số liệu ----------
    @app.get("/api/stats")
    def stats():
        with session_scope() as sess:
            by_status = dict(
                sess.execute(
                    select(ReviewItem.status, func.count()).group_by(ReviewItem.status)
                ).all()
            )
            by_source = dict(
                sess.execute(
                    select(ReviewItem.source, func.count())
                    .where(ReviewItem.status == "pending")
                    .group_by(ReviewItem.source)
                ).all()
            )
        return {
            "by_status": {k: int(v) for k, v in by_status.items()},
            "pending_by_source": {k: int(v) for k, v in by_source.items()},
            "agreement": agreement_summary(30),
            "target": s.agreement_target,
        }

    @app.get("/api/reasons")
    def reasons():
        return REJECT_REASONS

    @app.get("/", response_class=HTMLResponse)
    def index():
        from pathlib import Path

        return (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")

    return app


def main() -> None:
    import uvicorn

    from common.db import init_db

    init_db()
    s = get_settings()
    uvicorn.run(create_app(), host=s.review_host, port=s.review_port)


if __name__ == "__main__":
    main()
