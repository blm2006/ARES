"""ARES Server — máy chủ trung tâm: nhận số liệu + sự kiện từ nhiều máy Edge, dashboard tổng hợp nhiều địa điểm.

Không cần GPU, không cần torch/ultralytics: chạy được trên VPS rẻ hoặc chính máy Edge (chế độ local).
    python -m ares_server run
    python -m ares_server add-site "Trường A" --address "..."
"""

__version__ = "0.1.0"
