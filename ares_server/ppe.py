"""Đồ bảo hộ: danh mục món + mẫu theo loại nơi làm việc. Bản sao của ares/ppe/__init__.py (server chạy độc lập trong
Docker, không kèm gói `ares`) — test_ppe.py kiểm tra hai bản luôn giống nhau."""

ITEMS = {
    "goggles": ("kính bảo hộ", "kính"),
    "ear": ("chụp tai chống ồn", "chụp tai"),
    "mask": ("khẩu trang", "khẩu trang"),
    "helmet": ("nón bảo hộ", "nón"),
    "vest": ("áo phản quang", "áo phản quang"),
    "coat": ("áo bảo hộ / tạp dề", "áo bảo hộ"),
    "gloves": ("găng tay", "găng tay"),
    "boots": ("giày bảo hộ", "giày"),
}

PRESETS = {
    "woodshop": {"name": "Xưởng gỗ", "required": ["goggles", "ear", "mask"], "forbidden": ["gloves"]},
    "construction": {"name": "Công trường", "required": ["helmet", "vest", "boots"], "forbidden": []},
    "lab": {"name": "Phòng thí nghiệm", "required": ["goggles", "coat", "gloves", "mask"], "forbidden": []},
}


def clean_rules(r) -> dict:
    if isinstance(r, str):
        r = PRESETS.get(r.strip(), {})
    if not isinstance(r, dict):
        return {"required": [], "forbidden": []}
    req = [x for x in dict.fromkeys(r.get("required") or []) if x in ITEMS]
    forb = [x for x in dict.fromkeys(r.get("forbidden") or []) if x in ITEMS and x not in req]
    return {"required": req, "forbidden": forb}
