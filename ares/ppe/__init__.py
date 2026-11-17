"""AI bảo hộ lao động cho ARES: phát hiện người THIẾU đồ bảo hộ bắt buộc (kính, chụp tai, khẩu trang, nón, áo...)
hoặc ĐANG ĐEO thứ bị cấm (vd găng tay gần máy quay trong xưởng gỗ).

Luồng:  khung hình -> YOLO đồ bảo hộ (detector.py)  ┐
        khung xương từng người (AI hành vi)        ┴-> ghép đồ với người theo vùng đầu / thân / tay / chân (rules.py)
        -> thiếu liên tục vài giây -> cảnh báo "Vi phạm đồ bảo hộ" kèm clip (đi chung đường với cảnh báo té ngã)

Mỗi địa điểm tự chọn món nào bắt buộc / cấm (dashboard server -> Địa điểm -> Đồ bảo hộ). Không chọn = không kiểm.
Camera chỉ kết luận khi THẤY RÕ vùng đó (thấy mặt mới xét kính/khẩu trang; người ở quá xa thì không kết luận).

Lệnh:  python -m ares.ppe setup | check
"""

# Món đồ bảo hộ chuẩn hoá: mã -> (tên đầy đủ, tên ngắn vẽ lên hình)
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

# Mẫu có sẵn theo loại nơi làm việc. Xưởng gỗ: kính + chụp tai + khẩu trang chống bụi; găng tay bị CẤM vì găng
# dễ bị cuốn vào lưỡi cưa / mũi khoan đang quay (quy định an toàn xưởng mộc của các trường đều ghi như vậy).
PRESETS = {
    "woodshop": {"name": "Xưởng gỗ", "required": ["goggles", "ear", "mask"], "forbidden": ["gloves"]},
    "construction": {"name": "Công trường", "required": ["helmet", "vest", "boots"], "forbidden": []},
    "lab": {"name": "Phòng thí nghiệm", "required": ["goggles", "coat", "gloves", "mask"], "forbidden": []},
}


def clean_rules(r) -> dict:
    """Chuẩn hoá luật đồ bảo hộ từ JSON / chuỗi "goggles,ear" / tên mẫu -> {"required": [...], "forbidden": [...]}."""
    if isinstance(r, str):
        r = r.strip()
        if r in PRESETS:
            r = PRESETS[r]
        else:
            r = {"required": [x.strip() for x in r.split(",")]}
    if not isinstance(r, dict):
        return {"required": [], "forbidden": []}
    req = [x for x in dict.fromkeys(r.get("required") or []) if x in ITEMS]
    forb = [x for x in dict.fromkeys(r.get("forbidden") or []) if x in ITEMS and x not in req]
    return {"required": req, "forbidden": forb}


def describe(missing: list[str], forbidden: list[str], short: bool = False) -> str:
    """-> "thiếu kính bảo hộ, chụp tai chống ồn; đang đeo găng tay (bị cấm)"."""
    k = 1 if short else 0
    parts = []
    if missing:
        parts.append("thiếu " + ", ".join(ITEMS[i][k] for i in missing))
    if forbidden:
        parts.append("đang đeo " + ", ".join(ITEMS[i][k] for i in forbidden) + ("" if short else " (bị cấm ở đây)"))
    return "; ".join(parts)
