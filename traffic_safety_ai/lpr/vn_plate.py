"""Ngữ pháp biển số xe Việt Nam: chuẩn hoá, sửa lỗi OCR theo VỊ TRÍ, định dạng hiển thị, tra mã địa phương.

Căn cứ (đối chiếu ngày 04/11/2026):
  * Thông tư 79/2024/TT-BCA (sửa đổi bởi TT 13/2025, TT 51/2025): seri dùng 20 chữ cái
    A B C D E F G H K L M N P S T U V X Y Z (không dùng I, J, O, Q, R, W); thứ tự đăng ký 5 chữ số.
  * Ô tô: "51F-123.45"; mô tô cấp từ 8/2023: 2 chữ cái "29-AB 123.45"; mô tô cũ chữ+số "59-X1 123.45";
    biển cũ 4 số "51F-1234"; seri đặc biệt LD, KT, DA, CD, RM, HC, NG, NN, QT, CV...
  * Ký hiệu địa phương sau sáp nhập 34 tỉnh (từ 01/7/2025): tỉnh mới KẾ THỪA mọi ký hiệu của tỉnh cũ,
    biển đã cấp giữ nguyên giá trị.

Ý tưởng sửa lỗi: OCR hay nhầm 0↔D/O, 1↔I/T, 5↔S, 8↔B, 2↔Z, 6↔G, 4↔A. Vì mỗi vị trí trên biển VN
chỉ có thể là CHỮ hoặc SỐ, ta ép từng ký tự về đúng loại của vị trí đó theo mẫu và chọn mẫu ít phải sửa nhất.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SERIES_LETTERS = frozenset("ABCDEFGHKLMNPSTUVXYZ")
CAR_EXTRA_LETTERS = frozenset("R")            # rơ moóc / sơ mi rơ moóc (seri R, RM)
SPECIAL_SERIES = frozenset({"LD", "KT", "DA", "CD", "RM", "HC", "NG", "NN", "QT", "CV", "MK", "TĐ"})

DIGIT_FIX = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "1",
             "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}
LETTER_FIX = {"0": "D", "O": "D", "Q": "D", "8": "B", "5": "S", "2": "Z", "6": "G", "4": "A", "1": "T",
              "7": "T", "I": "T", "J": "T", "W": "V"}

# Ký hiệu địa phương → (tỉnh/thành từ 01/7/2025, tỉnh/thành trước sáp nhập)
PROVINCES: dict[str, tuple[str, str]] = {
    "11": ("Cao Bằng", "Cao Bằng"), "12": ("Lạng Sơn", "Lạng Sơn"), "14": ("Quảng Ninh", "Quảng Ninh"),
    "15": ("Hải Phòng", "Hải Phòng"), "16": ("Hải Phòng", "Hải Phòng"), "17": ("Hưng Yên", "Thái Bình"),
    "18": ("Ninh Bình", "Nam Định"), "19": ("Phú Thọ", "Phú Thọ"), "20": ("Thái Nguyên", "Thái Nguyên"),
    "21": ("Lào Cai", "Yên Bái"), "22": ("Tuyên Quang", "Tuyên Quang"), "23": ("Tuyên Quang", "Hà Giang"),
    "24": ("Lào Cai", "Lào Cai"), "25": ("Lai Châu", "Lai Châu"), "26": ("Sơn La", "Sơn La"),
    "27": ("Điện Biên", "Điện Biên"), "28": ("Phú Thọ", "Hòa Bình"), "29": ("Hà Nội", "Hà Nội"),
    "30": ("Hà Nội", "Hà Nội"), "31": ("Hà Nội", "Hà Nội"), "32": ("Hà Nội", "Hà Nội"),
    "33": ("Hà Nội", "Hà Nội"), "34": ("Hải Phòng", "Hải Dương"), "35": ("Ninh Bình", "Ninh Bình"),
    "36": ("Thanh Hóa", "Thanh Hóa"), "37": ("Nghệ An", "Nghệ An"), "38": ("Hà Tĩnh", "Hà Tĩnh"),
    "39": ("Đồng Nai", "Đồng Nai"), "40": ("Hà Nội", "Hà Nội"), "41": ("TP. Hồ Chí Minh", "TP. Hồ Chí Minh"),
    "43": ("Đà Nẵng", "Đà Nẵng"), "47": ("Đắk Lắk", "Đắk Lắk"), "48": ("Lâm Đồng", "Đắk Nông"),
    "49": ("Lâm Đồng", "Lâm Đồng"), "60": ("Đồng Nai", "Đồng Nai"), "61": ("TP. Hồ Chí Minh", "Bình Dương"),
    "62": ("Tây Ninh", "Long An"), "63": ("Đồng Tháp", "Tiền Giang"), "64": ("Vĩnh Long", "Vĩnh Long"),
    "65": ("Cần Thơ", "Cần Thơ"), "66": ("Đồng Tháp", "Đồng Tháp"), "67": ("An Giang", "An Giang"),
    "68": ("An Giang", "Kiên Giang"), "69": ("Cà Mau", "Cà Mau"), "70": ("Tây Ninh", "Tây Ninh"),
    "71": ("Vĩnh Long", "Bến Tre"), "72": ("TP. Hồ Chí Minh", "Bà Rịa - Vũng Tàu"),
    "73": ("Quảng Trị", "Quảng Bình"), "74": ("Quảng Trị", "Quảng Trị"), "75": ("Huế", "Thừa Thiên Huế"),
    "76": ("Quảng Ngãi", "Quảng Ngãi"), "77": ("Gia Lai", "Bình Định"), "78": ("Đắk Lắk", "Phú Yên"),
    "79": ("Khánh Hòa", "Khánh Hòa"), "80": ("Cục CSGT (xe cơ quan Trung ương)", "Cơ quan Trung ương"),
    "81": ("Gia Lai", "Gia Lai"), "82": ("Quảng Ngãi", "Kon Tum"), "83": ("Cần Thơ", "Sóc Trăng"),
    "84": ("Vĩnh Long", "Trà Vinh"), "85": ("Khánh Hòa", "Ninh Thuận"), "86": ("Lâm Đồng", "Bình Thuận"),
    "88": ("Phú Thọ", "Vĩnh Phúc"), "89": ("Hưng Yên", "Hưng Yên"), "90": ("Ninh Bình", "Hà Nam"),
    "92": ("Đà Nẵng", "Quảng Nam"), "93": ("Đồng Nai", "Bình Phước"), "94": ("Cà Mau", "Bạc Liêu"),
    "95": ("Cần Thơ", "Hậu Giang"), "97": ("Thái Nguyên", "Bắc Kạn"), "98": ("Bắc Ninh", "Bắc Giang"),
    "99": ("Bắc Ninh", "Bắc Ninh"),
}
for _c in range(50, 60):
    PROVINCES[str(_c)] = ("TP. Hồ Chí Minh", "TP. Hồ Chí Minh")

# Mẫu: D = chữ số, L = chữ cái seri, ký tự khác = cố định.  prior = phạt nhẹ cho mẫu hiếm.
TEMPLATES: list[tuple[str, str, list[str], float]] = [
    ("car5", "car", ["D", "D", "L", "D", "D", "D", "D", "D"], 0.0),
    ("car4", "car", ["D", "D", "L", "D", "D", "D", "D"], 0.35),
    ("car2l5", "car", ["D", "D", "L", "L", "D", "D", "D", "D", "D"], 0.25),
    ("car2l4", "car", ["D", "D", "L", "L", "D", "D", "D", "D"], 0.5),
    ("moto_ld5", "moto", ["D", "D", "L", "D", "D", "D", "D", "D", "D"], 0.0),
    ("moto_ll5", "moto", ["D", "D", "L", "L", "D", "D", "D", "D", "D"], 0.05),
    ("moto_ld4", "moto", ["D", "D", "L", "D", "D", "D", "D", "D"], 0.3),
    ("moto_ll4", "moto", ["D", "D", "L", "L", "D", "D", "D", "D"], 0.4),
    ("moto_md5", "moto", ["D", "D", "M", "Đ", "D", "D", "D", "D", "D", "D"], 0.1),
]
_SERIES_SPAN = {"car5": (2, 3), "car4": (2, 3), "car2l5": (2, 4), "car2l4": (2, 4), "moto_ld5": (2, 4),
                "moto_ll5": (2, 4), "moto_ld4": (2, 4), "moto_ll4": (2, 4), "moto_md5": (2, 5)}

COLOR_LABELS = {
    "white": "Nền màu trắng, chữ và số màu đen",
    "yellow": "Nền màu vàng, chữ và số màu đen",
    "blue": "Nền màu xanh, chữ và số màu trắng",
    "red": "Nền màu đỏ, chữ và số màu trắng",
    "unknown": "Chưa xác định",
}


@dataclass
class PlateParse:
    text: str
    display: str
    kind: str
    template: str
    province_code: str
    series: str
    serial: str
    province: str | None
    province_old: str | None
    corrections: int
    valid: bool
    score: float
    reasons: list[str] = field(default_factory=list)


def normalize_text(raw: str) -> str:
    s = (raw or "").upper()
    s = s.replace("Ð", "Đ").replace("ð", "Đ")
    return re.sub(r"[^0-9A-ZĐ]", "", s)


def vehicle_kind(vehicle_cls: str | None) -> str | None:
    if vehicle_cls in ("motorcycle",):
        return "moto"
    if vehicle_cls in ("car", "bus", "truck"):
        return "car"
    return None


def format_serial(serial: str) -> str:
    return f"{serial[:3]}.{serial[3:]}" if len(serial) == 5 else serial


def format_display(province: str, series: str, serial: str, kind: str) -> str:
    if kind == "moto":
        return f"{province}-{series} {format_serial(serial)}"
    return f"{province}{series}-{format_serial(serial)}"


def _coerce(chars: str, pattern: list[str], kind: str) -> tuple[str, int] | None:
    out = []
    fixes = 0
    for ch, p in zip(chars, pattern):
        if p == "D":
            if ch.isdigit():
                out.append(ch)
            elif ch in DIGIT_FIX:
                out.append(DIGIT_FIX[ch])
                fixes += 1
            else:
                return None
        elif p == "L":
            allowed = SERIES_LETTERS | (CAR_EXTRA_LETTERS if kind == "car" else frozenset())
            if ch in allowed or ch == "Đ":
                out.append(ch)
            elif ch in LETTER_FIX:
                out.append(LETTER_FIX[ch])
                fixes += 1
            else:
                return None
        elif p == "Đ":
            if ch in ("Đ", "D"):
                out.append("Đ")
            elif ch in ("0", "O"):
                out.append("Đ")
                fixes += 1
            else:
                return None
        else:  # ký tự cố định
            if ch == p:
                out.append(ch)
            elif p == "M" and ch in ("N", "H"):
                out.append("M")
                fixes += 1
            else:
                return None
    return "".join(out), fixes


def parse_plate(raw: str, vehicle_cls: str | None = None, layout: str | None = None) -> PlateParse | None:
    """Phân tích chuỗi OCR thành biển số VN hợp lệ nhất. None nếu không khớp mẫu nào."""
    s = normalize_text(raw)
    if len(s) < 7:
        return None
    kind_hint = vehicle_kind(vehicle_cls)
    best: PlateParse | None = None
    for name, kind, pattern, prior in TEMPLATES:
        if len(pattern) != len(s):
            continue
        if kind_hint is not None and kind != kind_hint:
            continue
        r = _coerce(s, pattern, kind)
        if r is None:
            continue
        text, fixes = r
        a, b = _SERIES_SPAN[name]
        prov, series, serial = text[:2], text[a:b], text[b:]
        reasons: list[str] = []
        pv = PROVINCES.get(prov)
        valid = True
        score = fixes + prior
        if pv is None:
            valid = False
            score += 2.5
            reasons.append(f"mã địa phương {prov} không tồn tại")
        if name.startswith("car2l") and series not in SPECIAL_SERIES:
            score += 0.2
        if kind_hint is None:
            # không biết loại xe: biển 2 dòng nghiêng về xe máy, 1 dòng nghiêng về ô tô
            if layout == "2line" and kind == "car" and name in ("car5", "car2l5"):
                score += 0.15
            if layout == "1line" and kind == "moto":
                score += 0.3
        disp = format_display(prov, series, serial, kind)
        cand = PlateParse(text, disp, kind, name, prov, series, serial, pv[0] if pv else None,
                          pv[1] if pv else None, fixes, valid, score, reasons)
        if best is None or cand.score < best.score:
            best = cand
    return best


def province_label(code: str) -> str:
    pv = PROVINCES.get(code)
    if not pv:
        return "Không xác định"
    new, old = pv
    return new if new == old else f"{new} (trước 7/2025: {old})"
