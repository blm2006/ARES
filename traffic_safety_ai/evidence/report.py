"""Ảnh ghép bằng chứng + phiếu ghi nhận (HTML in được) theo các trường của tra cứu phạt nguội.

Các trường: Biển kiểm soát · Màu biển · Loại phương tiện · Thời gian vi phạm · Địa điểm vi phạm · Hành vi vi phạm ·
Trạng thái · Đơn vị phát hiện vi phạm · Nơi giải quyết vụ việc — cộng thêm căn cứ/mức phạt THAM KHẢO, số đo,
độ tin cậy AI và mã băm SHA-256 của từng tệp (chống chỉnh sửa).

Phiếu KHÔNG mang quốc huy, tên cơ quan nhà nước hay hình thức quyết định xử phạt: đây là dữ liệu do AI
ghi nhận, chờ người có thẩm quyền xác minh (NĐ 135/2021/NĐ-CP).
"""

from __future__ import annotations

import base64
import hashlib
import html
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from traffic_safety_ai.core.types import SEVERITY_VI, STATUS_VI
from traffic_safety_ai.utils.text import TextRenderer, default_renderer

NAVY = (68, 42, 31)          # #1F2A44 (BGR)
RED = (53, 57, 229)
GRAY = (235, 235, 235)
DISCLAIMER = ("Dữ liệu do hệ thống AI tự động ghi nhận phục vụ giám sát an toàn giao thông khu vực trường học. "
              "Không phải quyết định xử phạt. Chỉ người có thẩm quyền mới được xác minh và xử lý vi phạm.")

MEAS_LABELS = {
    "toc_do_kmh": "Tốc độ ước lượng (km/h)", "gioi_han_kmh": "Tốc độ cho phép (km/h)", "vuot_kmh": "Vượt (km/h)",
    "phuong_phap_do": "Phương pháp đo", "sai_so_kmh": "Sai số ước lượng (± km/h)",
    "trang_thai_den_khi_cat_vach": "Đèn khi cắt vạch dừng", "den_do_da_bat_s": "Đèn đỏ đã bật (giây)",
    "vach_dung": "Vạch dừng", "khung_cat_vach": "Khung hình cắt vạch", "so_nguoi_tren_xe": "Số người trên xe",
    "ti_le_khong_mu": "Tỉ lệ khung không mũ", "so_lan_quan_sat": "Số lần quan sát", "vi_tri": "Vị trí",
    "lan_duong": "Làn đường", "cos_lech_huong": "Cos góc lệch hướng", "goc_lech_do": "Góc lệch (độ)",
    "quang_duong": "Quãng đường", "khu_vuc": "Khu vực", "quang_duong_tren_via_he": "Quãng đi trên vỉa hè",
    "TTC_s": "Thời gian tới va chạm TTC (s)", "PET_s": "PET (s)", "khoang_cach_gan_nhat": "Khoảng cách gần nhất",
    "toc_do_xe_kmh": "Tốc độ xe (km/h)", "vach_qua_duong": "Vạch qua đường", "gia_toc_ms2": "Gia tốc (m/s²)",
    "toc_do_truoc_kmh": "Tốc độ trước khi phanh (km/h)", "khoang_cach_thoi_gian_s": "Khoảng cách thời gian (s)",
    "khe_ho_m": "Khe hở (m)", "ti_le_be_rong": "Tỉ lệ bề rộng hàng/xe", "be_rong_uoc_luong_m": "Bề rộng ước lượng (m)",
    "phuong_phap": "Phương pháp", "thoi_gian_dung_s": "Thời gian dừng (s)", "nguong_s": "Ngưỡng (s)",
    "loai_vung": "Loại vùng", "so_xe_hang_ngang": "Số xe hàng ngang", "huong_re": "Hướng rẽ", "xi_nhan": "Xi-nhan",
    "ti_le_khung_co_dau_hieu": "Tỉ lệ khung có dấu hiệu", "dau_hieu": "Dấu hiệu",
    "toc_do_sau_kmh": "Tốc độ sau khi phanh (km/h)", "toc_do_xe_minh_kmh": "Tốc độ xe ghi hình (km/h)",
    "ben": "Phía", "khoang_cach_m": "Khoảng cách (m)", "lech_ngang_m": "Lệch ngang (m)",
    "chieu_cao_vat_che_m": "Chiều cao vật che (m)", "doi_tuong": "Đối tượng", "van_toc_ngang": "Vận tốc ngang",
    "vat_che": "Vật che khuất", "goc_cat_ngang_do": "Góc cắt ngang đường (độ)",
    "thoi_gian_trong_long_duong_s": "Thời gian trong lòng đường (s)", "huong_xe": "Hướng đầu xe",
    "muc_no_cua_vat_dung_yen": "Tốc độ nở của vật đứng yên (1/s)", "toc_do_no_hop": "Tốc độ nở hộp (1/s)",
    "van_toc_doc_xe_kia_ms": "Vận tốc dọc của xe kia (m/s)", "ti_le_dien_tich_xe_truoc": "Tỉ lệ diện tích xe trước",
    "ti_le_banh_xe_tren_via_he": "Tỉ lệ thời gian trên vỉa hè", "lan": "Làn", "vach": "Vạch",
    "loai_xe_duoc_phep": "Loại xe được phép", "nguon": "Nguồn", "vung_tam_nhin": "Vùng tầm nhìn",
}

ZONE_KIND_VI = {"crosswalk": "vạch qua đường", "no_stopping": "cấm dừng & đỗ", "no_parking": "cấm đỗ",
                "sidewalk": "vỉa hè"}
VALUE_VI = {
    **ZONE_KIND_VI,
    "homography": "hiệu chuẩn mặt đường (homography)", "vach_do": "cặp vạch đo tốc độ",
    "speed_trap": "cặp vạch đo tốc độ", "model": "model AI", "bgsub": "tách nền ảnh", "sim": "mô phỏng",
    "person": "người", "motorcycle": "xe máy", "bicycle": "xe đạp", "car": "ô tô", "bus": "xe buýt/khách",
    "truck": "xe tải", "front": "đầu xe (đang hướng về camera)", "rear": "đuôi xe", "left": "trái", "right": "phải",
    "red": "đỏ", "yellow": "vàng", "green": "xanh", "unknown": "không rõ", "off": "tắt",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return s.replace(".", ",")
    if isinstance(v, str):
        return VALUE_VI.get(v, v)
    if isinstance(v, (list, tuple)):
        return ", ".join(fmt_value(x) for x in v)
    return str(v)


def meas_lines(meas: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for k, v in (meas or {}).items():
        if k.startswith("_") or v is None:
            continue
        out.append((MEAS_LABELS.get(k, k.replace("_", " ").capitalize()), fmt_value(v)))
    return out


def fit(img: np.ndarray | None, w: int, h: int, bg: tuple[int, int, int] = (30, 30, 30)) -> np.ndarray:
    canvas = np.full((h, w, 3), bg, np.uint8)
    if img is None or img.size == 0:
        return canvas
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def wrap(text: str, tr: TextRenderer, size: int, width: int, bold: bool = False) -> list[str]:
    words = (text or "").split()
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if tr.measure(cand, size, bold)[0] <= width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def has_vehicle(rec: dict[str, Any]) -> bool:
    return bool(rec.get("vehicle_cls"))


def record_fields(rec: dict[str, Any]) -> list[tuple[str, str]]:
    plate = rec.get("plate_display") or ("Chưa đọc được — xem ảnh biển số" if has_vehicle(rec)
                                         else "Không áp dụng (sự kiện nguy cơ, không gắn với phương tiện)")
    if rec.get("plate_display") and rec.get("plate_conf") is not None:
        plate += f" (AI tin cậy {rec['plate_conf'] * 100:.0f}%{'' if rec.get('plate_valid') else ', chưa hợp lệ'})"
    when = rec.get("occurred_at", "")
    try:
        from datetime import datetime

        d = datetime.fromisoformat(when)
        when = d.strftime("%H:%M:%S, %d/%m/%Y")
    except Exception:
        pass
    beh = f"{rec.get('rule_code')} — {rec.get('behavior') or rec.get('rule_name')}"
    legal = rec.get("legal_ref") or "—"
    if rec.get("legal_ref") and not rec.get("legal_verified"):
        legal += " [chưa đối chiếu]"
    fine = rec.get("fine") or "—"
    if rec.get("points"):
        fine += f"; {rec['points']}"
    return [
        ("Biển kiểm soát", plate),
        ("Màu biển", rec.get("plate_color_label") or "Chưa xác định"),
        ("Loại phương tiện", rec.get("vehicle_label") or "—"),
        ("Thời gian vi phạm", when),
        ("Địa điểm vi phạm", rec.get("location") or "—"),
        ("Hành vi vi phạm", beh),
        ("Phân loại", SEVERITY_VI.get(rec.get("severity", ""), rec.get("severity", ""))),
        ("Căn cứ tham khảo", legal),
        ("Mức phạt tham khảo", fine),
        ("Trạng thái", STATUS_VI.get(rec.get("status", "cho_xac_minh"), "Chờ xác minh")),
        ("Đơn vị phát hiện vi phạm", f"{rec.get('detect_unit', 'Hệ thống AI')} — camera {rec.get('camera_id')}"),
        ("Nơi giải quyết vụ việc", rec.get("handling_note") or "Chưa chuyển cơ quan có thẩm quyền"),
    ]


def compose_sheet(rec: dict[str, Any], overview: np.ndarray, before: np.ndarray | None, after: np.ndarray | None,
                  vehicle: np.ndarray | None, plate: np.ndarray | None, snapshot_sha: str = "",
                  text: TextRenderer | None = None) -> np.ndarray:
    tr = text or default_renderer()
    W, H = 1920, 1250
    img = np.full((H, W, 3), (248, 248, 248), np.uint8)
    # ---- đầu trang
    cv2.rectangle(img, (0, 0), (W, 118), NAVY, -1)
    tr.put(img, "PHIẾU GHI NHẬN VI PHẠM / NGUY CƠ GIAO THÔNG — DỮ LIỆU AI CHỜ XÁC MINH", (24, 14), size=36,
           color=(255, 255, 255), bold=True)
    sev = rec.get("severity", "vi_pham")
    sub = f"Mã hồ sơ: {rec.get('id')}   |   Camera: {rec.get('camera_id')} — {rec.get('camera_name', '')}   |   " \
          f"{SEVERITY_VI.get(sev, sev)}"
    tr.put(img, sub, (24, 68), size=24, color=(225, 230, 240))
    # ---- ảnh toàn cảnh
    ov = fit(overview, 1180, 664)
    img[132:132 + 664, 12:12 + 1180] = ov
    tr.put(img, "Toàn cảnh tại thời điểm ghi nhận", (20, 140), size=20, color=(255, 255, 255), bg=(0, 0, 0))
    # ---- phương tiện + biển số
    x0 = 1206
    veh = has_vehicle(rec)
    tr.put(img, "PHƯƠNG TIỆN" if veh else "ĐỐI TƯỢNG LIÊN QUAN", (x0, 132), size=22, color=NAVY, bold=True)
    img[166:166 + 330, x0:x0 + 700] = fit(vehicle, 700, 330)
    tr.put(img, "BIỂN SỐ", (x0, 506), size=22, color=NAVY, bold=True)
    img[540:540 + 160, x0:x0 + 700] = fit(plate, 700, 160, bg=(60, 60, 60))
    if veh:
        ptxt = rec.get("plate_display") or "CHƯA ĐỌC ĐƯỢC"
        tr.put(img, ptxt, (x0 + 350, 712), size=52 if rec.get("plate_display") else 34, color=(20, 20, 20),
               bg=(255, 255, 255), bold=True, anchor="ct", pad=6)
    else:
        tr.put(img, "KHÔNG ÁP DỤNG", (x0 + 350, 590), size=30, color=(235, 235, 235), bold=True, anchor="ct")
        tr.put(img, "Sự kiện nguy cơ — không gắn với một phương tiện cụ thể", (x0 + 350, 640), size=18,
               color=(210, 210, 210), anchor="ct")
    # ---- trước / sau
    tr.put(img, "Trước ~1 giây", (12, 806), size=20, color=NAVY, bold=True)
    tr.put(img, "Sau ~1 giây", (606, 806), size=20, color=NAVY, bold=True)
    img[836:836 + 330, 12:12 + 586] = fit(before, 586, 330)
    img[836:836 + 330, 606:606 + 586] = fit(after, 586, 330)
    # ---- bảng thông tin
    y = 806
    fields = record_fields(rec)[2:10]
    lab_w = min(260, max(tr.measure(f"{lb}:", 19, True)[0] for lb, _ in fields) + 14)
    for label, val in fields:
        lines = wrap(val, tr, 19, 700 - lab_w - 6)[:3]
        if y + 24 * len(lines) > 1170:
            break
        tr.put(img, f"{label}:", (x0, y), size=19, color=(90, 90, 90), bold=True, pad=1)
        for i, line in enumerate(lines):
            tr.put(img, line, (x0 + lab_w, y + i * 24), size=19, color=(25, 25, 25), pad=1)
        y += 24 * len(lines) + 6
    ms = [f"{k}: {v}" for k, v in meas_lines(rec.get("measurements") or {})[:4]]
    if ms:
        for line in wrap(" · ".join(ms), tr, 17, 690)[:3]:
            if y + 22 > H - 76:            # không đè lên chân trang
                break
            tr.put(img, line, (x0, y), size=17, color=(140, 40, 40), pad=1)
            y += 22
    # ---- chân trang
    cv2.rectangle(img, (0, H - 72), (W, H), (230, 230, 230), -1)
    tr.put(img, DISCLAIMER, (16, H - 66), size=17, color=(60, 60, 60), pad=1)
    tr.put(img, f"SHA-256 ảnh gốc: {snapshot_sha or '—'}", (16, H - 38), size=16, color=(90, 90, 90), pad=1)
    return img


def _img_tag(path: Path, alt: str, max_w: str = "100%") -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f'<img alt="{html.escape(alt)}" src="data:{mime};base64,{b64}" style="max-width:{max_w}">'


def record_html(rec: dict[str, Any], folder: Path, catalog_warning: str = "") -> str:
    rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in record_fields(rec))
    meas = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
                   for k, v in meas_lines(rec.get("measurements") or {}))
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in (rec.get("notes") or []))
    files = rec.get("files") or {}
    frows = "".join(f"<tr><td>{html.escape(n)}</td><td class=mono>{html.escape(f.get('sha256', ''))}</td>"
                    f"<td>{f.get('size', 0):,}</td></tr>" for n, f in files.items())
    sheet = _img_tag(folder / "composite.jpg", "Ảnh ghép bằng chứng")
    plate = _img_tag(folder / "plate.jpg", "Ảnh biển số", "360px")
    cands = rec.get("plate_candidates") or []
    cand = ", ".join(f"{c[0]} ({c[1] * 100:.0f}%)" for c in cands) if cands else "—"
    return f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><title>Phiếu ghi nhận {html.escape(rec.get('id', ''))}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;margin:24px;color:#1b1b1b;max-width:1100px}}
h1{{font-size:21px;margin:0 0 4px;color:#1f2a44}} .sub{{color:#555;margin:0 0 16px}}
table{{border-collapse:collapse;width:100%;margin:10px 0 18px}} th,td{{border:1px solid #ccc;padding:7px 10px;
text-align:left;vertical-align:top;font-size:14px}} th{{background:#f2f4f7;width:32%}}
.warn{{background:#fff4e5;border-left:4px solid #e08a00;padding:10px 14px;font-size:13px;margin:12px 0}}
.mono{{font-family:Consolas,monospace;font-size:12px;word-break:break-all}} h2{{font-size:16px;margin:18px 0 6px}}
@media print{{body{{margin:8mm}} .noprint{{display:none}}}}
</style></head><body>
<h1>PHIẾU GHI NHẬN VI PHẠM / NGUY CƠ GIAO THÔNG</h1>
<p class="sub">Mã hồ sơ <b>{html.escape(rec.get('id', ''))}</b> · Dữ liệu do AI ghi nhận — CHỜ XÁC MINH</p>
<div class="warn">{html.escape(DISCLAIMER)} {html.escape(catalog_warning)}</div>
<table>{rows}</table>
<h2>Ảnh biển số</h2>{plate}<p>Các phương án đọc biển: {html.escape(cand)} · Số lần đọc: {rec.get('plate_reads') or 0}</p>
<h2>Số đo của hệ thống</h2><table>{meas or '<tr><td>—</td></tr>'}</table>
<h2>Ghi chú</h2><ul>{notes or '<li>—</li>'}</ul>
<h2>Ảnh ghép bằng chứng</h2>{sheet}
<h2>Tệp đính kèm và mã băm SHA-256</h2>
<table><tr><th>Tệp</th><th>SHA-256</th><th>Byte</th></tr>{frows}</table>
<p class="sub">Độ tin cậy AI tổng hợp: {rec.get('ai_confidence', 0) * 100:.0f}% · Khung hình {rec.get('frame_idx')} ·
Lần chạy {html.escape(str(rec.get('run_id', '')))} · Nguồn: {html.escape(str(rec.get('source', '')))}</p>
</body></html>"""
