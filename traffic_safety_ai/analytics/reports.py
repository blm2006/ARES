"""Báo cáo tổng hợp (HTML in được + CSV mở bằng Excel): vi phạm, lưu lượng quy đổi xe con, tốc độ V85, điểm nóng."""

from __future__ import annotations

import base64
import csv
import html
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from traffic_safety_ai.analytics.counting import PCU_DEFAULT, summarize_counts, summarize_speeds
from traffic_safety_ai.core.config import AppPaths
from traffic_safety_ai.core.types import CLASS_LABEL_VI, SEVERITY_VI, STATUS_VI
from traffic_safety_ai.evidence.store import Store, evidence_folder

CSV_COLUMNS = [
    ("id", "Mã hồ sơ"), ("occurred_at", "Thời gian"), ("camera_id", "Camera"), ("location", "Địa điểm"),
    ("rule_code", "Mã lỗi"), ("behavior", "Hành vi"), ("severity", "Phân loại"), ("vehicle_label", "Loại phương tiện"),
    ("plate_display", "Biển kiểm soát"), ("plate_conf", "Độ tin biển số"), ("plate_color_label", "Màu biển"),
    ("legal_ref", "Căn cứ tham khảo"), ("fine", "Mức phạt tham khảo"), ("status", "Trạng thái"),
    ("ai_confidence", "Độ tin AI"), ("evidence_dir", "Thư mục bằng chứng"),
]


def svg_bars(labels: list[str], values: list[float], color: str = "#1f5fd0", width: int = 640, bar_h: int = 22) -> str:
    if not values:
        return "<p>—</p>"
    vmax = max(values) or 1
    h = bar_h * len(values) + 10
    parts = [f'<svg width="{width}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI,Arial" font-size="12">']
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = 5 + i * bar_h
        w = int((width - 260) * v / vmax)
        parts.append(f'<text x="0" y="{y + 15}">{html.escape(lab[:34])}</text>')
        parts.append(f'<rect x="230" y="{y + 3}" width="{max(w, 1)}" height="{bar_h - 7}" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{236 + w}" y="{y + 15}">{v:g}</text>')
    parts.append("</svg>")
    return "".join(parts)


def heatmap_image(background: np.ndarray, points: list[tuple[float, float]], sigma: float = 22.0) -> np.ndarray:
    h, w = background.shape[:2]
    acc = np.zeros((h, w), np.float32)
    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            acc[int(y), int(x)] += 1.0
    if acc.max() <= 0:
        return background.copy()
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)
    acc = (255 * acc / acc.max()).astype(np.uint8)
    colored = cv2.applyColorMap(acc, cv2.COLORMAP_JET)
    alpha = (acc.astype(np.float32) / 255.0)[..., None] * 0.6
    out = (background.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha).astype(np.uint8)
    return out


def _b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:      # BOM để Excel đọc đúng tiếng Việt
        w = csv.writer(f)
        w.writerow([c[1] for c in columns])
        for r in rows:
            vals = []
            for k, _ in columns:
                v = r.get(k)
                if k == "severity":
                    v = SEVERITY_VI.get(v or "", v)
                elif k == "status":
                    v = STATUS_VI.get(v or "", v)
                vals.append("" if v is None else v)
            w.writerow(vals)


def build_report(store: Store, paths: AppPaths, camera_id: str | None = None, day: str | None = None,
                 run_id: str | None = None, speed_limit: float | None = None,
                 pcu: dict[str, float] | None = None, title_suffix: str = "") -> Path:
    paths.ensure()
    pcu = pcu or PCU_DEFAULT
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = paths.reports / f"bao_cao_{day or 'tat_ca'}_{camera_id or 'all'}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    viols, total = store.list_violations(camera=camera_id, day=day, limit=100000)
    if run_id:
        viols = [v for v in viols if v.get("run_id") == run_id]
    active = [v for v in viols if v.get("status") != "loai_bo"]
    counts = store.counts(camera_id=camera_id, day=day, run_id=run_id)
    speeds = store.speeds(camera_id=camera_id, day=day, run_id=run_id)
    cs = summarize_counts(counts, pcu)
    ss = summarize_speeds(speeds, speed_limit)

    write_csv(out_dir / "vi_pham.csv", viols, CSV_COLUMNS)
    write_csv(out_dir / "luu_luong.csv", counts, [("line_id", "Mặt cắt"), ("line_name", "Tên"), ("bucket_start", "Khung 15 phút"),
                                                  ("cls", "Loại xe"), ("direction", "Hướng"), ("n", "Số lượng")])
    write_csv(out_dir / "toc_do.csv", speeds, [("at", "Thời điểm"), ("track_id", "ID"), ("cls", "Loại xe"),
                                               ("speed_kmh", "Tốc độ (km/h)"), ("method", "Phương pháp")])

    # --- thống kê vi phạm
    by_rule: dict[str, list[Any]] = {}
    by_hour: dict[str, int] = {}
    by_sev: dict[str, int] = {}
    for v in active:
        k = v["rule_code"]
        by_rule.setdefault(k, [v.get("rule_name") or k, 0])
        by_rule[k][1] += 1
        hh = (v.get("occurred_at") or "")[11:13] or "?"
        by_hour[hh] = by_hour.get(hh, 0) + 1
        by_sev[v.get("severity")] = by_sev.get(v.get("severity"), 0) + 1
    rule_rows = sorted(by_rule.items())
    hours = sorted(by_hour)

    # --- bản đồ điểm nóng
    heat_html = ""
    runs = [r for r in store.runs(200) if not camera_id or r.get("camera_id") == camera_id]
    bg = None
    for r in runs:
        p = paths.runs / str(r["id"]) / "background.jpg"
        if p.exists():
            bg = cv2.imread(str(p))
            break
    pts = [(h["x"], h["y"]) for h in store.hazard_points(camera_id=camera_id, run_id=run_id, day=day)]
    if bg is not None and pts:
        hm = heatmap_image(bg, pts)
        cv2.imwrite(str(out_dir / "ban_do_diem_nong.jpg"), hm)
        heat_html = f'<img style="max-width:100%" src="data:image/jpeg;base64,{_b64(cv2.resize(hm, None, fx=0.6, fy=0.6))}">'

    # --- bảng lưu lượng
    flow_html = ""
    for lid, s in cs.items():
        rows = "".join(
            f"<tr><td>{html.escape(r['bucket'][11:16])}</td><td>{r['motorcycle']}</td><td>{r['bicycle']}</td>"
            f"<td>{r['car']}</td><td>{r['bus']}</td><td>{r['truck']}</td><td><b>{r['total']}</b></td>"
            f"<td>{r['pcu']:g}</td><td>{r['person']}</td></tr>" for r in s["rows"])
        peak = s["peak"]
        ptxt = (f"Giờ cao điểm bắt đầu {peak['start'][11:16]}: {peak['pcu_h']:g} xcqđ/giờ, PHF = {peak['phf']}"
                if peak else "Chưa đủ 1 giờ số liệu để xác định giờ cao điểm")
        flow_html += (f"<h3>{html.escape(s['name'])}</h3><p>{ptxt}. Tổng {s['total']} xe, {s['pcu']:g} xcqđ.</p>"
                      "<table><tr><th>Khung 15'</th><th>Xe máy</th><th>Xe đạp</th><th>Ô tô</th><th>Buýt/khách</th>"
                      f"<th>Tải</th><th>Tổng xe</th><th>Xcqđ</th><th>Người đi bộ</th></tr>{rows}</table>")
    if not flow_html:
        flow_html = "<p>Chưa có số liệu đếm (cần khai báo lines.counting).</p>"

    speed_rows = "".join(
        f"<tr><td>{html.escape(CLASS_LABEL_VI.get(c, c))}</td><td>{s['n']}</td><td>{s['mean']}</td><td>{s['v15']}</td>"
        f"<td>{s['v50']}</td><td><b>{s['v85']}</b></td><td>{s['max']}</td>"
        f"<td>{'' if s['pct_over'] is None else s['pct_over']}</td></tr>" for c, s in ss.items())

    def _link(v: dict[str, Any]) -> str:
        d = evidence_folder(v, paths.evidence)
        if d is None:
            return html.escape(v["id"])
        try:     # đường dẫn TƯƠNG ĐỐI → mở được cả từ ổ đĩa lẫn qua dashboard
            rel = os.path.relpath(Path(d) / "record.html", out_dir).replace(os.sep, "/")
        except ValueError:     # khác ổ đĩa (Windows)
            rel = Path(d).joinpath("record.html").as_uri()
        return f"<a href='{html.escape(rel)}'>{html.escape(v['id'])}</a>"

    vrows = "".join(
        f"<tr><td>{_link(v)}</td>"
        f"<td>{html.escape((v.get('occurred_at') or '').replace('T', ' '))}</td><td>{html.escape(v['rule_code'])}</td>"
        f"<td>{html.escape(v.get('behavior') or '')}</td><td>{html.escape(v.get('vehicle_label') or '')}</td>"
        f"<td>{html.escape(v.get('plate_display') or '—')}</td>"
        f"<td>{html.escape(SEVERITY_VI.get(v.get('severity'), ''))}</td>"
        f"<td>{html.escape(STATUS_VI.get(v.get('status'), ''))}</td></tr>" for v in active[:500])

    doc = f"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<title>Báo cáo giám sát ATGT khu vực trường học</title>
<style>body{{font-family:'Segoe UI',Arial,sans-serif;margin:24px;color:#1b1b1b;max-width:1180px}}
h1{{font-size:22px;color:#1f2a44;margin:0}} h2{{font-size:18px;margin:26px 0 8px;border-bottom:2px solid #1f2a44}}
h3{{font-size:15px;margin:14px 0 4px}} table{{border-collapse:collapse;width:100%;margin:8px 0}}
th,td{{border:1px solid #ccc;padding:5px 8px;font-size:13px;text-align:left}} th{{background:#f2f4f7}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}} .card{{border:1px solid #d5dbe3;border-radius:8px;padding:10px 14px;min-width:150px}}
.card b{{font-size:22px;display:block}} .note{{background:#fff4e5;border-left:4px solid #e08a00;padding:8px 12px;font-size:13px}}
</style></head><body>
<h1>BÁO CÁO GIÁM SÁT AN TOÀN GIAO THÔNG KHU VỰC TRƯỜNG HỌC{html.escape(title_suffix)}</h1>
<p>Camera: {html.escape(camera_id or 'tất cả')} · Ngày: {html.escape(day or 'tất cả')} · Lập lúc {datetime.now():%H:%M %d/%m/%Y}</p>
<div class="cards">
<div class="card">Vi phạm<b>{by_sev.get('vi_pham', 0)}</b></div>
<div class="card">Nghi vấn cần xác minh<b>{by_sev.get('nghi_van', 0)}</b></div>
<div class="card">Nguy cơ an toàn<b>{by_sev.get('nguy_co', 0)}</b></div>
<div class="card">Xe đếm được<b>{sum(s['total'] for s in cs.values())}</b></div>
<div class="card">Xe con quy đổi<b>{sum(s['pcu'] for s in cs.values()):g}</b></div>
<div class="card">V85 (km/h)<b>{ss.get('Tất cả', {}).get('v85', '—')}</b></div></div>
<div class="note">Số liệu do AI ghi nhận; vi phạm ở trạng thái "chờ xác minh" chưa phải kết luận. Căn cứ pháp lý chỉ để tham khảo.
Tốc độ đo bằng camera là ước lượng. Hệ số xcqđ: xe đạp {pcu['bicycle']}, xe máy {pcu['motorcycle']}, ô tô {pcu['car']},
tải {pcu['truck']}, buýt {pcu['bus']} (TCVN 4054:2005).</div>
<h2>1. Vi phạm và nguy cơ theo loại</h2>
<table><tr><th>Mã</th><th>Tên</th><th>Số trường hợp</th></tr>{''.join(f'<tr><td>{k}</td><td>{html.escape(n)}</td><td>{c}</td></tr>' for k, (n, c) in rule_rows)}</table>
{svg_bars([f'{k} {n}' for k, (n, c) in rule_rows], [c for _, (n, c) in rule_rows], '#c62828')}
<h2>2. Phân bố theo giờ</h2>{svg_bars([f'{h} giờ' for h in hours], [by_hour[h] for h in hours])}
<h2>3. Lưu lượng giao thông (15 phút)</h2>{flow_html}
<h2>4. Tốc độ</h2>
<table><tr><th>Loại xe</th><th>Số mẫu</th><th>TB</th><th>V15</th><th>V50</th><th>V85</th><th>Max</th><th>% vượt giới hạn</th></tr>
{speed_rows or '<tr><td colspan=8>Chưa có (cần hiệu chuẩn mặt đường)</td></tr>'}</table>
<h2>5. Bản đồ điểm nóng nguy cơ</h2>{heat_html or '<p>Chưa có điểm nguy cơ.</p>'}
<h2>6. Danh sách hồ sơ ({len(active)} / {total})</h2>
<table><tr><th>Mã hồ sơ</th><th>Thời gian</th><th>Mã</th><th>Hành vi</th><th>Phương tiện</th><th>Biển số</th><th>Phân loại</th><th>Trạng thái</th></tr>{vrows}</table>
<p style="color:#666;font-size:12px">Tệp kèm: vi_pham.csv, luu_luong.csv, toc_do.csv (mở bằng Excel).</p>
</body></html>"""
    out = out_dir / "bao_cao.html"
    out.write_text(doc, encoding="utf-8")
    return out
