# ARES — AI bảo hộ lao động (đồ bảo hộ)

> Phần mở rộng của AI hành vi (`BEHAVIOR_AI.md`). Mã nguồn: `ares/ppe/`. Cập nhật: 17/11/2026.

## 1. Làm gì

Phát hiện người **thiếu đồ bảo hộ bắt buộc** hoặc **đang đeo thứ bị cấm**, rồi cảnh báo "Vi phạm đồ bảo hộ" kèm clip — đi chung
đường với cảnh báo té ngã (dashboard, Đúng / Báo nhầm, Thử video).

| Món | Mã | Xét ở vùng | Điều kiện để dám kết luận "thiếu" |
|---|---|---|---|
| Kính bảo hộ (kể cả kính che mặt) | `goggles` | đầu | thấy mặt (mũi + mắt), đầu ≥ 22 px |
| Chụp tai chống ồn | `ear` | đầu | thấy ít nhất 1 tai, đầu ≥ 18 px |
| Khẩu trang | `mask` | đầu | thấy mặt, đầu ≥ 20 px |
| Nón bảo hộ | `helmet` | đầu (cả khi nhìn từ sau) | đầu ≥ 12 px |
| Áo phản quang · Áo bảo hộ / tạp dề | `vest` · `coat` | thân | thấy hai vai |
| Găng tay | `gloves` | bàn tay | thấy ít nhất 1 cổ tay |
| Giày | `boots` | bàn chân | thấy ít nhất 1 cổ chân |

Kích thước đầu tính sau khi ảnh được thu về cỡ model nhìn (`ARES_PPE_IMGSZ`, mặc định 960). Không thấy rõ thì **không kết luận**
(trên hình ghi "CHƯA THẤY RÕ ĐỒ BẢO HỘ") — thà bỏ sót còn hơn báo bừa.

## 2. Mẫu theo nơi làm việc

| Mẫu | Bắt buộc | Cấm |
|---|---|---|
| **Xưởng gỗ** | kính, chụp tai, khẩu trang | **găng tay** (găng dễ bị cuốn vào lưỡi cưa / mũi khoan đang quay) |
| Công trường | nón, áo phản quang, giày | — |
| Phòng thí nghiệm | kính, áo bảo hộ, găng tay, khẩu trang | — |

Mỗi **địa điểm** chọn riêng: dashboard → **Địa điểm → Chọn đồ bảo hộ** (chọn mẫu rồi sửa từng món). Không chọn = không kiểm.
Server gửi luật về máy AI trong heartbeat (≤ 30 giây), máy AI lưu ở `behavior/ppe_rules.json` để mất mạng vẫn dùng.
Mặc định cho mọi camera của một máy: `ARES_PPE_REQUIRED=woodshop` (hoặc `goggles,ear`) và `ARES_PPE_FORBIDDEN=gloves`.

Những thứ camera **không** kiểm được ở xưởng gỗ: tóc dài chưa buộc, tay áo rộng, trang sức, máy có đang chạy hay không
(nên chụp tai / khẩu trang được nhắc cả khi máy tắt — bỏ chọn nếu thấy phiền).

## 3. Cách quyết định

1. YOLO đồ bảo hộ chạy 2 lần/giây (`ARES_PPE_FPS`) cho camera **có luật và có người**.
2. Mỗi hộp đồ bảo hộ gán cho người có vùng cơ thể tương ứng (dựng từ khung xương) chứa tâm hộp.
3. Bỏ phiếu 4 giây gần nhất: chỉ cần ≥ 1/3 số lần "thấy" là coi như đang đeo (model hay sót vật nhỏ).
4. Thiếu (hoặc đeo đồ cấm) liên tục `ARES_PPE_MISSING_S` = 5 giây → cảnh báo, ghi rõ từng món. Đeo lại rồi tháo → báo lại.
   Vẫn vi phạm → nhắc lại sau `ARES_PPE_REMIND_S` = 300 giây.
5. Trên hình: dòng thứ hai dưới nhãn người — cam `THIẾU KÍNH, CHỤP TAI` / `ĐEO GĂNG TAY (CẤM)`, xanh `ĐỦ BẢO HỘ`, xám khi chưa thấy rõ;
   hộp mảnh xanh ngọc quanh các món AI nhìn thấy.

## 4. Model

- Cài: chạy **`CAI_AI_BAO_HO.bat`** trên máy chủ (`python -m ares.ppe setup`): tải `ppe_sh17_yolo8s.pt` (~22 MB), kiểm mã SHA-256, in bảng
  "lớp của model → món". Kiểm lại bất cứ lúc nào: `python -m ares.ppe check`.
- Mặc định: YOLOv8s của nhóm tác giả bộ dữ liệu **SH17** (17 lớp: kính, chụp tai, khẩu trang, kính che mặt, găng tay, nón, áo phản quang,
  giày, áo bảo hộ + bộ phận cơ thể). **Bộ dữ liệu SH17 theo giấy phép CC BY-NC-SA 4.0 (phi thương mại)** — dùng cho xưởng của trường,
  nghiên cứu, thử nghiệm. Bán thương mại: phải train model riêng trên dữ liệu được phép.
- Dùng model khác: đặt file `.pt` (Ultralytics, task detect) rồi `ARES_PPE_WEIGHTS=<file>`. Tên lớp được ánh xạ theo từ khoá
  (`helmet/hardhat`, `goggle/glass`, `earmuf`, `mask`, `vest`, `glove`, `boot/shoe`, `suit/apron/coat`; lớp `no_...` = thiếu).

## 5. Giới hạn đã biết

- Kính cận / kính thường có thể bị nhận là kính bảo hộ. Lớp "Shoes" của SH17 là giày nói chung, không riêng giày bảo hộ.
- Nút tai (earplug) quá nhỏ, camera không thấy — chỉ nhận chụp tai.
- Camera gắn trần xa: đầu người < 20 px thì không xét kính / khẩu trang. Đặt camera cách bàn máy 2–4 m, ngang tầm ngực–đầu.
- **Chưa đo độ chính xác trên xưởng thật.** Logic ghép và bộ đếm đã test bằng dữ liệu giả; model thật chưa chạy trên máy dev.
