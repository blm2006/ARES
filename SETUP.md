# Cài đặt trên máy mới

## 1. Giải nén và cài

Mở PowerShell tại thư mục chứa file zip:

```powershell
Expand-Archive .\yolo-al-code-YYYYMMDD.zip -DestinationPath .\yolo-al
cd .\yolo-al
Set-ExecutionPolicy -Scope Process Bypass -Force
.\install.ps1
```

`install.ps1` tự làm hết:

| Bước | Việc |
|---|---|
| 1 | Tìm Python ≥ 3.11 (thử `py -3.12`, `py -3.11`, rồi `python`) |
| 2 | Tạo `.venv` — không đụng vào Python hệ thống |
| 3 | Dò `nvidia-smi`. Có GPU → torch CUDA 12.4. Không có → torch CPU |
| 4 | Cài `requirements.txt` + `faiss-cpu` (thiếu faiss thì tự dùng numpy) |
| 5 | Cảnh báo nếu `.env` để `DEVICE=cuda` mà máy không có GPU |
| 6 | Tạo schema DB + thư mục `var/` |
| 7 | Chạy 73 test (không cần GPU, không cần camera) |

Tuỳ chọn:

```powershell
.\install.ps1 -Cpu        # ép dùng torch CPU dù có GPU
.\install.ps1 -NoVenv     # cài thẳng vào Python hệ thống
.\install.ps1 -SkipTest   # bỏ qua bước chạy test
```

## 2. Mang dữ liệu đã thu sang (tuỳ chọn)

Nếu có `yolo-al-data-YYYYMMDD.zip`, giải nén **vào trong** thư mục `yolo-al`
**trước khi** chạy `install.ps1`:

```powershell
Expand-Archive .\yolo-al-data-YYYYMMDD.zip -DestinationPath .\yolo-al -Force
```

Gói này chứa `var/` — ảnh gốc, `meta.db`, `queue.db`, và PCA đã freeze.
Không mang sang thì máy mới bắt đầu từ số 0: phải thu lại ≥2000 frame rồi fit PCA.

## 3. Chạy

```powershell
.\start_all.ps1
```

Mở 3 cửa sổ: thu mẫu · worker · bảng theo dõi. Thêm `-Review` để mở luôn Review UI.
Script tự nhận ra `.venv` và dùng đúng Python trong đó.

## 4. Kiểm tra camera

`.env` đang để `CAMERAS=cam01=0` — webcam đầu tiên. Máy mới có thể đánh số khác:

```powershell
python scripts\startup.py --phase 1
```

Mở http://127.0.0.1:8000. Không thấy hình thì đổi `CAMERAS=cam01=1` (hoặc 2) rồi thử lại.

### Không có camera? Test bằng video

Khi môi trường không có webcam, bỏ một file video vào thư mục `samples/` rồi chạy
với cờ `--video` (không cần sửa `.env`):

```powershell
python scripts\startup.py --phase 1 --video samples\test.mp4          # chỉ realtime
python scripts\startup.py --phase 2 --video samples\test.mp4 --fps 25  # realtime + thu mẫu
```

Mặc định video **chạy 1 lần rồi dừng**; thêm `--loop` để lặp lại, `--fps N` để giới hạn
tốc độ cho giống camera thật. Chi tiết xem `samples/README.md`.

## Không có GPU thì sao

Chạy được, nhưng FPS thấp và Phase 1 sẽ báo "CHƯA ĐẠT". Sửa `.env`:

```
DEVICE=cpu
HALF=false
TARGET_FPS=10
```

Phase 3 (worker) dùng embedder histogram màu, vốn không cần GPU — phần này không chậm đi.

## Lỗi hay gặp

| Triệu chứng | Nguyên nhân |
|---|---|
| `cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope Process Bypass -Force` |
| `Unexpected token '}'` trong file .ps1 | File bị lưu lại thành UTF-8 **không BOM**. Các file .ps1 ở đây cố ý viết không dấu + có BOM; đừng sửa rồi lưu bằng editor đặt mặc định UTF-8 no-BOM |
| `ConnectionError 10061 ... 6379` | `.env` để `QUEUE_BACKEND=redis`. Đổi về `sqlite` |
| `input_dim ... != 512` | Ép sai embedder. PCA đã freeze bằng `hist_v1`; để `--phase 3` tự chọn |
| Bảng theo dõi báo `0/5,000` | Hạn mức tính theo **ngày UTC**, reset lúc 7 giờ sáng giờ VN |
| `unrecognized arguments` | Copy nhầm cả phần chú thích phía sau lệnh |

## Tài liệu khác

- `README.md` — thiết kế: 7 ràng buộc và chỗ thực thi bằng code
- `scripts/README.md` — từng script làm gì, chạy từng phase
- `.env` — mọi ngưỡng, kèm chú thích vì sao chọn số đó
