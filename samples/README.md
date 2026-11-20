# samples — video để test khi không có camera

Bỏ file video vào đây (`.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`, `.m4v`) rồi
chạy pipeline với cờ `--video` thay cho camera thật:

```powershell
# Phase 1 — chỉ realtime (bbox + FPS), KHÔNG thu mẫu
python scripts\startup.py --phase 1 --video samples\test.mp4

# Phase 2 — realtime + collector (ghi ảnh, đẩy queue) từ video
python scripts\startup.py --phase 2 --video samples\test.mp4 --fps 25
```

Hoặc gọi thẳng module realtime:

```powershell
python -m realtime --video samples\test.mp4            # chạy 1 lần rồi DỪNG, đọc nhanh nhất
python -m realtime --video samples\test.mp4 --fps 25   # giới hạn ~25 fps cho giống realtime
python -m realtime --video samples\test.mp4 --loop     # lặp lại video khi hết
```

Ghi chú:
- Video mặc định **chạy 1 lần rồi dừng**; thêm `--loop` để lặp lại khi hết.
- `--fps N` giới hạn tốc độ đọc cho giống camera thật. Bỏ trống = đọc nhanh
  nhất có thể (đo throughput/ FPS thật của model).
- Reader cho video **không drop frame** (khác camera realtime), nên kết quả
  test không bị sai vì mất frame.
- Không cần sửa `.env`. Nếu muốn cố định dùng video, có thể đặt luôn
  `CAMERAS=cam01=samples\test.mp4` trong `.env`.

## Đọc HẾT video trong một thư mục (theo thứ tự)

Trỏ `--video` vào **thư mục** thay vì 1 file: pipeline đọc lần lượt tất cả video
trong đó, sắp theo tên (thứ tự tự nhiên: `vid2` trước `vid10`), hết cái này sang
cái kế:

```powershell
python -m realtime --video samples                    # đọc hết, hết thì DỪNG
python -m realtime --video samples --loop              # hết thì quay lại từ đầu
python scripts\startup.py --phase 2 --video samples --fps 25
```

Chỉ cần thả thêm file `.mp4/.avi/.mkv/.mov/.webm` vào `samples\` là lần chạy sau
tự có, không phải sửa lệnh.
