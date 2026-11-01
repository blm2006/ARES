# scripts/

## Chạy nhanh

```powershell
.\start_all.ps1          # kiểm tra điều kiện + mở 3 cửa sổ: thu mẫu / worker / bảng theo dõi
.\start_all.ps1 -Review  # thêm cửa sổ Review UI
.\start_all.ps1 -Check   # chỉ kiểm tra, không chạy gì
```

## Chạy từng phase

```powershell
python scripts\startup.py --check              # điều kiện: DB, queue, weights, PCA, số frame
python scripts\startup.py --phase 1            # realtime, KHÔNG thu mẫu — nghiệm thu FPS
python scripts\startup.py --phase 2            # realtime + collector (ghi ảnh, đẩy queue)
python scripts\startup.py --phase 3            # worker: embed -> dedup -> review_queue
python scripts\startup.py --phase 4            # Review UI, http://127.0.0.1:8010
python scripts\startup.py --phase 5 --notes "lô đầu"
python scripts\startup.py --phase 6 --dataset v001 --ablation
python scripts\startup.py --phase 7            # status | --promote v002 | --rollback
```

Phase 2 và Phase 3 chạy **song song** được: SQLite ở chế độ WAL, và hai bên đi qua
queue chứ không tranh file. Không cần tách ảnh ra thư mục khác.

## Theo dõi

```powershell
python -m scripts.status_monitor          # làm mới 5 giây/lần
python -m scripts.status_monitor --once   # in một lần
```

Bảng này đọc **trạng thái thật** từ `meta.db` + `queue.db` + `var/artifacts/`:
số frame đã ghi, hạn mức ngày còn bao nhiêu, hàng đợi worker, số mẫu chờ duyệt,
dataset/model đã có — và một dòng "việc tiếp theo".

## Các script còn lại

| Script | Dùng khi |
|---|---|
| `init_db.py` | tạo schema + thư mục, chạy một lần |
| `fit_pca.py` | fit PCA **một lần rồi freeze**. Phase 3 tự gọi khi chưa có file |
| `governance.py` | xoá ảnh quá TTL, xuất audit log |
| `requeue_orphans.py` | frame đã ghi xuống đĩa nhưng mất message queue — đẩy lại cho Phase 3 |
| `dev_quick_start.sh` | chạy test rồi bật Phase 1 (Linux/WSL) |

## Vì sao không còn rotate_frames / auto_pca_trigger / fit_pca_streaming

Ba script đó tách ảnh ra `var/batches/` rồi fit PCA từ đó. Chúng đã bị xoá vì:

1. `rotate_frames.py` quét `var/images/cam01/*.jpg`, nhưng ảnh nằm ở
   `var/images/cam01/<ngày>/<giờ>/*.jpg`. Nó không bao giờ tìm thấy file nào —
   `var/batches/` luôn rỗng, nên `auto_pca_trigger` đứng mãi ở `0/2000`.
2. Kể cả sửa đúng glob thì **di chuyển ảnh vẫn sai**: `frames.path` trong DB và
   `storage.read_image()` của worker đều trỏ vào `var/images/...`. Ảnh đi chỗ khác
   là Phase 3 báo "ảnh không tồn tại, ack bỏ" cho toàn bộ hàng đợi.
3. `fit_pca_streaming.py` đọc `results[0].features` — thuộc tính không tồn tại trong
   ultralytics, nên `total_frames` luôn bằng 0 và hàm luôn trả về False. Nếu có chạy
   được thì nó pickle một object `sklearn.PCA`, trong khi worker đòi `FrozenPCA` →
   `TypeError`, và nó ghi đè `pca_v1.pkl` vượt mặt cơ chế freeze.

Việc mà lớp đó định làm — tách "đang thu" khỏi "đang xử lý" — **queue đã làm rồi**.
Phase 2 ghi ảnh + đẩy message; Phase 3 lấy message ra xử lý. Không file nào bị khoá.
