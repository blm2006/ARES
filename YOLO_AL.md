# yolo-al — person detection + tracking + active learning loop

Implement của **Workflow v2**. Chạy local, offline, Python 3.11, Windows + NVIDIA.

> **ARES** (BEMS theo hiện diện: đèn từng vùng + điều hòa dự đoán, học chủ động, dashboard) nằm trong `ares/` —
> xem **[ARES_UPGRADE.md](ARES_UPGRADE.md)**. Chạy thử không cần phần cứng: `.\start_ares.ps1 -Demo`.

Bài toán là **person detection**, không phải face. Face là pipeline thứ hai nối
sau person crop nếu về sau cần — không phải cùng một model.

---

## 7 ràng buộc được thực thi bằng code, không phải bằng comment

| # | Ràng buộc | Chỗ thực thi | Test chứng minh |
|---|---|---|---|
| 1 | Realtime và learning là hai process, giao tiếp qua queue **có persistence** | `common/queue.py` (Redis Streams) | `test_queue_survives_restart` |
| 2 | **Không đường nào** đưa nhãn model tự sinh vào dataset | `dataset/builder.py` chỉ đọc `action in (approve, fix)` | `test_no_autolabel_path.py` (5 test) |
| 3 | Ảnh lưu là **full frame gốc**, không resize/crop/normalize | `collector/writer.py` | `test_original_frame_written_unmodified` |
| 4 | Queue payload chỉ path + metadata | `common/schemas.QueueMessage` (`extra="forbid"` + chốt chặn kích thước) | `test_queue_payload.py` (6 test) |
| 5 | Split theo **session** (cam+ngày), không theo frame | `dataset/split.py` | `test_split_leakage.py` (6 test) |
| 6 | PCA **fit một lần rồi freeze**, có version | `worker/pca.py` | `test_pca_freeze.py` (8 test) |
| 7 | Person, không phải face | `realtime/detector.py` (`classes=[0]`) | — |

```
pytest -q     # 73 test, chạy offline: không cần GPU, Redis hay ultralytics
```

---

## Cài đặt

> Máy mới: giải nén gói rồi chạy `.\install.ps1` — xem **[SETUP.md](SETUP.md)**.
> Đóng gói để chuyển máy: `.\package.ps1 -Data`.

### Thủ công

```powershell
# 1. torch CUDA TRƯỚC — nếu không pip sẽ kéo bản CPU và bạn sẽ ngồi debug FPS cả buổi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. Redis (persistence bật sẵn trong compose)
docker compose up -d

# 3. Cấu hình
copy .env.example .env      # sửa CAMERAS, DEVICE
python -m scripts.init_db
```

---

## Làm theo phase. Mỗi phase phải chạy được trước khi sang phase sau.

### Phase 1 — realtime chạy được, FPS **thật** ≥ 25

```powershell
python -m realtime --no-collector
# Dashboard: http://127.0.0.1:8000
```

Nghiệm thu: bbox + track ID hiển thị, và dòng tổng kết in ra `FPS thật ... — ĐẠT`.

> Dashboard cố tình hiển thị **FPS đo được**, không phải FPS khai báo của camera.
> v1 hiện 30 FPS trong khi thực chạy 14.8 — mất ~50% frame. Frame bị drop làm
> tracker mất ID, mà "ID nhảy" lại là tiêu chí active learning, nên lỗi hạ tầng
> sẽ **tự nhân bản thành hàng nghìn mẫu rác**. Chưa đạt FPS thì đừng bật Phase 2.
>
> Nếu chưa đạt: xem `frame_drop_rate` và `inference_ms` trên dashboard. Drop cao
> + inference thấp = nghẽn decode (thử GStreamer / hardware decode). Inference cao
> = model quá nặng (`yolov8n`, `HALF=true`, hoặc export TensorRT).

### Phase 2 — collector + queue + backpressure

```powershell
python -m realtime          # đã bật collector
```

Nghiệm thu: ảnh gốc xuống `var/images/cam01/YYYY-MM-DD/HH/`, queue chỉ có
metadata, và **FPS không giảm** so với Phase 1.

Ba nguồn lấy mẫu chạy song song, mỗi nguồn có hạn mức ngày riêng:

| Nguồn | Quota | Bắt được gì |
|---|---|---|
| A · uncertainty | 40% | Ca mơ hồ — chọn theo **phân vị** (top 5% của cửa sổ 24h), không theo band cứng 0.3–0.6 |
| B · failure mining | 40% | **Bỏ sót người** (track gap), ID switch, track chết sớm |
| C · random | 20% | Lỗi ngoài giả định + **nhóm đối chứng** |

Quota nằm trong DB, không trong RAM: crash rồi chạy lại không lấy mẫu gấp đôi.
Còn có cap theo **giờ** và theo **camera** — nếu không, 80% dataset sẽ là giờ cao
điểm của camera ồn nhất.

> **Dùng hết hạn mức ngày thì collector NGỪNG ghi cho tới 00:00 hôm sau.** Đây là
> thiết kế, không phải lỗi — nhưng nhìn từ ngoài thì giống hệt "hệ thống chết".
> `python -m scripts.status_monitor` hiện phần trăm đã dùng của từng nguồn, nên
> luôn xem bảng đó trước khi đi sửa camera.

Backpressure: `depth > N1` → cắt nguồn A trước; `depth > N2` hoặc `disk < 10%` →
dừng hẳn. Đây là điều kiện để câu "learning không ảnh hưởng realtime" là thật.

### Phase 3 — worker: embedding → PCA → dedup → pre-annotation

```powershell
python scripts\startup.py --phase 3      # tự fit PCA nếu chưa có, rồi chạy worker
```

> `--phase 3` chọn embedder **đúng bằng embedder mà PCA đã freeze** (đọc từ
> `var/artifacts/<pca_version>.json`). Ép sai embedder là worker chết ngay message
> đầu với `ValueError: input_dim ... != 512`, vì vector thô đổi số chiều còn phép
> chiếu thì đã đóng băng.
>
> `DEDUP_THETA` phải khớp embedder: với `hist_v1` trên camera cố định, các frame
> gần nhau có cosine ~0.99, nên theta 0.94 vứt >95% dữ liệu. Số trong `.env` đã đo
> trên dữ liệu thật — xem chú thích ngay tại chỗ đó.

`fit_pca` cố tình tách khỏi worker. Nếu worker tự fit, mỗi lần khởi động lại với
dữ liệu khác sẽ ra một phép chiếu khác, và faiss index cũ thành rác **mà không có
lỗi nào nổ ra** — dạng hỏng tệ nhất.

- faiss index trên **toàn bộ dataset**, không chỉ batch hiện tại (và bắt cả trùng
  trong cùng một batch).
- `cosine > θ` → discard nhưng **tăng counter**. Counter tăng nhanh = dataset bão
  hoà → nên nới tiêu chí hoặc thêm camera: `python -c "from worker.dedup import saturation_report; print(saturation_report())"`
- Nguồn B dùng θ **lỏng hơn**: lỗi hay lặp lại nhưng vẫn cần học.
- Core-set selection chạy sau PCA — đây chính là "khung hình đại diện" của v1,
  đặt đúng chỗ (ở 2.1 nó phụ thuộc vòng vì cần embedding).

### Phase 4 — review UI

```powershell
python -m review.app
# http://127.0.0.1:8010
```

| Phím | Hành động |
|---|---|
| `A` | Approve (fast lane, ~1 giây/frame) |
| `S` | Lưu sửa |
| `R` | Reject, chọn lý do |
| `→` | Bỏ qua |
| kéo chuột | thêm box · `Del` xoá · `U` hoàn tác |

Box **nét đứt** là gợi ý nội suy từ track lân cận (t−1/t+1). Approve **không** lấy
nó — phải xác nhận bằng Lưu sửa. Đó là ranh giới giữa "gợi ý" và "ground truth".

**Agreement rate** = % pre-annotation được approve không sửa. Tính riêng theo
nguồn: agreement trên nguồn B thấp là bình thường (toàn ca khó); con số đáng tin
để đánh giá model là **agreement trên nguồn random**.

### Phase 5 — dataset builder

```powershell
python -m dataset.builder --notes "batch dau tien"
```

- Lưu **ảnh gốc** + label YOLO `.txt`. Resize/normalize/augment là việc của
  dataloader lúc train. Nướng 640×640 vào dữ liệu thì sau này đổi sang 960 là mất
  thông tin vĩnh viễn.
- Split theo session. Frame liền nhau trong cùng track giống nhau gần tuyệt đối →
  chia ngẫu nhiên theo frame là leakage, mAP bị thổi phồng.
- **Golden set** đóng băng vĩnh viễn (`golden_sessions.json`) — `freeze()` gọi lần
  hai sẽ raise. **Rolling set** = 30 ngày gần nhất, để bắt drift.
- `manifest.json` ghi: pca version, model nào sinh pre-annotation, ai duyệt, phân
  bố nguồn A/B/C, git rev.

### Phase 6 — train + validate + registry

```powershell
python -m training.train --dataset v001 --ablation
```

- Fine-tune **luôn từ checkpoint gốc**, không chồng lên model đã fine-tune trước
  → tránh catastrophic forgetting. Mix 70% dữ liệu cũ / 30% mới.
- Validation có **gate**, không chỉ in số:
  1. `per_camera` — không camera nào tụt > 2%. mAP tổng tăng mà một camera tụt là
     chuyện rất thường, và người dùng chỉ nhìn camera của họ.
  2. `golden` — không được tụt so với model trước.
  3. `beats_random_baseline` — phải thắng model train chỉ bằng `source='random'`.
     **Nếu trượt gate này thì toàn bộ nguồn A đang là chi phí review vô ích** —
     báo cáo nói thẳng điều đó thay vì im lặng.
- Metric chính là **recall@IoU0.5**, không phải mAP tổng: đếm người thì bỏ sót đắt
  hơn lệch box vài pixel.

Trigger: `≥ N mẫu mới **đã duyệt**` HOẶC rolling mAP tụt HOẶC agreement tụt.
Không train theo lịch cứng khi không có dữ liệu mới.

### Phase 7 — deploy: shadow → canary → full

```powershell
python -m deploy.controller status
python -m deploy.controller promote --version v002
python -m deploy.controller watch          # auto rollback
```

Shadow chạy song song chỉ log (mặc định mỗi 3 frame một lần, để không ăn GPU của
production). Canary 1 camera. Auto rollback theo telemetry thật: `people_count`
drift, `id_switch_per_min` tăng, FPS tụt. Model thắng trên test set vẫn có thể tệ
hơn trên camera thật.

`check_promotion` phân biệt rõ **"chưa đủ telemetry"** và **"đạt"** — không tự
nhảy bước khi chưa có dữ liệu.

---

## Governance

```powershell
python -m scripts.governance ttl --dry-run    # xoá ảnh quá 90 ngày
python -m scripts.governance audit --days 7
```

Ảnh người là dữ liệu cá nhân (NĐ 13/2023/NĐ-CP). TTL + audit log có sẵn; biển
thông báo tại điểm lắp camera và phân quyền truy cập là việc ngoài code.

---

## Cấu trúc

```
common/      config, db, storage, queue, metrics, schemas
realtime/    camera (ring buffer + drop policy), detector, tracker, pipeline, dashboard
collector/   sampling router (A/B/C + quota), uncertainty, failure_mining, writer, backpressure
worker/      embedding, pca (freeze), dedup (faiss), coreset, preannotate, consumer
review/      FastAPI UI + agreement rate
dataset/     builder, split theo session, golden set, manifest
training/    trigger, train, validate (gates), registry
deploy/      shadow, rollback, controller
tests/       73 test, chạy offline
```

## Những chỗ cố tình **không** làm

- **MinIO**: `common/storage.py` có interface đầy đủ nhưng `MinioStorage.__init__`
  raise `NotImplementedError`. Spec nói "interface hoá để sau đổi được", không nói
  phải chạy MinIO ngay — và một stub giả vờ hoạt động còn tệ hơn không có.
- **TensorRT export**: `HALF=true` + ultralytics là đủ cho 1080p/25FPS trên GPU
  rời. Export TRT khi Phase 1 đo được là chưa đạt, không phải trước đó.
- **`rolling_map_now` trong trigger**: cần một job eval định kỳ trên model
  production đang chạy. Chỗ nối đã có (`training/trigger.py`), job thì chưa —
  hiện trigger dựa vào số mẫu mới và agreement rate.
- **Auth cho review UI**: đang bind `127.0.0.1`. Mở ra mạng LAN thì phải thêm
  auth trước, vì endpoint `/api/image/{id}` trả ảnh người.
