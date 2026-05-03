# 📖 Hướng Dẫn Sử Dụng & Giải Thích Workflow

> **Illegal Parking Detection System** — Tài liệu chi tiết về cách sử dụng source code và cơ chế hoạt động

---

## 📋 Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Cài đặt & Chạy chương trình](#2-cài-đặt--chạy-chương-trình)
3. [Cấu trúc source code](#3-cấu-trúc-source-code)
4. [Workflow tổng thể](#4-workflow-tổng-thể)
5. [Giải thích chi tiết từng module](#5-giải-thích-chi-tiết-từng-module)
6. [Luồng dữ liệu giữa các module](#6-luồng-dữ-liệu-giữa-các-module)
7. [Cấu hình hệ thống](#7-cấu-hình-hệ-thống)
8. [Tương tác người dùng](#8-tương-tác-người-dùng)
9. [Kết quả đầu ra](#9-kết-quả-đầu-ra)
10. [Xử lý sự cố](#10-xử-lý-sự-cố)

---

## 1. Tổng quan hệ thống

Hệ thống phát hiện đậu xe trái phép hoạt động theo **pipeline tuần tự 4 bước**:

```
Camera/Video ──▶ Detection (YOLOv8) ──▶ Tracking (DeepSORT) ──▶ Parking Logic ──▶ Alert & Hiển thị
```

**Nguyên lý cốt lõi:** Hệ thống đọc từng frame video, nhận diện phương tiện bằng AI, theo dõi từng xe qua các frame liên tiếp bằng thuật toán tracking, sau đó kiểm tra xem xe có đứng yên trong vùng cấm quá lâu hay không để xác định vi phạm.

---

## 2. Cài đặt & Chạy chương trình

### 2.1 Cài đặt

```bash
# Clone và cài đặt
git clone <repo-url>
cd illegal-parking-detection-ITS

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Cài thư viện
pip install -r requirements.txt
```

**Thư viện cần thiết:**

| Thư viện | Vai trò |
|---|---|
| `opencv-python` ≥ 4.8.0 | Đọc video, vẽ hình, hiển thị |
| `ultralytics` ≥ 8.0.0 | Model YOLOv8 nhận diện vật thể |
| `deep-sort-realtime` ≥ 1.3.2 | Thuật toán tracking DeepSORT |
| `numpy` ≥ 1.24.0 | Tính toán mảng số |

### 2.2 Chạy chương trình

```bash
# Cơ bản (dùng video mặc định data/video.mp4)
python src/main.py

# Tuỳ chọn nâng cao
python src/main.py --source data/video.mp4   # Video cụ thể
python src/main.py --source 0                # Webcam
python src/main.py --threshold 10            # Ngưỡng 10 giây
python src/main.py --model yolov8s.pt        # Model lớn hơn
python src/main.py --confidence 0.6          # Ngưỡng confidence
python src/main.py --draw-roi                # Vẽ vùng cấm trước
python src/main.py --no-save                 # Không lưu video đầu ra

# Kết hợp
python src/main.py --source data/video.mp4 --threshold 8 --draw-roi
```

### 2.3 Phím tắt khi chạy

| Phím | Chức năng |
|------|-----------|
| `q` | Thoát chương trình |
| `r` | Vẽ lại vùng cấm đậu xe (ROI) |
| `p` | Tạm dừng / Tiếp tục |
| `s` | Chụp screenshot |

---

## 3. Cấu trúc source code

```
illegal-parking-detection-ITS/
├── config/
│   └── config.py                  ← Tất cả tham số cấu hình
├── data/
│   ├── video.mp4                  ← Video đầu vào
│   ├── output.mp4                 ← Video đầu ra (tự tạo)
│   └── violations.log             ← Log vi phạm (tự tạo)
├── src/
│   ├── main.py                    ← Entry point, điều phối pipeline
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py       ← Module 1: Nhận diện xe (YOLOv8)
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py             ← Module 2: Theo dõi xe (DeepSORT)
│   ├── parking/
│   │   ├── __init__.py
│   │   └── parking_detector.py    ← Module 3: Logic phát hiện vi phạm
│   └── alert/
│       ├── __init__.py
│       └── alert_manager.py       ← Module 4: Hiển thị & ghi log
├── requirements.txt
└── yolov8n.pt                     ← Model weights (tự download)
```

**Thiết kế module hóa:** Mỗi module đảm nhiệm đúng 1 chức năng, giao tiếp với nhau qua các dictionary Python chuẩn hóa. Điều này giúp dễ bảo trì, test, và thay thế từng thành phần.

---

## 4. Workflow tổng thể

### 4.1 Sơ đồ luồng xử lý chính

```
┌──────────────────────────────────────────────────────────────────┐
│                        main.py (Orchestrator)                    │
│                                                                  │
│  1. Parse arguments & Load config                                │
│  2. Khởi tạo 4 module: Detector, Tracker, ParkingDetector, Alert│
│  3. Mở video source (file hoặc webcam)                          │
│  4. [Tuỳ chọn] Vẽ ROI bằng chuột                               │
│  5. VÒNG LẶP CHÍNH (mỗi frame):                                │
│     ┌─────────────────────────────────────────────────────┐      │
│     │  Frame ──▶ YOLO detect ──▶ DeepSORT track           │      │
│     │        ──▶ Parking check ──▶ Alert draw ──▶ Display  │      │
│     └─────────────────────────────────────────────────────┘      │
│  6. Cleanup & In thống kê                                        │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Chi tiết vòng lặp chính (mỗi frame)

```
Frame (numpy array BGR)
    │
    ▼
[Step A] detector.detect(frame)
    │   → YOLOv8 xử lý frame
    │   → Lọc chỉ giữ xe (car, motorcycle, bus, truck)
    │   → Trả về: list[{bbox, confidence, class_id, class_name}]
    │
    ▼
[Step B] tracker.update(detections, frame)
    │   → Chuyển detection sang format DeepSORT
    │   → Tạo embedding từ ảnh crop của mỗi xe
    │   → DeepSORT ghép detection với track cũ (Hungarian + Kalman)
    │   → Trả về: list[{track_id, bbox, class_name, confirmed}]
    │
    ▼
[Step C] parking_detector.update(tracked_vehicles)
    │   → Kiểm tra từng xe: trong ROI? đứng yên? bao lâu?
    │   → Cập nhật vehicle_history
    │   → Gán status: normal / warning / violation
    │   → Trả về: list[{...vehicle, in_roi, is_violation, duration, status}]
    │
    ▼
[Step D] alert_manager.draw_*(frame, ...)
    │   → Vẽ vùng cấm (ROI) lên frame
    │   → Vẽ bbox màu theo status (xanh/cam/đỏ)
    │   → Vẽ "ILLEGAL PARKING" cho xe vi phạm
    │   → Vẽ bảng thống kê (FPS, số xe, vi phạm)
    │   → Ghi log vi phạm vào file
    │
    ▼
Hiển thị frame (cv2.imshow) + Lưu video (cv2.VideoWriter)
```

---

## 5. Giải thích chi tiết từng module

### 5.1 Module Detection — `src/detection/yolo_detector.py`

**Class:** `YOLODetector`

**Vai trò:** Nhận diện phương tiện trong mỗi frame bằng YOLOv8.

**Cách hoạt động:**

1. **Khởi tạo (`__init__`):** Load model YOLOv8 pre-trained trên COCO dataset (80 classes). Model `yolov8n.pt` sẽ tự download nếu chưa có.

2. **Phát hiện (`detect`):**
   - Đưa frame vào model YOLOv8 → model trả về tất cả vật thể phát hiện được
   - **Lọc class:** Chỉ giữ 4 loại phương tiện: Car (2), Motorcycle (3), Bus (5), Truck (7)
   - **Lọc confidence:** Bỏ detection có confidence < ngưỡng (mặc định 0.5)
   - Trả về danh sách detection

**Format dữ liệu đầu ra:**
```python
{
    'bbox': [x1, y1, x2, y2],   # Toạ độ góc trên-trái và dưới-phải (pixel)
    'confidence': 0.92,          # Độ tin cậy (0.0 → 1.0)
    'class_id': 2,               # ID class trong COCO
    'class_name': 'Car'          # Tên loại xe
}
```

**Tại sao dùng YOLOv8?** YOLO (You Only Look Once) là kiến trúc detection 1 giai đoạn — chỉ cần 1 lần forward pass qua neural network để phát hiện tất cả vật thể, rất nhanh cho real-time.

---

### 5.2 Module Tracking — `src/tracking/tracker.py`

**Class:** `VehicleTracker`

**Vai trò:** Gán ID duy nhất cho mỗi xe và theo dõi qua các frame liên tiếp.

**Tại sao cần tracking?** Detection chỉ cho biết "có xe ở đây" trong 1 frame. Tracking liên kết các detection giữa các frame để biết "xe A ở frame 1 chính là xe A ở frame 2" — cần thiết để tính thời gian đậu.

**Cách hoạt động:**

1. **Khởi tạo (`__init__`):** Tạo DeepSORT tracker với `embedder=None` (không dùng model pretrained nặng cho appearance feature, thay vào đó tự tạo embedding nhẹ).

2. **Tạo embedding (`_generate_embeddings`):**
   - Crop ảnh xe từ frame theo bbox
   - Resize về 32×32 pixel
   - Flatten thành vector 1 chiều → giảm chiều xuống 128 → L2 normalize
   - Đây là feature đơn giản giúp DeepSORT phân biệt các xe khác nhau

3. **Cập nhật tracker (`update`):**
   - Chuyển bbox từ format `[x1,y1,x2,y2]` sang `[x1,y1,w,h]` (DeepSORT yêu cầu)
   - Gọi `tracker.update_tracks()` với detections + embeddings
   - DeepSORT nội bộ thực hiện:
     - **Kalman Filter:** Dự đoán vị trí xe ở frame tiếp theo
     - **Hungarian Algorithm:** Ghép nối detection mới với track cũ (tối ưu hóa)
     - **IOU + Appearance matching:** Dùng cả overlap và visual similarity
   - Chỉ trả về track đã **confirmed** (xuất hiện ≥ `n_init` frame liên tiếp)

**Format dữ liệu đầu ra:**
```python
{
    'track_id': 1,               # ID duy nhất, không đổi qua các frame
    'bbox': [x1, y1, x2, y2],   # Bbox đã được Kalman Filter smooth
    'class_name': 'Car',
    'confirmed': True
}
```

**Các tham số quan trọng:**
- `max_age=30`: Giữ track sống 30 frame dù không detect được (xử lý occlusion tạm)
- `n_init=3`: Track phải được detect 3 frame liên tiếp mới xác nhận (lọc false positive)

---

### 5.3 Module Parking — `src/parking/parking_detector.py`

**Class:** `ParkingDetector`

**Vai trò:** Xác định xe nào đang đậu trái phép dựa trên 3 điều kiện.

**Logic phát hiện vi phạm — 3 điều kiện đồng thời:**

```
Điều kiện 1: Xe nằm TRONG vùng cấm (ROI)?           ──▶ ✅
Điều kiện 2: Xe ĐỨNG YÊN (di chuyển < 15px)?         ──▶ ✅
Điều kiện 3: Đứng yên QUÁ LÂU (> 5 giây)?            ──▶ ✅ = VI PHẠM!
```

**Cách hoạt động chi tiết (`update`):**

```
Với mỗi xe tracked:
│
├─ Tính tâm bbox: cx = (x1+x2)/2, cy = (y1+y2)/2
│
├─ Kiểm tra ROI: cx,cy có nằm trong (rx1,ry1)-(rx2,ry2)?
│   ├─ KHÔNG trong ROI → status = "normal", reset timer
│   └─ CÓ trong ROI:
│       ├─ Tính displacement = √((cx-old_cx)² + (cy-old_cy)²)
│       ├─ displacement ≥ 15px → xe đang di chuyển, reset timer
│       └─ displacement < 15px → xe đứng yên
│           ├─ Thời gian đứng yên < 5s → status = "warning"
│           └─ Thời gian đứng yên ≥ 5s → status = "violation" 🚨
```

**Quản lý lịch sử (`vehicle_history`):**

Mỗi track_id có 1 entry trong dictionary:
```python
vehicle_history[track_id] = {
    'first_seen': 1714600000.0,      # Timestamp lần đầu thấy
    'last_position': (400, 350),     # Vị trí tâm gần nhất
    'stationary_since': 1714600002,  # Bắt đầu đứng yên từ lúc nào
    'is_violation': False,           # Đã vi phạm chưa
    'in_roi': True                   # Có đang trong vùng cấm
}
```

**Dọn dẹp:** Xe không còn được track sẽ bị xóa khỏi `vehicle_history` để tránh memory leak.

**3 trạng thái xe:**

| Status | Điều kiện | Màu |
|--------|-----------|-----|
| `normal` | Ngoài ROI hoặc đang di chuyển | 🟢 Xanh |
| `warning` | Trong ROI, đứng yên, chưa quá ngưỡng | 🟠 Cam |
| `violation` | Trong ROI, đứng yên, quá ngưỡng thời gian | 🔴 Đỏ |

---

### 5.4 Module Alert — `src/alert/alert_manager.py`

**Class:** `AlertManager`

**Vai trò:** Hiển thị trực quan kết quả lên video và ghi log vi phạm.

**Các phương thức vẽ:**

| Phương thức | Vẽ gì |
|---|---|
| `draw_roi()` | Vùng cấm: hình chữ nhật cyan + overlay mờ + label "NO PARKING ZONE" |
| `draw_vehicles()` | Bbox màu theo status + ID + class + thời gian + alert "ILLEGAL PARKING" |
| `draw_stats()` | Panel thống kê: FPS, số xe, số xe trong zone, số vi phạm |
| `draw_instructions()` | Hướng dẫn phím tắt ở góc dưới |

**Cơ chế log vi phạm (`_log_violation`):**
- Mỗi track_id chỉ được log **1 lần** (dùng `logged_violations` set để track)
- Format: `[2026-05-02 01:32:52] VIOLATION - Track ID: 8, Vehicle: Car, Duration: 5.0s`
- Ghi vào file `data/violations.log`

---

### 5.5 Entry Point — `src/main.py`

**Vai trò:** Điều phối (orchestrate) toàn bộ pipeline.

**Luồng thực thi tuần tự:**

```
1. parse_args()
   → Đọc tham số dòng lệnh (--source, --threshold, --model, ...)

2. Khởi tạo 4 module
   → YOLODetector, VehicleTracker, ParkingDetector, AlertManager

3. Mở video source
   → cv2.VideoCapture(source) — file MP4 hoặc webcam index

4. Setup VideoWriter (nếu --no-save không bật)
   → Ghi video đầu ra với cùng resolution & FPS

5. [Tuỳ chọn] Vẽ ROI (nếu --draw-roi)
   → select_roi() — callback chuột cho user vẽ hình chữ nhật

6. VÒNG LẶP CHÍNH (while True):
   │  ret, frame = cap.read()
   │  detections = detector.detect(frame)
   │  tracked = tracker.update(detections, frame)
   │  vehicles = parking_detector.update(tracked)
   │  frame = alert_manager.draw_roi(frame, roi)
   │  frame = alert_manager.draw_vehicles(frame, vehicles)
   │  frame = alert_manager.draw_stats(frame, fps, stats)
   │  cv2.imshow(frame) + writer.write(frame)
   │  Handle keyboard (q/r/p/s)
   └──▶ Lặp lại cho frame tiếp theo

7. Cleanup
   → Release video, destroy windows, in thống kê cuối
```

**Tính năng ROI bằng chuột (`select_roi`):**
- Dùng `cv2.setMouseCallback()` để bắt sự kiện chuột
- User click-drag để vẽ hình chữ nhật
- ENTER = xác nhận, ESC = huỷ
- ROI mới được truyền vào `parking_detector.set_roi()`

---

## 6. Luồng dữ liệu giữa các module

```
                    ┌─────────────────────┐
                    │    config/config.py  │
                    │  (Tham số toàn cục)  │
                    └──┬──┬──┬──┬─────────┘
                       │  │  │  │
          ┌────────────┘  │  │  └────────────┐
          ▼               ▼  ▼               ▼
  ┌───────────────┐ ┌──────────┐ ┌───────────────┐ ┌────────────┐
  │ YOLODetector  │ │ Vehicle  │ │   Parking     │ │   Alert    │
  │               │ │ Tracker  │ │   Detector    │ │   Manager  │
  │ Input:        │ │          │ │               │ │            │
  │  frame (BGR)  │ │ Input:   │ │ Input:        │ │ Input:     │
  │               │ │  dets +  │ │  tracked      │ │  vehicles  │
  │ Output:       │ │  frame   │ │  vehicles     │ │  + status  │
  │  detections[] │ │          │ │               │ │            │
  │  {bbox,conf,  │ │ Output:  │ │ Output:       │ │ Output:    │
  │   class}      │ │  tracked │ │  vehicles +   │ │  annotated │
  │               │ │  [{id,   │ │  {in_roi,     │ │  frame     │
  └───────┬───────┘ │   bbox,  │ │   violation,  │ │            │
          │         │   class}]│ │   duration,   │ │  Side:     │
          │         └────┬─────┘ │   status}     │ │  log file  │
          │              │       └──────┬────────┘ └─────┬──────┘
          ▼              ▼              ▼                ▼
       Step A         Step B         Step C           Step D
```

**Tóm tắt data flow qua 1 frame:**

| Bước | Input | Xử lý | Output |
|------|-------|--------|--------|
| A | `frame` (numpy array) | YOLOv8 inference + filter | `list[{bbox, confidence, class_id, class_name}]` |
| B | `detections` + `frame` | DeepSORT matching + Kalman | `list[{track_id, bbox, class_name, confirmed}]` |
| C | `tracked_vehicles` | ROI check + movement + timer | `list[{..., in_roi, is_violation, duration, status}]` |
| D | `vehicles_with_status` | Draw + log | Annotated frame + log entry |

---

## 7. Cấu hình hệ thống

File `config/config.py` chứa tất cả tham số có thể điều chỉnh:

### Video
| Tham số | Giá trị mặc định | Giải thích |
|---------|-------------------|------------|
| `VIDEO_SOURCE` | `"data/video.mp4"` | Đường dẫn video đầu vào |
| `OUTPUT_VIDEO` | `"data/output.mp4"` | Đường dẫn video đầu ra |
| `SAVE_OUTPUT` | `True` | Bật/tắt lưu video |

### Detection (YOLO)
| Tham số | Giá trị mặc định | Giải thích |
|---------|-------------------|------------|
| `YOLO_MODEL` | `"yolov8n.pt"` | Model YOLOv8 (n=nhanh, x=chính xác) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Ngưỡng tối thiểu để chấp nhận detection |
| `VEHICLE_CLASSES` | `[2, 3, 5, 7]` | COCO class ID của xe |

### Tracking (DeepSORT)
| Tham số | Giá trị mặc định | Giải thích |
|---------|-------------------|------------|
| `MAX_AGE` | `30` | Số frame giữ track khi mất detection |
| `N_INIT` | `3` | Số frame liên tiếp để xác nhận track |
| `MAX_IOU_DISTANCE` | `0.7` | Ngưỡng IOU cho matching |

### Parking Logic
| Tham số | Giá trị mặc định | Giải thích |
|---------|-------------------|------------|
| `NO_PARKING_ZONE` | `(200, 300, 800, 600)` | Toạ độ vùng cấm mặc định |
| `PARKING_TIME_THRESHOLD` | `5.0` | Ngưỡng thời gian vi phạm (giây) |
| `MOVEMENT_THRESHOLD` | `15.0` | Ngưỡng pixel di chuyển tối đa |

### Hiển thị
| Tham số | Giá trị mặc định | Giải thích |
|---------|-------------------|------------|
| `COLOR_NORMAL` | `(0,255,0)` Green | Xe bình thường |
| `COLOR_WARNING` | `(0,165,255)` Orange | Xe cảnh báo |
| `COLOR_VIOLATION` | `(0,0,255)` Red | Xe vi phạm |
| `LOG_FILE` | `"data/violations.log"` | File ghi log |

---

## 8. Tương tác người dùng

### 8.1 Vẽ vùng cấm (ROI)

**Cách 1:** Chạy với `--draw-roi` → vẽ trước khi xử lý video
**Cách 2:** Nhấn `r` khi đang chạy → vẽ lại ROI bất kỳ lúc nào

Thao tác:
1. Click chuột trái + kéo để vẽ hình chữ nhật
2. Nhấn **ENTER** để xác nhận
3. Nhấn **ESC** để huỷ

### 8.2 Tạm dừng/Tiếp tục

Nhấn `p` để pause, nhấn `p` lần nữa để resume. Khi pause, hệ thống chỉ lắng nghe keyboard, không xử lý frame mới.

### 8.3 Chụp screenshot

Nhấn `s` → lưu frame hiện tại thành `data/screenshot_{frame_count}.png`.

---

## 9. Kết quả đầu ra

| Đầu ra | Đường dẫn | Mô tả |
|--------|-----------|-------|
| Video annotated | `data/output.mp4` | Video gốc + bbox + alert + thống kê |
| Log vi phạm | `data/violations.log` | Danh sách vi phạm với timestamp |
| Hiển thị real-time | Cửa sổ OpenCV | Xem trực tiếp khi chạy |

**Format log:**
```
[2026-05-02 01:32:52] VIOLATION - Track ID: 8, Vehicle: Car, Duration: 5.0s
```

---

## 10. Xử lý sự cố

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `ModuleNotFoundError` | Chưa cài thư viện | `pip install -r requirements.txt` |
| `Cannot open video source` | Sai đường dẫn | Kiểm tra file `data/video.mp4` tồn tại |
| FPS thấp | Model nặng / CPU yếu | Dùng `--model yolov8n.pt` hoặc giảm resolution |
| Không detect được xe | Confidence quá cao | `--confidence 0.3` |
| ROI không phù hợp | Toạ độ mặc định sai | Nhấn `r` để vẽ lại hoặc sửa `config.py` |
| Tracking ID nhảy | Xe bị che khuất lâu | Tăng `MAX_AGE` trong config |

---

> 📝 **Ghi chú:** Tài liệu này mô tả source code tại thời điểm viết. Nếu có thay đổi code, vui lòng cập nhật tài liệu tương ứng.
