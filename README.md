# 🚗 Illegal Parking Detection System

> **ITS Project** — Hệ thống phát hiện đậu xe trái phép sử dụng Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://docs.ultralytics.com/)
[![DeepSORT](https://img.shields.io/badge/DeepSORT-Tracking-orange.svg)](https://github.com/levan92/deep_sort_realtime)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Giải thích chi tiết từng module](#-giải-thích-chi-tiết-từng-module)
- [Cấu hình](#-cấu-hình)
- [Kết quả đầu ra](#-kết-quả-đầu-ra)
- [Xử lý sự cố](#-xử-lý-sự-cố)
- [License](#-license)

---

## 🎯 Tổng quan

Hệ thống phát hiện đậu xe trái phép theo thời gian thực sử dụng:

- **YOLOv8** để nhận diện phương tiện (ô tô, xe máy, xe buýt, xe tải)
- **DeepSORT** để theo dõi từng phương tiện với ID duy nhất
- **Parking Logic** để xác định vi phạm dựa trên vị trí và thời gian
- **OpenCV** để hiển thị kết quả trực quan

### Tính năng chính

| Tính năng | Mô tả |
|---|---|
| 🔍 Nhận diện phương tiện | YOLOv8 phát hiện 4 loại xe với confidence score |
| 🏷️ Theo dõi ID | DeepSORT gán ID duy nhất cho mỗi xe qua các frame |
| ⛔ Phát hiện đậu trái phép | Xác định xe đứng yên trong vùng cấm quá thời gian |
| 🔴 Cảnh báo trực quan | Khung đỏ + text "ILLEGAL PARKING" cho xe vi phạm |
| 🖱️ Vẽ vùng cấm bằng chuột | Tương tác trực tiếp trên video |
| 📊 Hiển thị thống kê | FPS, số xe, số vi phạm real-time |
| 💾 Lưu video đầu ra | Xuất video đã annotated |
| 📝 Ghi log vi phạm | Lưu lịch sử vi phạm ra file |

---

## 🧠 Kiến trúc hệ thống

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│   Camera /  │───▶│  Detection   │───▶│   Tracking   │───▶│ Parking Logic │───▶│    Alert    │
│   Video     │    │  (YOLOv8)    │    │  (DeepSORT)  │    │  (ROI + Time) │    │ (Visualize) │
└─────────────┘    └──────────────┘    └──────────────┘    └───────────────┘    └─────────────┘
       │                  │                   │                    │                    │
   MP4 file          Bounding box        Track ID +           Violation          Red bbox +
   or Webcam        + Confidence          Position            Detection          Alert text
                      + Class                                  Status             + Logging
```

### Luồng xử lý chi tiết

1. **Input**: Đọc từng frame từ video/camera
2. **Detection**: YOLOv8 phát hiện phương tiện → trả về bbox + confidence + class
3. **Tracking**: DeepSORT gán ID → theo dõi vị trí qua các frame
4. **Parking Logic**: Kiểm tra xe trong ROI + đứng yên + vượt ngưỡng thời gian
5. **Alert**: Vẽ annotation + hiển thị cảnh báo + ghi log

---

## 📁 Cấu trúc dự án

```
illegal-parking-detection-ITS/
│
├── config/
│   └── config.py              # ⚙️ Tập trung tất cả cấu hình
│
├── data/
│   ├── video.mp4              # 🎬 Video đầu vào mẫu
│   ├── output.mp4             # 📹 Video đầu ra (tự tạo)
│   └── violations.log         # 📝 Log vi phạm (tự tạo)
│
├── docs/                      # 📚 Tài liệu bổ sung
│
├── models/                    # 🤖 Model weights (tự download)
│
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py   # 🔍 Module nhận diện YOLOv8
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py         # 🏷️ Module theo dõi DeepSORT
│   │
│   ├── parking/
│   │   ├── __init__.py
│   │   └── parking_detector.py # ⛔ Module logic đậu xe
│   │
│   ├── alert/
│   │   ├── __init__.py
│   │   └── alert_manager.py   # 🔔 Module cảnh báo & hiển thị
│   │
│   └── main.py                # 🚀 Entry point chính
│
├── requirements.txt           # 📦 Dependencies
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
└── README.md                  # 📖 Tài liệu này
```

---

## 💻 Yêu cầu hệ thống

### Phần cứng tối thiểu
- **CPU**: Intel Core i5 trở lên (hoặc tương đương)
- **RAM**: 8 GB
- **GPU**: Không bắt buộc (chạy được trên CPU, GPU giúp tăng tốc)

### Phần mềm
- **Python**: 3.8 trở lên
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+

### Thư viện Python

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `opencv-python` | ≥ 4.8.0 | Xử lý video, hiển thị |
| `ultralytics` | ≥ 8.0.0 | YOLOv8 detection |
| `deep-sort-realtime` | ≥ 1.3.2 | DeepSORT tracking |
| `numpy` | ≥ 1.24.0 | Tính toán số |

---

## 🚀 Hướng dẫn cài đặt

### Bước 1: Clone repository

```bash
git clone https://github.com/ltb2k/illegal-parking-detection-ITS.git
cd illegal-parking-detection-ITS
```

### Bước 2: Tạo virtual environment (khuyến nghị)

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/macOS)
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Chuẩn bị video đầu vào

Đặt file video vào thư mục `data/`:

```bash
data/video.mp4
```

> **Lưu ý**: Model YOLOv8 (`yolov8n.pt`) sẽ tự động download lần chạy đầu tiên.

---

## 🎮 Hướng dẫn sử dụng

### Chạy cơ bản

```bash
python src/main.py
```

### Chạy với tuỳ chọn

```bash
# Sử dụng video khác
python src/main.py --source data/custom_video.mp4

# Sử dụng webcam
python src/main.py --source 0

# Thay đổi ngưỡng thời gian (10 giây)
python src/main.py --threshold 10

# Sử dụng model YOLO khác (chính xác hơn nhưng chậm hơn)
python src/main.py --model yolov8s.pt

# Thay đổi ngưỡng confidence
python src/main.py --confidence 0.6

# Vẽ vùng cấm đậu xe trước khi chạy
python src/main.py --draw-roi

# Không lưu video đầu ra
python src/main.py --no-save

# Kết hợp nhiều tuỳ chọn
python src/main.py --source data/video.mp4 --threshold 8 --draw-roi
```

### Phím tắt khi chạy

| Phím | Chức năng |
|------|-----------|
| `q`  | Thoát chương trình |
| `r`  | Vẽ lại vùng cấm đậu xe (ROI) bằng chuột |
| `p`  | Tạm dừng / Tiếp tục video |
| `s`  | Chụp screenshot frame hiện tại |

### Vẽ vùng cấm đậu xe (ROI)

1. Nhấn phím `r` hoặc chạy với `--draw-roi`
2. **Click chuột trái** và **kéo** để vẽ hình chữ nhật
3. Nhấn **ENTER** để xác nhận
4. Nhấn **ESC** để huỷ

---

## 🔬 Giải thích chi tiết từng module

### 1. Detection Module (`src/detection/yolo_detector.py`)

**Mục đích**: Nhận diện phương tiện trong mỗi frame video.

**Cách hoạt động**:
```
Frame ──▶ YOLOv8 Model ──▶ Filter Vehicle Classes ──▶ Detections List
```

**Chi tiết**:
- Sử dụng model YOLOv8 pre-trained trên COCO dataset
- Chỉ giữ lại 4 class phương tiện:
  - `car` (class ID: 2)
  - `motorcycle` (class ID: 3)
  - `bus` (class ID: 5)
  - `truck` (class ID: 7)
- Lọc theo ngưỡng confidence (mặc định: 0.5)

**Output format**:
```python
{
    'bbox': [x1, y1, x2, y2],     # Toạ độ bounding box
    'confidence': 0.92,            # Độ tin cậy
    'class_id': 2,                 # COCO class ID
    'class_name': 'Car'            # Tên class
}
```

**Class chính**: `YOLODetector`

| Method | Mô tả |
|--------|--------|
| `__init__(model_path, confidence)` | Khởi tạo và load model YOLO |
| `detect(frame)` | Chạy detection trên 1 frame, trả về list detections |
| `get_detection_summary(detections)` | Tạo chuỗi tóm tắt kết quả |

---

### 2. Tracking Module (`src/tracking/tracker.py`)

**Mục đích**: Gán ID duy nhất cho mỗi phương tiện và theo dõi qua các frame.

**Cách hoạt động**:
```
Detections + Frame ──▶ DeepSORT ──▶ Tracked Vehicles (with persistent IDs)
```

**Chi tiết**:
- Sử dụng thuật toán DeepSORT (Deep Simple Online Realtime Tracking)
- Kết hợp:
  - **Kalman Filter**: Dự đoán vị trí tiếp theo
  - **Hungarian Algorithm**: Ghép detection với track
  - **Appearance Features**: Phân biệt xe dựa trên visual features
- Track chỉ được xác nhận sau `n_init` frame liên tiếp (mặc định: 3)
- Track bị xoá nếu mất detection quá `max_age` frame (mặc định: 30)

**Output format**:
```python
{
    'track_id': 1,                  # ID duy nhất
    'bbox': [x1, y1, x2, y2],      # Bounding box
    'class_name': 'Car',            # Loại phương tiện
    'confirmed': True               # Track đã xác nhận
}
```

**Class chính**: `VehicleTracker`

| Method | Mô tả |
|--------|--------|
| `__init__(max_age, n_init, max_iou_distance)` | Khởi tạo DeepSORT tracker |
| `update(detections, frame)` | Cập nhật tracker với detections mới |
| `get_active_track_count()` | Số track đang active |

---

### 3. Parking Detection Module (`src/parking/parking_detector.py`)

**Mục đích**: Xác định phương tiện nào đang đậu trái phép.

**Cách hoạt động**:
```
Tracked Vehicles ──▶ Check ROI ──▶ Check Movement ──▶ Check Duration ──▶ Violation Status
```

**Logic phát hiện vi phạm** (3 điều kiện đồng thời):

```
1. Xe nằm TRONG vùng cấm (ROI)?                    ──▶ ✅ In ROI
2. Xe ĐỨNG YÊN (di chuyển < movement_threshold)?    ──▶ ✅ Stationary
3. Đứng yên QUÁ LÂU (> parking_time_threshold)?     ──▶ ✅ VIOLATION!
```

**Trạng thái xe**:

| Status | Ý nghĩa | Màu hiển thị |
|--------|----------|---------------|
| `normal` | Xe ngoài vùng cấm hoặc đang di chuyển | 🟢 Xanh lá |
| `warning` | Xe trong vùng cấm, chưa quá ngưỡng | 🟠 Cam |
| `violation` | Xe đậu trái phép (quá ngưỡng thời gian) | 🔴 Đỏ |

**Output format**:
```python
{
    'track_id': 1,
    'bbox': [x1, y1, x2, y2],
    'class_name': 'Car',
    'in_roi': True,                 # Có trong vùng cấm
    'is_violation': True,           # Là vi phạm
    'duration': 7.5,                # Thời gian đậu (giây)
    'status': 'violation'           # Trạng thái
}
```

**Class chính**: `ParkingDetector`

| Method | Mô tả |
|--------|--------|
| `__init__(roi, time_threshold, movement_threshold)` | Khởi tạo với ROI và ngưỡng |
| `set_roi(x1, y1, x2, y2)` | Cập nhật vùng cấm |
| `update(tracked_vehicles)` | Cập nhật trạng thái parking |
| `get_violation_count()` | Số vi phạm hiện tại |
| `get_stats()` | Thống kê tổng hợp |

---

### 4. Alert Module (`src/alert/alert_manager.py`)

**Mục đích**: Hiển thị trực quan và ghi log vi phạm.

**Cách hoạt động**:
```
Vehicles + Status ──▶ Draw Boxes ──▶ Draw Alerts ──▶ Draw Stats ──▶ Annotated Frame
                                                                        │
                                                        Log Violations ──┘
```

**Chi tiết hiển thị**:
- **Vùng cấm (ROI)**: Hình chữ nhật màu cyan + overlay mờ + label "NO PARKING ZONE"
- **Xe bình thường**: Khung xanh lá + ID + tên class
- **Xe cảnh báo**: Khung cam + ID + thời gian
- **Xe vi phạm**: Khung đỏ dày + ID + thời gian + text "ILLEGAL PARKING"
- **Bảng thống kê**: FPS, số xe, số xe trong vùng cấm, số vi phạm

**Class chính**: `AlertManager`

| Method | Mô tả |
|--------|--------|
| `draw_roi(frame, roi)` | Vẽ vùng cấm đậu xe |
| `draw_vehicles(frame, vehicles)` | Vẽ bbox + label cho tất cả xe |
| `draw_stats(frame, fps, stats)` | Vẽ bảng thống kê |
| `draw_instructions(frame)` | Vẽ hướng dẫn phím tắt |

---

### 5. Main (`src/main.py`)

**Mục đích**: Điều phối toàn bộ pipeline.

**Luồng thực thi**:
```
1. Parse arguments
2. Initialize modules (Detector, Tracker, ParkingDetector, AlertManager)
3. Open video source
4. [Optional] Draw ROI
5. Main loop:
   ├── Read frame
   ├── Detect vehicles (YOLO)
   ├── Track vehicles (DeepSORT)
   ├── Check parking violations
   ├── Draw annotations
   ├── Display & Save
   └── Handle keyboard input
6. Cleanup & Print statistics
```

---

## ⚙️ Cấu hình

Tất cả cấu hình nằm trong file `config/config.py`:

### Video
| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `VIDEO_SOURCE` | `data/video.mp4` | Đường dẫn video đầu vào |
| `OUTPUT_VIDEO` | `data/output.mp4` | Đường dẫn video đầu ra |
| `SAVE_OUTPUT` | `True` | Bật/tắt lưu video |

### Detection (YOLO)
| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `YOLO_MODEL` | `yolov8n.pt` | Model YOLOv8 (`n`/`s`/`m`/`l`/`x`) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Ngưỡng confidence tối thiểu |
| `VEHICLE_CLASSES` | `[2, 3, 5, 7]` | COCO class IDs |

### Tracking (DeepSORT)
| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `MAX_AGE` | `30` | Số frame giữ track khi mất detection |
| `N_INIT` | `3` | Số detection liên tiếp để xác nhận track |
| `MAX_IOU_DISTANCE` | `0.7` | Ngưỡng IOU để matching |

### Parking Logic
| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `NO_PARKING_ZONE` | `(200, 300, 800, 600)` | Toạ độ vùng cấm (x1,y1,x2,y2) |
| `PARKING_TIME_THRESHOLD` | `5.0` | Ngưỡng thời gian vi phạm (giây) |
| `MOVEMENT_THRESHOLD` | `15.0` | Ngưỡng di chuyển tối đa (pixel) |

### Lựa chọn model YOLO

| Model | Kích thước | Tốc độ | Độ chính xác |
|-------|-----------|--------|-------------|
| `yolov8n.pt` | 6.2 MB | ⚡ Nhanh nhất | Thấp nhất |
| `yolov8s.pt` | 21.5 MB | ⚡ Nhanh | Trung bình |
| `yolov8m.pt` | 49.7 MB | 🔄 Vừa | Khá |
| `yolov8l.pt` | 83.7 MB | 🐢 Chậm | Cao |
| `yolov8x.pt` | 130.5 MB | 🐢 Chậm nhất | Cao nhất |

---

## 📊 Kết quả đầu ra

### 1. Video hiển thị real-time
- Bounding box màu theo trạng thái (xanh/cam/đỏ)
- ID tracking + tên loại xe
- Vùng cấm với overlay
- Bảng thống kê (FPS, số xe, vi phạm)

### 2. Video đầu ra (`data/output.mp4`)
- Video đã annotated với tất cả thông tin trực quan
- Cùng resolution và FPS với video gốc

### 3. Log vi phạm (`data/violations.log`)
```
[2026-05-01 17:00:15] VIOLATION - Track ID: 3, Vehicle: Car, Duration: 5.2s
[2026-05-01 17:00:28] VIOLATION - Track ID: 7, Vehicle: Truck, Duration: 6.1s
```

---

## 🔧 Xử lý sự cố

### Lỗi thường gặp

**1. `ModuleNotFoundError: No module named 'ultralytics'`**
```bash
pip install ultralytics
```

**2. `Cannot open video source`**
- Kiểm tra đường dẫn file video có đúng không
- Đảm bảo file `data/video.mp4` tồn tại

**3. Video chạy chậm (FPS thấp)**
- Dùng model nhẹ hơn: `--model yolov8n.pt`
- Giảm resolution video đầu vào
- Sử dụng GPU nếu có (CUDA)

**4. Không phát hiện được xe**
- Giảm confidence threshold: `--confidence 0.3`
- Kiểm tra video có chứa phương tiện rõ ràng không

**5. ROI không phù hợp**
- Nhấn `r` để vẽ lại vùng cấm
- Hoặc chỉnh sửa `NO_PARKING_ZONE` trong `config/config.py`

---

## 📄 License

[MIT License](LICENSE) © 2026 ltb2k

---

<p align="center">
  <b>Illegal Parking Detection System</b><br>
  ITS Project — Computer Vision — YOLOv8 — DeepSORT
</p>
