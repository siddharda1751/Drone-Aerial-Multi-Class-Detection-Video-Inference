# 🚁 DroneVision: Aerial Drone Multi-Class Detection & DeepSORT Video Tracking

An end-to-end, high-performance Computer Vision pipeline for aerial multi-class object detection, DeepSORT tracking, persistent ID assignment, dynamic FPS benchmarking, and multi-category analytics reporting in drone surveillance footage.

Built by integrating and enhancing components from **[baturalpoz06/aerial-object-detection-yolov8](https://github.com/baturalpoz06/aerial-object-detection-yolov8)** (fine-tuned VisDrone detector) and **[Shreya140724/YOLOv8-DeepSORT-VisDrone](https://github.com/Shreya140724/YOLOv8-DeepSORT-VisDrone)** (DeepSORT tracking & statistics pipeline).

---

## 📌 Table of Contents
- [System Architecture](#system-architecture)
- [Key Features & Improvements](#key-features--improvements)
- [Multi-Class Object Taxonomy & Categories](#multi-class-object-taxonomy--categories)
- [Test Videos (Pedestrian & Traffic Footage)](#test-videos-pedestrian--traffic-footage)
- [Pretrained Models & Dataset](#pretrained-models--dataset)
- [Environment & GPU Requirements](#environment--gpu-requirements)
- [Quick Start Installation](#quick-start-installation)
- [Running Inference](#running-inference)
- [Command Line Options](#command-line-options)
- [CSV Analytics Format](#csv-analytics-format)
- [Testing & Benchmarks](#testing--benchmarks)
- [Repository Attribution & Licenses](#repository-attribution--licenses)

---

## 🏗 System Architecture

```text
Input Drone Video (.mp4, .avi, .mov)
         │
         ▼
OpenCV VideoCapture (Frame-by-frame stream)
         │
         ▼
YOLOv8 Aerial Detector (VisDrone Pretrained: 640 or 1280)
   - PyTorch CUDA Acceleration (NVIDIA RTX 4050 GPU) / CPU Fallback
   - Multi-class bounding boxes [left, top, w, h], Confidence, Class ID & Label
   - Agnostic NMS & spatial deduplication
         │
         ▼
DeepSORT Tracking Association (deep-sort-realtime)
   - MobileNetV2 appearance feature embedding (FP16 on GPU)
   - Kalman filter spatial position gating (gating_only_position=True)
   - Confirmation gate (n_init=1) & strict current-frame updates
   - Persistent class identity and confidence retention
         │
         ▼
VideoAnalytics Engine
   - Multi-class & high-level category tracking:
       * Pedestrians (pedestrian, people)
       * Vehicles (car, van, truck, bus)
       * Two-Wheelers & Cycles (bicycle, motor, tricycle, awning-tricycle)
   - Temporal metrics: first seen, last seen, duration, average confidence
   - Peak concurrent objects in-frame vs total unique tracks assigned
         │
         ▼
Responsive HUD Renderer & VideoWriter
   - Dynamic scale-adapted overlay for high-res drone footage (up to 4K / 2.7K)
   - Real-time processing FPS vs video source FPS
   - Active in-frame object counts with category summary badges
   - Color-coded class indicators & bounding boxes
         │
         ├────────────────────────────────────────┐
         ▼                                        ▼
Annotated Output Video (.mp4)          CSV Analytics Reports
                                       ├─ tracking_analytics_detailed.csv
                                       └─ tracking_analytics_summary.csv
```

---

## ✨ Key Features & Improvements

### Improvement A: Full Multi-Class & Category Tracking
- **Not Just Vehicles**: Tracks **all** objects visible from aerial drone cameras—pedestrians walking in crowds, people standing, bicycles, motorcycles, cars, vans, trucks, buses, tricycles, and awning-tricycles.
- **High-Level Category Groupings**:
  1. **Pedestrians**: `pedestrian`, `people`
  2. **Vehicles**: `car`, `van`, `truck`, `bus`
  3. **Two-Wheelers & Cycles**: `bicycle`, `motor`, `tricycle`, `awning-tricycle`
- **Duplicate Prevention**: Persistent track IDs ensure an object detected across 300 frames is counted as **1 unique object** rather than 300 duplicates.

### Improvement B: Dynamic FPS & Responsive High-Res HUD
- **Real-Time FPS**: Computes smoothed exponential moving average processing FPS across loop iterations.
- **Responsive HUD Layout**: Automatically scales font size, line thickness, tag dimensions, and overlay box size based on input video resolution (from 720p up to 2.7K/4K drone video).
- **Category Summary Badges**: Displays instantaneous in-frame counts for Pedestrians, Vehicles, and Two-Wheelers directly on the video HUD.

### Improvement C: Categorized CSV Analytics Reports
- **Detailed Track Log (`tracking_analytics_detailed.csv`)**:
  - `track_id`: Unique tracking integer identifier.
  - `category`: High-level category (`Pedestrians`, `Vehicles`, `Two-Wheelers & Cycles`).
  - `class_name`: Detected VisDrone class name.
  - `first_timestamp_s`: Timestamp (seconds) when track was first confirmed.
  - `last_timestamp_s`: Timestamp (seconds) of the most recent observation.
  - `duration_s`: Total observation window (`last - first`).
  - `frames_detected`: Number of frames in which the object was tracked.
  - `avg_confidence`: Mean detection confidence score over the track's lifetime.
- **Summary Report (`tracking_analytics_summary.csv`)**:
  - Breakdown by class and category subtotals.
  - Peak simultaneous count in any frame vs total unique tracks assigned.
  - Overall `TOTAL` unique tracked count.

---

## 🏷 Multi-Class Object Taxonomy & Categories

All 10 canonical VisDrone classes are mapped, categorized, and rendered with visually distinct BGR colors:

| Category | Class ID | Class Name | HUD Color (BGR) |
|:---|:---:|:---|:---|
| **Pedestrians** | `0` | `pedestrian` | Vibrant Green `(0, 255, 0)` |
| **Pedestrians** | `1` | `people` | Darker Green `(0, 180, 0)` |
| **Two-Wheelers & Cycles** | `2` | `bicycle` | Cyan / Yellow-Cyan `(255, 255, 0)` |
| **Vehicles** | `3` | `car` | Vivid Blue `(255, 120, 0)` |
| **Vehicles** | `4` | `van` | Sky Blue `(255, 180, 0)` |
| **Vehicles** | `5` | `truck` | Red `(0, 0, 255)` |
| **Two-Wheelers & Cycles** | `6` | `tricycle` | Magenta `(255, 0, 255)` |
| **Two-Wheelers & Cycles** | `7` | `awning-tricycle` | Purple-Pink `(180, 0, 255)` |
| **Vehicles** | `8` | `bus` | Yellow `(0, 255, 255)` |
| **Two-Wheelers & Cycles** | `9` | `motor` | Orange `(0, 140, 255)` |

---

## 🎥 Test Videos (Pedestrian & Traffic Footage)

The `input_videos/` directory contains curated test drone videos:

| Video File | Resolution | Scene Description | Detected Objects |
|:---|:---:|:---|:---|
| `drone_people_car_traffic.mp4` | **2688x1512** (2.7K) | **Dense Urban Intersection**: Busy city street filmed from drone with active pedestrian crosswalks and heavy car/scooter traffic | **Car, Pedestrian, People, Motor, Bicycle, Van, Tricycle** (Peak 90+ objects in frame!) |
| `drone_test1.mp4` | 2720x1530 | Urban Plaza / Promenade with high pedestrian crowd density | Pedestrian, People, Car, Motor, Van |
| `drone_test4.mp4` | 1904x1072 | Mixed Urban Transit Junction | Car, Van, Truck, Bus, Motor, Pedestrian, Tricycle |
| `sample_drone.mp4` | 768x432 | Lightweight sample drone clip | Pedestrian |

---

## 🧠 Pretrained Models & Dataset

Trained and evaluated on the **VisDrone2019-DET** benchmark (6,471 training images, 548 validation images) captured by drone-mounted cameras across various altitudes, angles, and lighting conditions.

| Model File | Resolution | Size | Source | Recommended Use |
|:---|:---:|:---:|:---|:---|
| `models/visdrone_yolov8n.pt` | 640x640 | 6.2 MB | [Repository A](https://github.com/baturalpoz06/aerial-object-detection-yolov8) | Real-time inference (100+ FPS on RTX 4050) |
| `models/visdrone_yolov8n_1280.pt` | 1280x1280 | 6.3 MB | [Repository A](https://github.com/baturalpoz06/aerial-object-detection-yolov8) | Small-object high-altitude imagery (+73% small object mAP) |

---

## 💻 Environment & GPU Requirements

### Verified Hardware Environment
- **Operating System**: Windows 11 (PowerShell)
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6, Ada Lovelace, Compute Capability 8.9)
- **CPU**: Intel Core i7-13620H (16 threads)
- **RAM**: 16 GB
- **NVIDIA Driver**: 617.42 (supports CUDA 12.x / 13.x)
- **Python**: 3.10.11

---

## 🚀 Quick Start Installation

Run these commands in Windows PowerShell:

### 1. Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install PyTorch with CUDA 12.1 Acceleration
```powershell
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Core Project Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Verify GPU Acceleration
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

---

## 🎬 Running Inference

### Option 1: Full People & Car Traffic Video (Default)
```powershell
python main.py --source input_videos/drone_people_car_traffic.mp4 --weights models/visdrone_yolov8n_1280.pt --imgsz 1280 --output output/drone_people_car_traffic_tracked.mp4
```

### Option 2: Auto-Detect Input Video
If `--source` is omitted, `main.py` automatically picks `drone_people_car_traffic.mp4`:
```powershell
python main.py
```

### Option 3: Process with Live Preview Window
```powershell
python main.py --source input_videos/drone_people_car_traffic.mp4 --show
```

### Option 4: CPU Fallback Mode
```powershell
python main.py --source input_videos/drone_people_car_traffic.mp4 --device cpu
```

---

## ⚙ Command Line Options

| Argument | Type | Default | Description |
|:---|:---:|:---|:---|
| `--source` | `str` | `None` (auto-detects) | Path to input video (`.mp4`, `.avi`, `.mov`). |
| `--output` | `str` | `output/tracked_drone_output.mp4` | Filepath for saving annotated output video. |
| `--weights` | `str` | `models/visdrone_yolov8n.pt` | Pretrained YOLOv8 weights file. |
| `--imgsz` | `int` | `auto` (`1280` if weights have 1280, else `640`) | Inference resolution. |
| `--conf` | `float` | `0.35` | Detector confidence threshold (`0.0` - `1.0`). |
| `--device` | `str` | `auto` | Execution device: `auto`, `cuda:0`, `cuda`, or `cpu`. |
| `--save-csv` | `flag` | `True` | Exports detailed and summary CSV reports. |
| `--no-csv` | `flag` | `False` | Disables CSV analytics generation. |
| `--csv-dir` | `str` | `output` | Directory where CSV files are written. |
| `--max-frames` | `int` | `None` | Max number of frames to process (useful for rapid testing). |
| `--show` | `flag` | `False` | Displays live OpenCV preview window while processing. |

---

## 📊 CSV Analytics Format

### 1. Detailed Track Report (`tracking_analytics_detailed.csv`)
```csv
track_id,category,class_name,first_timestamp_s,last_timestamp_s,duration_s,frames_detected,avg_confidence
1,Vehicles,car,0.033,7.667,7.633,230,0.8523
2,Pedestrians,pedestrian,0.033,5.200,5.167,155,0.7812
3,Two-Wheelers & Cycles,motor,0.067,4.800,4.733,142,0.6940
```

### 2. Summary Report (`tracking_analytics_summary.csv`)
```csv
category,class_name,peak_concurrent_in_frame,total_tracks_assigned
Pedestrians,pedestrian,20,38
Pedestrians,people,28,45
SUBTOTAL_Pedestrians,ALL_Pedestrians,48,83
Vehicles,car,37,62
Vehicles,van,2,4
Vehicles,truck,0,0
Vehicles,bus,0,0
SUBTOTAL_Vehicles,ALL_Vehicles,39,66
Two-Wheelers & Cycles,bicycle,5,9
Two-Wheelers & Cycles,motor,14,26
Two-Wheelers & Cycles,tricycle,1,2
Two-Wheelers & Cycles,awning-tricycle,0,0
SUBTOTAL_Two-Wheelers & Cycles,ALL_Two-Wheelers & Cycles,20,37
TOTAL,TOTAL,91,186
```

---

## 🧪 Testing & Benchmarks

Run the complete automated test suite:

```powershell
python -m unittest discover -s tests -v
```

### Verified Benchmark Results (RTX 4050 Laptop GPU vs Intel i7-13620H)

| Test Module | Description | Result | Measured Performance |
|:---|:---|:---:|:---|
| `test_class_mapping.py` | 10 VisDrone class mapping, indices, and color tuples | **PASS** | Complete class fidelity |
| `test_analytics.py` | Unique counting, duplicate suppression, duration, categories & CSV exports | **PASS** | 5 unit tests passing |
| `test_gpu_benchmark.py` | CUDA GPU vs CPU inference speed | **PASS** | **GPU**: 7.19 ms (139.1 FPS)<br>**CPU**: 49.43 ms (20.2 FPS)<br>**Speedup: 6.88x** |
| `End-to-End Drone Test` | 2.7K Full People & Traffic video tracking | **PASS** | Tracks 90+ concurrent objects across all classes |

---

## 📜 Repository Attribution & Licenses

This project integrates open-source components with proper attribution:

- **Aerial Object Detection YOLOv8 Model**: Reused pretrained weights and resolution study from [baturalpoz06/aerial-object-detection-yolov8](https://github.com/baturalpoz06/aerial-object-detection-yolov8).
- **YOLOv8-DeepSORT Pipeline & VisDrone Architecture**: Reused tracking framework and visualization patterns from [Shreya140724/YOLOv8-DeepSORT-VisDrone](https://github.com/Shreya140724/YOLOv8-DeepSORT-VisDrone) (MIT License).
- **DeepSORT Implementation**: Provided via `deep-sort-realtime` utilizing MobileNetV2 appearance descriptors.
- **VisDrone2019 Dataset**: AISKYEYE team, Tianjin University (Academic Research License).
- **Ultralytics YOLOv8**: AGPL-3.0 License.
