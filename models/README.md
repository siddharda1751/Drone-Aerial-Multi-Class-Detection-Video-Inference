# Pretrained Drone Vision Models

This directory contains pretrained object detection weights fine-tuned on the **VisDrone2019-DET** dataset for aerial drone imagery.

## Available Models

### 1. `visdrone_yolov8n.pt` (Primary / Default)
- **Architecture**: YOLOv8 Nano (`yolov8n`)
- **Trained Resolution**: 640x640 pixels
- **File Size**: ~6.2 MB
- **Source**: fine-tuned by Baturalp Öz ([Repository A](https://github.com/baturalpoz06/aerial-object-detection-yolov8))
- **Use Case**: Real-time video inference, high FPS on consumer laptops and edge devices.

### 2. `visdrone_yolov8n_1280.pt` (High Resolution)
- **Architecture**: YOLOv8 Nano (`yolov8n`)
- **Trained Resolution**: 1280x1280 pixels
- **File Size**: ~6.3 MB
- **Source**: [Repository A](https://github.com/baturalpoz06/aerial-object-detection-yolov8)
- **Use Case**: High-altitude drone footage where small objects (such as bicycles, pedestrians, and motorcycles) occupy only a few pixels. Offers significant mAP gains for small aerial objects (+73% overall mAP, ~13x for bicycles).

### 3. Alternative: `dronefreak/yolov8m-visdrone` (Hugging Face)
- **Architecture**: YOLOv8 Medium (`yolov8m`)
- **File Size**: ~52 MB
- **Source**: [Hugging Face dronefreak/yolov8m-visdrone](https://huggingface.co/dronefreak/yolov8m-visdrone)
- **Download Command**:
  ```bash
  python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='dronefreak/yolov8m-visdrone', filename='best.pt', local_dir='models')"
  ```

---

## Supported Object Classes (10 VisDrone Classes)

| Class ID | Class Name | Description |
|:--------:|:-----------|:------------|
| 0 | `pedestrian` | Standing, walking, or running individual |
| 1 | `people` | Crowd, group, or less distinctly articulated humans |
| 2 | `bicycle` | Human-powered two-wheelers |
| 3 | `car` | Standard passenger automobiles |
| 4 | `van` | Passenger / cargo vans |
| 5 | `truck` | Light and heavy freight trucks |
| 6 | `tricycle` | Open three-wheeled human/motor cargo cycles |
| 7 | `awning-tricycle` | Three-wheelers with protective rain/sun roofs |
| 8 | `bus` | Public transit or coach buses |
| 9 | `motor` | Motorcycles and mopeds |
