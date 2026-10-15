"""
Detector module for aerial object detection using YOLOv8.

Handles:
1. Loading pretrained VisDrone YOLOv8 weights.
2. GPU acceleration with CPU fallback.
3. Verification of class names and mappings.
4. Standardized detection output for DeepSORT tracking.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Standard 10 VisDrone object classes
VISDRONE_CLASSES: Dict[int, str] = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

# Standard visually distinct BGR colors for annotations
VISDRONE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "pedestrian": (0, 255, 0),        # Vibrant Green
    "people": (0, 180, 0),            # Darker Green
    "bicycle": (255, 255, 0),         # Cyan / Yellow-Cyan
    "car": (255, 120, 0),             # Vivid Blue
    "van": (255, 180, 0),             # Sky Blue
    "truck": (0, 0, 255),             # Red
    "tricycle": (255, 0, 255),        # Magenta
    "awning-tricycle": (180, 0, 255), # Purple-Pink
    "bus": (0, 255, 255),             # Yellow
    "motor": (0, 140, 255),           # Orange
}


class AerialDetector:
    """
    Wraps Ultralytics YOLOv8 for aerial drone video inference.
    """

    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        agnostic_nms: bool = True,
        imgsz: int = 640,
        device: Optional[str] = "auto"
    ):
        """
        Parameters
        ----------
        weights_path : str
            Path to YOLOv8 weights (.pt file).
        conf_threshold : float
            Confidence threshold for detections (0.0 to 1.0).
        iou_threshold : float
            NMS IoU threshold.
        agnostic_nms : bool
            Whether to use class-agnostic NMS to suppress overlapping boxes across classes.
        imgsz : int
            Input resolution for inference (e.g. 640, 1280).
        device : str
            Device selection: 'cuda', 'cuda:0', 'cpu', or 'auto'.
        """
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.agnostic_nms = agnostic_nms
        self.imgsz = imgsz
        self.device = self._resolve_device(device)

        logging.info("Loading YOLOv8 model from %s on device: %s...", weights_path, self.device)
        from ultralytics import YOLO
        self.model = YOLO(weights_path)

        # Retrieve model class names or fallback to VisDrone standard
        raw_names = getattr(self.model, "names", {})
        if isinstance(raw_names, dict) and raw_names:
            self.class_names: Dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}
        elif isinstance(raw_names, list) and raw_names:
            self.class_names = {i: str(v) for i, v in enumerate(raw_names)}
        else:
            self.class_names = dict(VISDRONE_CLASSES)

        logging.info("Model loaded successfully. Supported classes (%d): %s",
                     len(self.class_names), list(self.class_names.values()))

    @staticmethod
    def _resolve_device(requested: Optional[str]) -> str:
        """
        Resolves device selection, checking CUDA availability for GPU acceleration.
        """
        import torch
        cuda_available = torch.cuda.is_available()

        if requested in (None, "", "auto"):
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                logging.info("CUDA GPU detected: %s. Using GPU acceleration.", gpu_name)
                return "cuda:0"
            logging.info("CUDA not available. Falling back to CPU.")
            return "cpu"

        req_lower = requested.lower()
        if "cuda" in req_lower or req_lower.isdigit():
            if not cuda_available:
                logging.warning("CUDA requested ('%s') but torch.cuda.is_available() is False. Falling back to CPU.", requested)
                return "cpu"
            return requested
        return "cpu"

    @staticmethod
    def _suppress_overlapping_boxes(
        detections: List[Tuple[List[float], float, int, str]],
        iou_thresh: float = 0.40,
        iomin_thresh: float = 0.65
    ) -> List[Tuple[List[float], float, int, str]]:
        """
        Suppresses redundant overlapping detections on the same physical object.
        Keeps the detection with higher confidence.
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d[1], reverse=True)
        kept: List[Tuple[List[float], float, int, str]] = []

        for candidate in sorted_dets:
            c_box = candidate[0]  # [x, y, w, h]
            c_x1, c_y1 = c_box[0], c_box[1]
            c_x2, c_y2 = c_x1 + c_box[2], c_y1 + c_box[3]
            c_area = max(0.0, c_box[2]) * max(0.0, c_box[3])
            if c_area <= 0:
                continue

            suppress = False
            for k in kept:
                k_box = k[0]
                k_x1, k_y1 = k_box[0], k_box[1]
                k_x2, k_y2 = k_x1 + k_box[2], k_y1 + k_box[3]
                k_area = max(0.0, k_box[2]) * max(0.0, k_box[3])
                if k_area <= 0:
                    continue

                # Intersection
                ix1 = max(c_x1, k_x1)
                iy1 = max(c_y1, k_y1)
                ix2 = min(c_x2, k_x2)
                iy2 = min(c_y2, k_y2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

                if inter > 0:
                    union = c_area + k_area - inter
                    iou = inter / union if union > 0 else 0.0
                    min_area = min(c_area, k_area)
                    iomin = inter / min_area if min_area > 0 else 0.0

                    # Suppress if boxes heavily overlap or one is inside the other
                    if iou > iou_thresh or iomin > iomin_thresh:
                        suppress = True
                        break

            if not suppress:
                kept.append(candidate)

        return kept

    def detect(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> List[Tuple[List[float], float, int, str]]:
        """
        Run object detection on an image / video frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV (H x W x C).
        conf : float, optional
            Override default confidence threshold.
        imgsz : int, optional
            Override default inference size.

        Returns
        -------
        list of tuples
            Each detection is ([left, top, width, height], confidence, class_id, class_name).
            [left, top, width, height] is in pixel coordinates format required by DeepSORT.
        """
        conf_val = conf if conf is not None else self.conf_threshold
        size_val = imgsz if imgsz is not None else self.imgsz

        results = self.model.predict(
            source=frame,
            conf=conf_val,
            iou=self.iou_threshold,
            agnostic_nms=self.agnostic_nms,
            imgsz=size_val,
            device=self.device,
            verbose=False
        )[0]

        detections: List[Tuple[List[float], float, int, str]] = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            # Coordinates in [x1, y1, x2, y2]
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            if confidence < conf_val:
                continue

            class_name = self.class_names.get(class_id, f"class_{class_id}")
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            # DeepSORT format: [left, top, w, h]
            detections.append(([x1, y1, w, h], confidence, class_id, class_name))

        # Suppress duplicate overlapping detections to prevent multi-box artifacts
        return self._suppress_overlapping_boxes(detections)
