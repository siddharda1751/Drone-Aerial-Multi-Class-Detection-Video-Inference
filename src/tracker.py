"""
Tracker module integrating DeepSORT multi-object tracking with YOLOv8 detections.

Reuses the DeepSORT implementation and parameters from Repository B
(YOLOv8-DeepSORT-VisDrone) with enhancements for robust class mapping,
persistent ID preservation, and confidence tracking across frames.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class DeepSORTTracker:
    """
    Wraps DeepSORT (via deep-sort-realtime) for aerial multi-object tracking.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 1,
        max_iou_distance: float = 0.70,
        max_cosine_distance: float = 0.45,
        nn_budget: Optional[int] = None,
        override_track_class: Optional[str] = None,
        strict_active: bool = True,
        overlap_iou_threshold: float = 0.30,
        overlap_iomin_threshold: float = 0.60,
        gating_only_position: bool = True,
    ):
        """
        Parameters
        ----------
        max_age : int
            Maximum number of missed misses before a track is deleted. Default: 30.
        n_init : int
            Number of consecutive detections before the track is confirmed. Default: 1 (instant confirmation).
        max_iou_distance : float
            Maximum IOU distance threshold for association. Default: 0.70 (preserves identity during motion).
        max_cosine_distance : float
            Cosine distance threshold for appearance feature matching. Default: 0.45.
        strict_active : bool
            If True, only outputs tracks that have an active detection in the current frame
            (time_since_update == 0). Matches Repository B to prevent coasting ghost boxes.
        overlap_iou_threshold : float
            IoU threshold for suppressing redundant overlapping tracks on the same object.
        overlap_iomin_threshold : float
            Intersection-over-minimum-area threshold for nested duplicate tracks.
        gating_only_position : bool
            If True, Kalman gating uses spatial position only instead of aspect-ratio/height,
            preventing track fragmentation when aerial vehicles maneuver or turn.
        """
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.strict_active = strict_active
        self.overlap_iou_threshold = overlap_iou_threshold
        self.overlap_iomin_threshold = overlap_iomin_threshold
        self.gating_only_position = gating_only_position

        logging.info(
            "Initializing DeepSORT tracker (max_age=%d, n_init=%d, max_iou=%.2f, max_cosine=%.2f, strict_active=%s, gating_only_pos=%s)...",
            max_age, n_init, max_iou_distance, max_cosine_distance, strict_active, gating_only_position
        )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
                max_cosine_distance=max_cosine_distance,
                gating_only_position=gating_only_position,
                nn_budget=nn_budget,
                override_track_class=override_track_class,
            )

        # Persistent memory for class label and confidence retention across frames
        self.track_class_history: Dict[str, str] = {}
        self.track_conf_history: Dict[str, float] = {}

    @staticmethod
    def _suppress_overlapping_tracks(
        tracks: List[Tuple[str, str, Tuple[int, int, int, int], float]],
        iou_thresh: float = 0.30,
        iomin_thresh: float = 0.60
    ) -> List[Tuple[str, str, Tuple[int, int, int, int], float]]:
        """
        Deduplicates tracks that overlap on the same physical object.
        Keeps the track with higher confidence (or earlier confirmed track).
        """
        if len(tracks) <= 1:
            return tracks

        # Sort by confidence descending
        sorted_tracks = sorted(tracks, key=lambda t: t[3], reverse=True)
        kept: List[Tuple[str, str, Tuple[int, int, int, int], float]] = []

        for candidate in sorted_tracks:
            c_left, c_top, c_right, c_bottom = candidate[2]
            c_area = max(0, c_right - c_left) * max(0, c_bottom - c_top)
            if c_area <= 0:
                continue

            suppress = False
            for k in kept:
                k_left, k_top, k_right, k_bottom = k[2]
                k_area = max(0, k_right - k_left) * max(0, k_bottom - k_top)
                if k_area <= 0:
                    continue

                # Intersection
                ix1 = max(c_left, k_left)
                iy1 = max(c_top, k_top)
                ix2 = min(c_right, k_right)
                iy2 = min(c_bottom, k_bottom)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                if inter > 0:
                    union = c_area + k_area - inter
                    iou = inter / union if union > 0 else 0.0
                    min_area = min(c_area, k_area)
                    iomin = inter / min_area if min_area > 0 else 0.0

                    if iou > iou_thresh or iomin > iomin_thresh:
                        suppress = True
                        break

            if not suppress:
                kept.append(candidate)

        return kept

    def update(
        self,
        detections: List[Tuple[List[float], float, int, str]],
        frame: np.ndarray
    ) -> List[Tuple[str, str, Tuple[int, int, int, int], float]]:
        """
        Updates DeepSORT tracks with new frame detections.

        Parameters
        ----------
        detections : list of tuples
            Each detection is ([left, top, width, height], confidence, class_id, class_name)
        frame : np.ndarray
            The current image frame (needed by DeepSORT for appearance feature extraction).

        Returns
        -------
        list of tuples
            Confirmed tracks in format:
            (track_id, class_name, (left, top, right, bottom), confidence)
        """
        # Format detections for deep_sort_realtime:
        # expects: list of ([left, top, w, h], confidence, class_name_or_id)
        raw_detections = []
        for bbox, conf, cls_id, cls_name in detections:
            raw_detections.append((bbox, conf, cls_name))

        # Update DeepSORT tracker
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        active_tracks: List[Tuple[str, str, Tuple[int, int, int, int], float]] = []

        for track in tracks:
            # Only process confirmed tracks
            if not track.is_confirmed():
                continue

            # Following Repository B: strictly skip tracks with no detection in current frame
            # to eliminate ghost/coasting boxes drifting over the object
            if self.strict_active and track.time_since_update > 0:
                continue

            track_id = str(track.track_id)

            # Retrieve class name
            det_class = track.get_det_class()
            if det_class is not None:
                class_name = str(det_class)
                self.track_class_history[track_id] = class_name
            else:
                # Retain previous known class name for this track
                class_name = self.track_class_history.get(track_id, "unknown")

            # Retrieve confidence score
            det_conf = track.get_det_conf()
            if det_conf is not None:
                confidence = float(det_conf)
                self.track_conf_history[track_id] = confidence
            else:
                confidence = self.track_conf_history.get(track_id, 0.5)

            # Retrieve bounding box coordinates (left, top, right, bottom)
            ltrb = track.to_ltrb()
            left = max(0, int(round(ltrb[0])))
            top = max(0, int(round(ltrb[1])))
            right = max(left + 1, int(round(ltrb[2])))
            bottom = max(top + 1, int(round(ltrb[3])))

            active_tracks.append((track_id, class_name, (left, top, right, bottom), confidence))

        # Perform track deduplication to eliminate any double bounding boxes
        return self._suppress_overlapping_tracks(
            active_tracks,
            iou_thresh=self.overlap_iou_threshold,
            iomin_thresh=self.overlap_iomin_threshold
        )

    def reset(self):
        """Reset internal tracker state."""
        self.tracker.delete_all_tracks()
        self.track_class_history.clear()
        self.track_conf_history.clear()
