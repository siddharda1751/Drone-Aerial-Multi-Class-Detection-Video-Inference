"""
Analytics module for aerial object tracking across multi-class objects.

Provides:
1. Unique object counting per class and category without multi-frame duplication.
2. High-level categorization:
   - Pedestrians (pedestrian, people)
   - Vehicles (car, van, truck, bus)
   - Two-Wheelers & Cycles (bicycle, motor, tricycle, awning-tricycle)
3. Per-track temporal statistics: first seen, last seen, duration, frames detected, average confidence.
4. Export for detailed track metrics and categorized summary statistics.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# Standard VisDrone class categories
OBJECT_CATEGORIES: Dict[str, List[str]] = {
    "Pedestrians": ["pedestrian", "people"],
    "Vehicles": ["car", "van", "truck", "bus"],
    "Two-Wheelers & Cycles": ["bicycle", "motor", "tricycle", "awning-tricycle"],
}

CLASS_TO_CATEGORY: Dict[str, str] = {
    "pedestrian": "Pedestrians",
    "people": "Pedestrians",
    "car": "Vehicles",
    "van": "Vehicles",
    "truck": "Vehicles",
    "bus": "Vehicles",
    "bicycle": "Two-Wheelers & Cycles",
    "motor": "Two-Wheelers & Cycles",
    "tricycle": "Two-Wheelers & Cycles",
    "awning-tricycle": "Two-Wheelers & Cycles",
}


def get_category_for_class(class_name: str) -> str:
    """Returns the high-level category for a given VisDrone class name."""
    return CLASS_TO_CATEGORY.get(class_name.lower().strip(), "Other")


class TrackRecord:
    """Stores lifetime observation statistics for an individual track ID."""

    def __init__(self, track_id: str, class_name: str, frame_idx: int, timestamp_s: float, confidence: float):
        self.track_id: str = str(track_id)
        self.class_name: str = class_name
        self.category: str = get_category_for_class(class_name)
        self.first_frame: int = frame_idx
        self.last_frame: int = frame_idx
        self.first_timestamp_s: float = timestamp_s
        self.last_timestamp_s: float = timestamp_s
        self.frames_detected: int = 1
        self.confidences: List[float] = [float(confidence)]

    def update(self, frame_idx: int, timestamp_s: float, confidence: float):
        """Update track statistics with a new observation."""
        self.last_frame = frame_idx
        self.last_timestamp_s = timestamp_s
        self.frames_detected += 1
        self.confidences.append(float(confidence))

    @property
    def duration_s(self) -> float:
        """Total observation duration in seconds."""
        return max(0.0, self.last_timestamp_s - self.first_timestamp_s)

    @property
    def avg_confidence(self) -> float:
        """Mean detection confidence over all observed frames."""
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)


class VideoAnalytics:
    """
    Maintains current-frame object counts, peak concurrent counts, category groupings,
    and track-level metrics for all aerial objects.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names: List[str] = list(class_names) if class_names else []
        self.records: Dict[str, TrackRecord] = {}
        
        # Set of unique track IDs observed for each class and category
        self.unique_track_ids_by_class: Dict[str, Set[str]] = defaultdict(set)
        self.unique_track_ids_by_category: Dict[str, Set[str]] = defaultdict(set)
        
        # Tracks active in the most recently processed frame
        self.current_frame_counts: Dict[str, int] = defaultdict(int)
        self.current_category_counts: Dict[str, int] = defaultdict(int)
        
        # Peak concurrent objects observed simultaneously in any single frame
        self.peak_concurrent_by_class: Dict[str, int] = defaultdict(int)
        self.peak_concurrent_by_category: Dict[str, int] = defaultdict(int)
        self.peak_total_concurrent: int = 0
        
        self.total_processed_frames: int = 0
        self.total_video_duration_s: float = 0.0

    def update_frame(
        self,
        frame_idx: int,
        timestamp_s: float,
        active_tracks: List[Tuple[str, str, float]]
    ) -> Dict[str, int]:
        """
        Update analytics with tracks confirmed in the current frame.

        Parameters
        ----------
        frame_idx : int
            Zero-indexed or 1-indexed frame number.
        timestamp_s : float
            Timestamp of the frame in seconds.
        active_tracks : list of (track_id, class_name, confidence)
            Confirmed active tracks in this frame.

        Returns
        -------
        dict
            Counts of active objects in this specific frame by class name.
        """
        self.total_processed_frames = max(self.total_processed_frames, frame_idx)
        self.total_video_duration_s = max(self.total_video_duration_s, timestamp_s)
        self.current_frame_counts = defaultdict(int)
        self.current_category_counts = defaultdict(int)

        for track_id, class_name, confidence in active_tracks:
            tid_str = str(track_id)
            cat_name = get_category_for_class(class_name)
            
            self.current_frame_counts[class_name] += 1
            self.current_category_counts[cat_name] += 1
            
            self.unique_track_ids_by_class[class_name].add(tid_str)
            self.unique_track_ids_by_category[cat_name].add(tid_str)

            if tid_str not in self.records:
                self.records[tid_str] = TrackRecord(
                    track_id=tid_str,
                    class_name=class_name,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    confidence=confidence
                )
            else:
                self.records[tid_str].update(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    confidence=confidence
                )

        # Update peak concurrent object counts observed in any single frame
        total_in_frame = len(active_tracks)
        self.peak_total_concurrent = max(self.peak_total_concurrent, total_in_frame)
        
        for cls_name, count in self.current_frame_counts.items():
            self.peak_concurrent_by_class[cls_name] = max(
                self.peak_concurrent_by_class[cls_name], count
            )
            
        for cat_name, count in self.current_category_counts.items():
            self.peak_concurrent_by_category[cat_name] = max(
                self.peak_concurrent_by_category[cat_name], count
            )

        return dict(self.current_frame_counts)

    def get_current_counts_by_class(self) -> Dict[str, int]:
        """Returns live object counts in the most recently processed frame."""
        result = {name: self.current_frame_counts.get(name, 0) for name in self.class_names}
        for name, cnt in self.current_frame_counts.items():
            result[name] = cnt
        return result

    def get_current_counts_by_category(self) -> Dict[str, int]:
        """Returns live category counts in the most recently processed frame."""
        result = {cat: self.current_category_counts.get(cat, 0) for cat in OBJECT_CATEGORIES}
        for cat, cnt in self.current_category_counts.items():
            result[cat] = cnt
        return result

    def get_peak_concurrent_by_class(self) -> Dict[str, int]:
        """Returns maximum concurrent count in any single frame by class."""
        result = {name: self.peak_concurrent_by_class.get(name, 0) for name in self.class_names}
        for name, cnt in self.peak_concurrent_by_class.items():
            result[name] = cnt
        return result

    def get_peak_concurrent_by_category(self) -> Dict[str, int]:
        """Returns maximum concurrent count in any single frame by category."""
        result = {cat: self.peak_concurrent_by_category.get(cat, 0) for cat in OBJECT_CATEGORIES}
        for cat, cnt in self.peak_concurrent_by_category.items():
            result[cat] = cnt
        return result

    def get_peak_total_count(self) -> int:
        """Returns the highest number of simultaneous objects seen in any frame."""
        return self.peak_total_concurrent

    def get_unique_counts_by_class(self) -> Dict[str, int]:
        """Returns unique object count per class."""
        result = {}
        for name in self.class_names:
            result[name] = len(self.unique_track_ids_by_class[name])
        for name, ids in self.unique_track_ids_by_class.items():
            result[name] = len(ids)
        return result

    def get_unique_counts_by_category(self) -> Dict[str, int]:
        """Returns unique object count per high-level category."""
        result = {cat: len(self.unique_track_ids_by_category[cat]) for cat in OBJECT_CATEGORIES}
        for cat, ids in self.unique_track_ids_by_category.items():
            result[cat] = len(ids)
        return result

    def get_total_unique_count(self) -> int:
        """Returns total number of unique track IDs across all classes."""
        all_ids = set()
        for ids in self.unique_track_ids_by_class.values():
            all_ids.update(ids)
        return len(all_ids)

    def export_csv(
        self,
        detailed_path: str,
        summary_path: str
    ) -> Tuple[str, str]:
        """
        Export both detailed track-level CSV and category/class summary CSV.
        """
        os.makedirs(os.path.dirname(os.path.abspath(detailed_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)

        # 1. Detailed per-track CSV
        with open(detailed_path, "w", encoding="utf-8") as f:
            f.write("track_id,category,class_name,first_timestamp_s,last_timestamp_s,duration_s,frames_detected,avg_confidence\n")
            sorted_records = sorted(
                self.records.values(),
                key=lambda r: int(r.track_id) if r.track_id.isdigit() else r.track_id
            )
            for r in sorted_records:
                f.write(
                    f"{r.track_id},{r.category},{r.class_name},"
                    f"{r.first_timestamp_s:.3f},{r.last_timestamp_s:.3f},"
                    f"{r.duration_s:.3f},{r.frames_detected},{r.avg_confidence:.4f}\n"
                )

        # 2. Summary CSV reporting category, peak concurrent count in frame, and unique tracks
        peaks = self.get_peak_concurrent_by_class()
        unique_counts = self.get_unique_counts_by_class()
        cat_peaks = self.get_peak_concurrent_by_category()
        cat_unique = self.get_unique_counts_by_category()

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("category,class_name,peak_concurrent_in_frame,total_tracks_assigned\n")
            for cat_name, classes in OBJECT_CATEGORIES.items():
                for cls_name in classes:
                    f.write(f"{cat_name},{cls_name},{peaks.get(cls_name, 0)},{unique_counts.get(cls_name, 0)}\n")
                f.write(f"SUBTOTAL_{cat_name},ALL_{cat_name},{cat_peaks.get(cat_name, 0)},{cat_unique.get(cat_name, 0)}\n")

            standard_classes = {c for clist in OBJECT_CATEGORIES.values() for c in clist}
            for cls_name in sorted(peaks.keys()):
                if cls_name not in standard_classes:
                    f.write(f"Other,{cls_name},{peaks[cls_name]},{unique_counts.get(cls_name, 0)}\n")

            f.write(f"TOTAL,TOTAL,{self.get_peak_total_count()},{self.get_total_unique_count()}\n")

        return detailed_path, summary_path
