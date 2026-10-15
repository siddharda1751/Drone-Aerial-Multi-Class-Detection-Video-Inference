"""
Unit tests for VideoAnalytics and unique class-wise and category counting.
"""

import os
import sys
import tempfile
import unittest

# Ensure DroneVision root and parent cv_project are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from DroneVision.src.analytics import (
        VideoAnalytics,
        OBJECT_CATEGORIES,
        CLASS_TO_CATEGORY,
        get_category_for_class,
    )
except ImportError:
    from src.analytics import (
        VideoAnalytics,
        OBJECT_CATEGORIES,
        CLASS_TO_CATEGORY,
        get_category_for_class,
    )


class TestVideoAnalytics(unittest.TestCase):

    def setUp(self):
        self.class_names = [
            "pedestrian", "people", "bicycle", "car", "van",
            "truck", "tricycle", "awning-tricycle", "bus", "motor"
        ]
        self.analytics = VideoAnalytics(self.class_names)

    def test_single_track_multi_frame_not_counted_multiple_times(self):
        """Verify that a track appearing in multiple consecutive frames is counted only once."""
        # Frame 1: track 1 (car) seen
        self.analytics.update_frame(1, 0.033, [("1", "car", 0.90)])
        # Frame 2: track 1 (car) seen again
        self.analytics.update_frame(2, 0.066, [("1", "car", 0.85)])
        # Frame 3: track 1 (car) seen again
        self.analytics.update_frame(3, 0.100, [("1", "car", 0.95)])

        counts = self.analytics.get_unique_counts_by_class()
        self.assertEqual(counts["car"], 1, "Track 1 should only be counted once for class 'car'")
        self.assertEqual(self.analytics.get_total_unique_count(), 1)

        record = self.analytics.records["1"]
        self.assertEqual(record.frames_detected, 3)
        self.assertAlmostEqual(record.avg_confidence, (0.90 + 0.85 + 0.95) / 3, places=3)
        self.assertAlmostEqual(record.duration_s, 0.100 - 0.033, places=3)

    def test_multiple_tracks_and_classes(self):
        """Verify distinct tracks across multiple classes."""
        # Frame 1: car #1, pedestrian #2
        self.analytics.update_frame(1, 0.033, [("1", "car", 0.80), ("2", "pedestrian", 0.75)])
        # Frame 2: car #1, bus #3
        self.analytics.update_frame(2, 0.066, [("1", "car", 0.82), ("3", "bus", 0.92)])
        # Frame 3: pedestrian #2, motor #4
        self.analytics.update_frame(3, 0.100, [("2", "pedestrian", 0.70), ("4", "motor", 0.88)])

        counts = self.analytics.get_unique_counts_by_class()
        self.assertEqual(counts["car"], 1)
        self.assertEqual(counts["pedestrian"], 1)
        self.assertEqual(counts["bus"], 1)
        self.assertEqual(counts["motor"], 1)
        self.assertEqual(counts["bicycle"], 0)
        self.assertEqual(self.analytics.get_total_unique_count(), 4)

    def test_category_groupings(self):
        """Verify category aggregation for Pedestrians, Vehicles, and Two-Wheelers/Cycles."""
        # Frame 1: pedestrian #1, people #2, car #3, bicycle #4, motor #5
        self.analytics.update_frame(1, 0.033, [
            ("1", "pedestrian", 0.80),
            ("2", "people", 0.75),
            ("3", "car", 0.90),
            ("4", "bicycle", 0.70),
            ("5", "motor", 0.85),
        ])

        cat_counts = self.analytics.get_unique_counts_by_category()
        self.assertEqual(cat_counts["Pedestrians"], 2)            # pedestrian #1, people #2
        self.assertEqual(cat_counts["Vehicles"], 1)               # car #3
        self.assertEqual(cat_counts["Two-Wheelers & Cycles"], 2)  # bicycle #4, motor #5

        cat_peaks = self.analytics.get_peak_concurrent_by_category()
        self.assertEqual(cat_peaks["Pedestrians"], 2)
        self.assertEqual(cat_peaks["Vehicles"], 1)
        self.assertEqual(cat_peaks["Two-Wheelers & Cycles"], 2)

    def test_empty_frames(self):
        """Verify empty detections do not crash or corrupt counts."""
        self.analytics.update_frame(1, 0.033, [])
        self.analytics.update_frame(2, 0.066, [])
        self.assertEqual(self.analytics.get_total_unique_count(), 0)

    def test_csv_export(self):
        """Verify CSV export produces correct header and data rows with categories."""
        self.analytics.update_frame(1, 0.040, [("10", "truck", 0.84)])
        self.analytics.update_frame(2, 0.080, [("10", "truck", 0.88)])

        with tempfile.TemporaryDirectory() as tmpdir:
            det_csv = os.path.join(tmpdir, "test_detailed.csv")
            sum_csv = os.path.join(tmpdir, "test_summary.csv")

            self.analytics.export_csv(det_csv, sum_csv)

            self.assertTrue(os.path.exists(det_csv))
            self.assertTrue(os.path.exists(sum_csv))

            with open(det_csv, "r", encoding="utf-8") as f:
                det_lines = [line.strip() for line in f.readlines()]
            self.assertEqual(
                det_lines[0],
                "track_id,category,class_name,first_timestamp_s,last_timestamp_s,duration_s,frames_detected,avg_confidence"
            )
            self.assertEqual(len(det_lines), 2)
            self.assertTrue(det_lines[1].startswith("10,Vehicles,truck,"))

            with open(sum_csv, "r", encoding="utf-8") as f:
                sum_lines = [line.strip() for line in f.readlines()]
            self.assertEqual(
                sum_lines[0],
                "category,class_name,peak_concurrent_in_frame,total_tracks_assigned"
            )
            self.assertTrue(any("Vehicles,truck,1,1" in line for line in sum_lines))
            self.assertTrue(any("TOTAL,TOTAL,1,1" in line for line in sum_lines))


if __name__ == "__main__":
    unittest.main()
