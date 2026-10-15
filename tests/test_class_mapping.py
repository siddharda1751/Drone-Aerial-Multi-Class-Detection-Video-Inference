"""
Unit tests for VisDrone class mapping and color configurations.
"""

import os
import sys
import unittest

# Ensure DroneVision root and parent cv_project are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from DroneVision.src.detector import VISDRONE_CLASSES, VISDRONE_COLORS
except ImportError:
    from src.detector import VISDRONE_CLASSES, VISDRONE_COLORS


class TestClassMapping(unittest.TestCase):

    def test_visdrone_classes_count_and_keys(self):
        """Verify VisDrone has exactly 10 canonical classes with indices 0..9."""
        self.assertEqual(len(VISDRONE_CLASSES), 10)
        expected_classes = {
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
        self.assertEqual(VISDRONE_CLASSES, expected_classes)

    def test_visdrone_colors_coverage_and_format(self):
        """Verify every VisDrone class has a valid 3-channel BGR color tuple."""
        for class_id, class_name in VISDRONE_CLASSES.items():
            self.assertIn(class_name, VISDRONE_COLORS, f"Missing color for class '{class_name}'")
            color = VISDRONE_COLORS[class_name]
            self.assertIsInstance(color, tuple)
            self.assertEqual(len(color), 3)
            for channel in color:
                self.assertIsInstance(channel, int)
                self.assertTrue(0 <= channel <= 255, f"Channel {channel} outside [0, 255] in {class_name}")

    def test_model_names_mapping_if_weights_present(self):
        """If local pretrained weights exist, verify their embedded class names match VisDrone."""
        weights_path = os.path.join(os.path.dirname(__file__), "..", "models", "visdrone_yolov8n.pt")
        if os.path.exists(weights_path):
            try:
                from ultralytics import YOLO
                model = YOLO(weights_path)
                model_names = model.names
                self.assertEqual(len(model_names), 10)
                for idx, expected_name in VISDRONE_CLASSES.items():
                    self.assertEqual(
                        model_names[idx].lower().strip(),
                        expected_name.lower().strip(),
                        f"Mismatch at index {idx}: expected '{expected_name}', got '{model_names[idx]}'"
                    )
            except ImportError:
                self.skipTest("Ultralytics not yet installed in current testing environment.")


if __name__ == "__main__":
    unittest.main()
