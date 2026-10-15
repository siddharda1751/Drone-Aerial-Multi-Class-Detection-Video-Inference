"""
Benchmark test measuring GPU acceleration and CPU fallback performance.
"""

import time
import unittest
import numpy as np
import torch
import os
import sys

# Ensure project root in path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.detector import AerialDetector
from src.tracker import DeepSORTTracker


class TestGPUBenchmark(unittest.TestCase):

    def setUp(self):
        self.weights_path = os.path.join(PROJECT_DIR, "models", "visdrone_yolov8n.pt")
        # Synthetic aerial test frame (640x640)
        self.dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    def test_cuda_hardware_info(self):
        """Verify NVIDIA GPU is detected and accessible."""
        self.assertTrue(torch.cuda.is_available(), "CUDA is not available on this system")
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"\n[GPU INFO] Device Name: {gpu_name}")
        print(f"[GPU INFO] Total VRAM: {total_vram_gb:.2f} GB")
        print(f"[GPU INFO] PyTorch CUDA Version: {torch.version.cuda}")

    def test_gpu_vs_cpu_benchmark(self):
        """Benchmark YOLOv8 aerial detector on GPU vs CPU."""
        iterations = 30

        # 1. Benchmark GPU
        if torch.cuda.is_available():
            gpu_detector = AerialDetector(self.weights_path, device="cuda:0")
            # Warm-up GPU
            for _ in range(5):
                gpu_detector.detect(self.dummy_frame)
            torch.cuda.synchronize()

            start_gpu = time.perf_counter()
            for _ in range(iterations):
                gpu_detector.detect(self.dummy_frame)
            torch.cuda.synchronize()
            gpu_time = time.perf_counter() - start_gpu
            gpu_fps = iterations / gpu_time
            gpu_ms = (gpu_time / iterations) * 1000
            print(f"\n[BENCHMARK] GPU (cuda:0) Inference: {gpu_ms:.2f} ms/frame | {gpu_fps:.1f} FPS")
        else:
            gpu_fps = None
            print("\n[BENCHMARK] GPU not available; skipping GPU benchmark")

        # 2. Benchmark CPU
        cpu_detector = AerialDetector(self.weights_path, device="cpu")
        for _ in range(3):
            cpu_detector.detect(self.dummy_frame)

        start_cpu = time.perf_counter()
        for _ in range(10):  # Smaller run on CPU
            cpu_detector.detect(self.dummy_frame)
        cpu_time = time.perf_counter() - start_cpu
        cpu_fps = 10 / cpu_time
        cpu_ms = (cpu_time / 10) * 1000
        print(f"[BENCHMARK] CPU Fallback Inference: {cpu_ms:.2f} ms/frame | {cpu_fps:.1f} FPS")

        if gpu_fps:
            speedup = gpu_fps / cpu_fps
            print(f"[BENCHMARK] GPU Speedup: {speedup:.2f}x faster than CPU")
            self.assertGreater(gpu_fps, cpu_fps, "GPU should be faster than CPU")


if __name__ == "__main__":
    unittest.main()
