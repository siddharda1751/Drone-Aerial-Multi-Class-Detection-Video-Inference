#!/usr/bin/env python3
"""
Convenience script for running DroneVision video inference and DeepSORT tracking.
Delegates directly to main.py.
"""

import os
import sys

# Ensure parent directory is in Python path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from main import main

if __name__ == "__main__":
    main()
