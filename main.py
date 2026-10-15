"""
DroneVision: Aerial Drone Multi-Class Detection and Video Tracking CLI.

Usage example:
    python main.py --source input_videos/sample_drone.mp4
    python main.py --source input_videos/sample_drone.mp4 --imgsz 1280 --device cuda:0
"""

import argparse
import glob
import logging
import os
import sys
import time
import warnings

# Suppress non-critical third-party warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.analytics import VideoAnalytics
from src.detector import AerialDetector
from src.tracker import DeepSORTTracker
from src.video_processor import VideoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="DroneVision: Drone Aerial Multi-Class Detection & DeepSORT Tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to input video (MP4/AVI/MOV). If omitted, searches 'input_videos/'."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/tracked_drone_output.mp4",
        help="Path for saving the annotated output video."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/visdrone_yolov8n.pt",
        help="Path to pretrained YOLOv8 model weights (.pt). E.g. models/visdrone_yolov8n.pt (640) or models/visdrone_yolov8n_1280.pt (1280)."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image resolution (auto-detected: 1280 for 1280 weights, otherwise 640)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Object detection confidence threshold (recommended 0.35 - 0.45)."
    )
    parser.add_argument(
        "--agnostic-nms",
        dest="agnostic_nms",
        action="store_true",
        default=True,
        help="Use class-agnostic NMS to eliminate multi-class duplicate boxes."
    )
    parser.add_argument(
        "--no-agnostic-nms",
        dest="agnostic_nms",
        action="store_false",
        help="Disable class-agnostic NMS."
    )
    parser.add_argument(
        "--strict-active",
        dest="strict_active",
        action="store_true",
        default=True,
        help="Only display tracks updated in current frame (prevents ghost/coasting boxes as in Repo B)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'auto', 'cuda', 'cuda:0', or 'cpu'."
    )
    parser.add_argument(
        "--save-csv",
        dest="save_csv",
        action="store_true",
        default=True,
        help="Export CSV reports for detailed tracking and class counts."
    )
    parser.add_argument(
        "--no-csv",
        dest="save_csv",
        action="store_false",
        help="Disable CSV report generation."
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="output",
        help="Directory to save CSV reports."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process (useful for quick tests)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display OpenCV preview window during processing."
    )
    return parser.parse_args()


def resolve_path(path: str, default_parent: str = SCRIPT_DIR) -> str:
    """Resolves relative and absolute file paths cleanly."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    # Check relative to default_parent
    alt = os.path.join(default_parent, path)
    if os.path.exists(alt):
        return alt
    # If doesn't exist yet, return absolute relative to CWD or default_parent
    return os.path.abspath(path)


def resolve_source_video(requested_source: str) -> str:
    """Finds or resolves the input video path."""
    if requested_source:
        resolved = resolve_path(requested_source)
        if os.path.exists(resolved):
            return resolved
        # Also check inside input_videos directory
        in_video_candidate = os.path.join(SCRIPT_DIR, "input_videos", requested_source)
        if os.path.exists(in_video_candidate):
            return os.path.abspath(in_video_candidate)

    # Preferred default videos if none requested
    preferred_names = [
        "drone_people_car_traffic.mp4",
        "drone_test2.mp4",
        "sample_drone.mp4"
    ]
    for sdir in [os.path.join(SCRIPT_DIR, "input_videos"), "input_videos"]:
        if os.path.exists(sdir):
            for pname in preferred_names:
                candidate = os.path.join(sdir, pname)
                if os.path.exists(candidate):
                    print(f"[*] No specific source provided; using preferred drone video: {candidate}")
                    return os.path.abspath(candidate)

    # Search inside input_videos directory
    search_dirs = [
        os.path.join(SCRIPT_DIR, "input_videos"),
        "input_videos"
    ]
    patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    for sdir in search_dirs:
        if os.path.exists(sdir):
            for pat in patterns:
                matches = glob.glob(os.path.join(sdir, pat))
                if matches:
                    print(f"[*] No specific source provided; using found video: {matches[0]}")
                    return os.path.abspath(matches[0])

    raise FileNotFoundError(
        f"No input video found at '{requested_source}'. Please provide a valid video path with --source "
        f"or place an MP4 video into '{os.path.join(SCRIPT_DIR, 'input_videos')}'."
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    args = parse_args()

    # 1. Resolve Input Video
    try:
        source_path = resolve_source_video(args.source)
    except FileNotFoundError as err:
        logging.error("%s", err)
        sys.exit(1)

    # 2. Resolve Model Weights & Resolution
    weights_path = resolve_path(args.weights)
    if not os.path.exists(weights_path):
        logging.error("Pretrained weights not found at: %s", weights_path)
        logging.info("Please verify the file exists or refer to models/README.md.")
        sys.exit(1)

    # Auto-detect resolution from weights name if not explicitly set
    if args.imgsz is not None:
        imgsz = args.imgsz
    elif "1280" in os.path.basename(weights_path):
        imgsz = 1280
    else:
        imgsz = 640

    # Resolve output paths
    output_video_path = resolve_path(args.output)
    csv_dir = resolve_path(args.csv_dir)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    print("\n" + "=" * 65)
    print("      DRONEVISION: AERIAL DETECTION & TRACKING PIPELINE")
    print("=" * 65)
    print(f"Input Video   : {source_path}")
    print(f"Model Weights : {weights_path}")
    print(f"Resolution    : {imgsz}x{imgsz}")
    print(f"Confidence    : {args.conf}")
    print(f"Agnostic NMS  : {args.agnostic_nms}")
    print(f"Strict Active : {args.strict_active}")
    print(f"Device        : {args.device}")
    print(f"Output Video  : {output_video_path}")
    print(f"Export CSVs   : {args.save_csv} (Dir: {csv_dir})")
    print("=" * 65 + "\n")

    # 3. Initialize Pipeline Components
    try:
        detector = AerialDetector(
            weights_path=weights_path,
            conf_threshold=args.conf,
            agnostic_nms=args.agnostic_nms,
            imgsz=imgsz,
            device=args.device
        )
    except Exception as e:
        logging.error("Failed to initialize detector: %s", e)
        sys.exit(1)

    tracker = DeepSORTTracker(
        max_age=30,
        n_init=1,
        max_iou_distance=0.70,
        max_cosine_distance=0.45,
        strict_active=args.strict_active,
        gating_only_position=True
    )

    analytics = VideoAnalytics(list(detector.class_names.values()))

    processor = VideoProcessor(
        detector=detector,
        tracker=tracker,
        analytics=analytics
    )

    # 4. Run Video Inference
    result = processor.process_video(
        input_path=source_path,
        output_path=output_video_path,
        conf_threshold=args.conf,
        imgsz=imgsz,
        max_frames=args.max_frames,
        show_live=args.show
    )

    # 5. Export CSV Analytics
    detailed_csv_path = None
    summary_csv_path = None
    if args.save_csv:
        detailed_csv_path = os.path.join(csv_dir, "tracking_analytics_detailed.csv")
        summary_csv_path = os.path.join(csv_dir, "tracking_analytics_summary.csv")
        analytics.export_csv(detailed_csv_path, summary_csv_path)

    # 6. Print Formatted Summary Report
    print("\n" + "=" * 65)
    print("                 INFERENCE & TRACKING REPORT")
    print("=" * 65)
    print(f"Frames Processed       : {result['frames_processed']}")
    print(f"Source Video FPS       : {result['source_fps']:.2f}")
    print(f"Processing Total Time  : {result['total_time_s']:.2f} seconds")
    print(f"Average Inference FPS  : {result['average_fps']:.2f} FPS")
    print(f"Peak Concurrent Objects: {result['peak_concurrent_objects']} simultaneous in frame")
    print(f"Total Unique Objects   : {result['total_unique_tracks']} distinct tracks")
    print("-" * 65)
    print("CATEGORY SUMMARY (Unique Tracks Assigned / Peak Simultaneous):")
    cat_unique = result.get("unique_counts_by_category", {})
    cat_peaks = result.get("peak_counts_by_category", {})
    from src.analytics import OBJECT_CATEGORIES
    for cat_name, members in OBJECT_CATEGORIES.items():
        u_cnt = cat_unique.get(cat_name, 0)
        p_cnt = cat_peaks.get(cat_name, 0)
        members_str = ", ".join(members)
        print(f"  * {cat_name:<24} : {u_cnt:>3} unique  | {p_cnt:>3} peak  ({members_str})")
    print("-" * 65)
    print("DETAILED CLASS-WISE BREAKDOWN:")
    cls_unique = result.get("unique_counts_by_class", {})
    cls_peaks = result.get("peak_counts_by_class", {})
    for cls_name in sorted(detector.class_names.values()):
        u_cnt = cls_unique.get(cls_name, 0)
        p_cnt = cls_peaks.get(cls_name, 0)
        if u_cnt > 0 or p_cnt > 0:
            print(f"  - {cls_name:<20} : {u_cnt:>3} unique tracks  (Peak simultaneous: {p_cnt})")
    print("-" * 65)
    print(f"Annotated Video Saved  : {output_video_path}")
    if args.save_csv:
        print(f"Detailed Track CSV     : {detailed_csv_path}")
        print(f"Summary Class CSV      : {summary_csv_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
