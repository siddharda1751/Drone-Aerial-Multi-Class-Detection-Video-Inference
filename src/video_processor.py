"""
Video processing pipeline for aerial drone object detection and tracking.

Coordinates:
1. Frame reading via OpenCV.
2. Object detection via AerialDetector (YOLOv8).
3. Tracking association via DeepSORTTracker.
4. Analytics & unique counting via VideoAnalytics.
5. Rich HUD overlay (FPS, unique counts, class badges).
6. Annotated video writing and verification.
"""

import logging
import os
import time
from typing import Callable, Dict, Optional, Tuple
import cv2
import numpy as np

from .analytics import VideoAnalytics
from .detector import AerialDetector, VISDRONE_COLORS
from .tracker import DeepSORTTracker


class VideoProcessor:
    """
    Main aerial video inference and tracking pipeline.
    """

    def __init__(
        self,
        detector: AerialDetector,
        tracker: DeepSORTTracker,
        analytics: Optional[VideoAnalytics] = None,
        custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ):
        self.detector = detector
        self.tracker = tracker
        self.colors = custom_colors or VISDRONE_COLORS

        # Initialize analytics with detector class names
        class_list = list(self.detector.class_names.values())
        self.analytics = analytics or VideoAnalytics(class_list)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        show_live: bool = False
    ) -> Dict[str, object]:
        """
        Process an input video file end-to-end.

        Parameters
        ----------
        input_path : str
            Path to input video.
        output_path : str
            Path to save the annotated video.
        conf_threshold : float, optional
            Detection confidence threshold override.
        imgsz : int, optional
            Detection inference resolution override.
        max_frames : int, optional
            Optional limit on number of frames to process (useful for tests/demos).
        progress_callback : callable, optional
            Callback function signature: callback(current_frame, total_frames, current_fps).
        show_live : bool
            Whether to display live OpenCV preview window (requires GUI environment).

        Returns
        -------
        dict
            Processing summary metrics (frames processed, execution time, average FPS, unique counts).
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Unable to open video stream: {input_path}")

        # Query video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or np.isnan(source_fps):
            source_fps = 30.0  # Fallback default
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Setup VideoWriter with reliable codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, source_fps, (width, height))
        if not writer.isOpened():
            # Codec fallback
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(output_path, fourcc, source_fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise IOError(f"Failed to initialize VideoWriter for path: {output_path}")

        logging.info("Started processing '%s' -> '%s' (%dx%d, source FPS: %.2f, total frames: %d)",
                     input_path, output_path, width, height, source_fps, total_video_frames)

        frame_idx = 0
        start_time = time.time()
        fps_moving_avg = 0.0
        fps_alpha = 0.1  # Exponential moving average smoothing factor

        try:
            while True:
                if max_frames and frame_idx >= max_frames:
                    break

                loop_start = time.time()
                success, frame = cap.read()
                if not success or frame is None:
                    break

                frame_idx += 1
                timestamp_s = (frame_idx - 1) / source_fps

                # 1. YOLOv8 Detection
                detections = self.detector.detect(
                    frame=frame,
                    conf=conf_threshold,
                    imgsz=imgsz
                )

                # 2. DeepSORT Tracking
                active_tracks = self.tracker.update(
                    detections=detections,
                    frame=frame
                )

                # 3. Analytics Update
                # Convert active tracks for analytics: (track_id, class_name, confidence)
                analytics_input = [(t[0], t[1], t[3]) for t in active_tracks]
                active_counts = self.analytics.update_frame(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    active_tracks=analytics_input
                )

                # 4. Instantaneous & Average FPS Calculation
                loop_elapsed = time.time() - loop_start
                inst_fps = 1.0 / loop_elapsed if loop_elapsed > 0 else 0.0
                if fps_moving_avg == 0.0:
                    fps_moving_avg = inst_fps
                else:
                    fps_moving_avg = (fps_alpha * inst_fps) + ((1.0 - fps_alpha) * fps_moving_avg)

                total_elapsed = time.time() - start_time
                avg_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0.0

                # 5. Render Annotations & HUD Overlay
                annotated_frame = self._render_frame(
                    frame=frame,
                    active_tracks=active_tracks,
                    frame_idx=frame_idx,
                    total_frames=total_video_frames,
                    proc_fps=fps_moving_avg,
                    source_fps=source_fps,
                    device_str=self.detector.device
                )

                # 6. Write Frame
                writer.write(annotated_frame)

                if show_live:
                    cv2.imshow("DroneVision Live Preview", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if progress_callback:
                    progress_callback(frame_idx, total_video_frames, fps_moving_avg)

                if frame_idx % 30 == 0 or frame_idx == 1:
                    logging.info("Frame %d/%d (%.1f%%) | Proc FPS: %.1f | In-Frame Objects: %d",
                                 frame_idx, total_video_frames,
                                 (frame_idx / total_video_frames * 100) if total_video_frames > 0 else 0.0,
                                 fps_moving_avg, len(active_tracks))

        finally:
            cap.release()
            writer.release()
            if show_live:
                cv2.destroyAllWindows()

        total_time = time.time() - start_time
        overall_avg_fps = frame_idx / total_time if total_time > 0 else 0.0

        logging.info("Video processing complete. %d frames processed in %.2f s (Average FPS: %.2f)",
                     frame_idx, total_time, overall_avg_fps)

        return {
            "input_path": input_path,
            "output_path": output_path,
            "frames_processed": frame_idx,
            "total_time_s": total_time,
            "average_fps": overall_avg_fps,
            "source_fps": source_fps,
            "peak_concurrent_objects": self.analytics.get_peak_total_count(),
            "peak_counts_by_class": self.analytics.get_peak_concurrent_by_class(),
            "peak_counts_by_category": self.analytics.get_peak_concurrent_by_category(),
            "unique_counts_by_class": self.analytics.get_unique_counts_by_class(),
            "unique_counts_by_category": self.analytics.get_unique_counts_by_category(),
            "total_unique_tracks": self.analytics.get_total_unique_count(),
        }

    def _render_frame(
        self,
        frame: np.ndarray,
        active_tracks: list,
        frame_idx: int,
        total_frames: int,
        proc_fps: float,
        source_fps: float,
        device_str: str
    ) -> np.ndarray:
        """
        Draws bounding boxes, track labels, and the heads-up display overlay on the frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Resolution scaling factor for responsive fonts and elements
        scale = min(1.8, max(0.85, w / 1280.0))
        box_thickness = max(1, int(round(1.8 * scale)))
        tag_font_scale = 0.42 * scale
        tag_thickness = max(1, int(round(1.0 * scale)))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Draw each active track bounding box and tag
        for track_id, class_name, (raw_l, raw_t, raw_r, raw_b), confidence in active_tracks:
            color = self.colors.get(class_name, (0, 255, 255))
            left = max(0, min(w - 2, int(raw_l)))
            top = max(0, min(h - 2, int(raw_t)))
            right = max(left + 1, min(w - 1, int(raw_r)))
            bottom = max(top + 1, min(h - 1, int(raw_b)))

            # Bounding box
            cv2.rectangle(out, (left, top), (right, bottom), color, box_thickness)

            # Label text: class name, track ID, confidence
            label = f"{class_name} #{track_id} ({confidence:.2f})"
            (text_w, text_h), baseline = cv2.getTextSize(label, font, tag_font_scale, tag_thickness)

            # Tag background box positioned above the bounding box
            tag_pad_y = int(6 * scale)
            tag_pad_x = int(5 * scale)
            tag_top = max(0, top - text_h - tag_pad_y)
            tag_bottom = max(text_h + tag_pad_y, top)
            tag_right = min(w, left + text_w + tag_pad_x)

            cv2.rectangle(out, (left, tag_top), (tag_right, tag_bottom), color, -1)
            # Text inside tag (dark text on bright color, or white text)
            text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) > 400 else (255, 255, 255)
            cv2.putText(
                out,
                label,
                (left + int(3 * scale), tag_bottom - int(3 * scale)),
                font,
                tag_font_scale,
                text_color,
                tag_thickness,
                cv2.LINE_AA
            )

        # -------------------------------------------------------------
        # HUD Panel (Semi-transparent top-left overlay)
        # -------------------------------------------------------------
        hud_w = int(min(420 * scale, w - 20))
        curr_counts = self.analytics.current_frame_counts
        active_in_current_frame = [(cls, count) for cls, count in curr_counts.items() if count > 0]
        cat_counts = self.analytics.current_category_counts

        # Adaptive height calculation
        line_height = int(22 * scale)
        base_h = int(160 * scale)
        hud_h = base_h + max(int(20 * scale), len(active_in_current_frame) * line_height)
        hud_h = min(hud_h, h - 20)

        # Create semi-transparent overlay
        overlay = out.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (18, 20, 26), -1)
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (70, 75, 85), 1)
        # Blend overlay (alpha = 0.78)
        cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)

        hud_font_sm = 0.40 * scale
        hud_font_md = 0.45 * scale
        hud_font_title = 0.54 * scale

        # Title
        cv2.putText(out, "DRONEVISION AERIAL MOT", (20, int(32 * scale)),
                    font, hud_font_title, (0, 220, 255), 2, cv2.LINE_AA)

        # Performance & Status lines
        fps_text = f"Proc FPS: {proc_fps:5.1f} | Video: {source_fps:.1f} FPS"
        cv2.putText(out, fps_text, (20, int(52 * scale)),
                    font, hud_font_sm, (0, 255, 200), 1, cv2.LINE_AA)

        frame_str = f"Frame: {frame_idx}" + (f" / {total_frames}" if total_frames > 0 else "")
        cv2.putText(out, frame_str, (20, int(70 * scale)),
                    font, hud_font_sm, (220, 220, 220), 1, cv2.LINE_AA)

        dev_text = f"Device: {device_str}"
        cv2.putText(out, dev_text, (20, int(88 * scale)),
                    font, hud_font_sm, (180, 220, 255), 1, cv2.LINE_AA)

        # In-frame total count
        in_frame_count = len(active_tracks)
        count_color = (0, 255, 128) if in_frame_count > 0 else (180, 180, 190)
        cv2.putText(out, f"Active Objects In-Frame: {in_frame_count}", (20, int(110 * scale)),
                    font, hud_font_md, count_color, 2, cv2.LINE_AA)

        # Category badges line
        ped_cnt = cat_counts.get("Pedestrians", 0)
        veh_cnt = cat_counts.get("Vehicles", 0)
        cyc_cnt = cat_counts.get("Two-Wheelers & Cycles", 0)
        cat_badge_text = f"Pedestrians: {ped_cnt}  |  Vehicles: {veh_cnt}  |  Cycles: {cyc_cnt}"
        cv2.putText(out, cat_badge_text, (20, int(128 * scale)),
                    font, hud_font_sm, (255, 215, 0), 1, cv2.LINE_AA)

        # Divider line
        div_y = int(136 * scale)
        cv2.line(out, (20, div_y), (10 + hud_w - 10, div_y), (90, 95, 105), 1)

        # Current Frame Class Counts
        y_offset = int(154 * scale)
        if active_in_current_frame:
            for cls_name, count in sorted(active_in_current_frame, key=lambda x: x[1], reverse=True):
                if y_offset + int(14 * scale) > 10 + hud_h:
                    break
                cls_color = self.colors.get(cls_name, (200, 200, 200))
                pip_radius = max(3, int(4 * scale))
                cv2.circle(out, (26, y_offset - int(4 * scale)), pip_radius, cls_color, -1)
                cv2.putText(out, f"{cls_name}: {count}", (int(38 * scale), y_offset),
                            font, hud_font_sm, (240, 240, 240), 1, cv2.LINE_AA)
                y_offset += line_height
        else:
            cv2.putText(out, "No objects in current frame", (26, y_offset),
                        font, hud_font_sm, (140, 140, 150), 1, cv2.LINE_AA)

        return out
