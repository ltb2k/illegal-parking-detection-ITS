"""
Main Entry Point - Illegal Parking Detection System
====================================================
Orchestrates the complete detection pipeline:

    Camera/Video → Detection → Tracking → Parking Logic → Alert

Usage:
    python src/main.py                          # Use default video
    python src/main.py --source path/to/video   # Custom video
    python src/main.py --source 0               # Webcam

Controls:
    q - Quit
    r - Draw new ROI (No Parking Zone) with mouse
    p - Pause / Resume
    s - Screenshot current frame
"""

import cv2
import time
import argparse
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    VIDEO_SOURCE, OUTPUT_VIDEO, SAVE_OUTPUT,
    NO_PARKING_ZONE, WINDOW_NAME
)
from src.detection.yolo_detector import YOLODetector
from src.tracking.tracker import VehicleTracker
from src.parking.parking_detector import ParkingDetector
from src.alert.alert_manager import AlertManager


# ============================================================
# ROI SELECTION WITH MOUSE
# ============================================================

# Global variables for mouse-based ROI drawing
roi_points = []
roi_drawing = False
roi_done = False


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for interactive ROI selection.

    Left-click to set top-left corner, release to set bottom-right corner.
    Draws a rectangle on the frame as the user drags.
    """
    global roi_points, roi_drawing, roi_done

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing - record top-left corner
        roi_points = [(x, y)]
        roi_drawing = True
        roi_done = False

    elif event == cv2.EVENT_MOUSEMOVE and roi_drawing:
        # Update preview while dragging
        if len(roi_points) == 1:
            roi_points.append((x, y))
        else:
            roi_points[1] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing - record bottom-right corner
        if len(roi_points) == 1:
            roi_points.append((x, y))
        else:
            roi_points[1] = (x, y)
        roi_drawing = False
        roi_done = True


def select_roi(frame):
    """
    Interactive ROI selection mode.

    Pauses video and lets the user draw a rectangle for the
    No-Parking Zone using the mouse.

    Args:
        frame (numpy.ndarray): Current video frame to display.

    Returns:
        tuple or None: (x1, y1, x2, y2) if ROI selected, None if cancelled.
    """
    global roi_points, roi_drawing, roi_done

    roi_points = []
    roi_drawing = False
    roi_done = False

    window_name = "Draw No-Parking Zone (drag mouse, then press ENTER to confirm, ESC to cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n[ROI] Draw the No-Parking Zone:")
    print("[ROI] - Click and drag to draw rectangle")
    print("[ROI] - Press ENTER to confirm")
    print("[ROI] - Press ESC to cancel\n")

    while True:
        display = frame.copy()

        # Draw current selection
        if len(roi_points) == 2:
            cv2.rectangle(display, roi_points[0], roi_points[1], (255, 255, 0), 2)

            # Draw semi-transparent overlay
            overlay = display.copy()
            cv2.rectangle(overlay, roi_points[0], roi_points[1], (255, 255, 0), -1)
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

        # Instructions
        cv2.putText(display, "Draw ROI: Click + Drag | ENTER = Confirm | ESC = Cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and roi_done:  # ENTER
            cv2.destroyWindow(window_name)
            x1 = min(roi_points[0][0], roi_points[1][0])
            y1 = min(roi_points[0][1], roi_points[1][1])
            x2 = max(roi_points[0][0], roi_points[1][0])
            y2 = max(roi_points[0][1], roi_points[1][1])
            print(f"[ROI] Selected: ({x1}, {y1}, {x2}, {y2})")
            return (x1, y1, x2, y2)

        elif key == 27:  # ESC
            cv2.destroyWindow(window_name)
            print("[ROI] Selection cancelled")
            return None


# ============================================================
# MAIN PIPELINE
# ============================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Illegal Parking Detection System using Computer Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                              # Default video
  python src/main.py --source data/video.mp4      # Custom video
  python src/main.py --source 0                   # Webcam
  python src/main.py --threshold 10               # 10s parking threshold
  python src/main.py --no-save                    # Don't save output video
        """
    )
    parser.add_argument('--source', type=str, default=VIDEO_SOURCE,
                        help='Video source: file path or camera index (default: data/video.mp4)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Parking time threshold in seconds (default: from config)')
    parser.add_argument('--model', type=str, default=None,
                        help='YOLOv8 model path (default: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=None,
                        help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable output video saving')
    parser.add_argument('--draw-roi', action='store_true',
                        help='Start with interactive ROI drawing')
    return parser.parse_args()


def main():
    """
    Main function - runs the complete illegal parking detection pipeline.

    Pipeline flow:
    1. Initialize all modules (Detector, Tracker, ParkingDetector, AlertManager)
    2. Open video source
    3. For each frame:
       a. Detect vehicles (YOLO)
       b. Track vehicles (DeepSORT)
       c. Check parking violations (Parking Logic)
       d. Draw annotations (Alert)
       e. Display result
    4. Handle user input (quit, ROI drawing, pause)
    5. Clean up resources
    """
    args = parse_args()

    print("=" * 60)
    print("  Illegal Parking Detection System")
    print("  ITS Project - Computer Vision")
    print("=" * 60)

    # ----------------------------------------------------------
    # STEP 1: Initialize modules
    # ----------------------------------------------------------
    print("\n[Main] Initializing modules...\n")

    # Detection module
    detector_kwargs = {}
    if args.model:
        detector_kwargs['model_path'] = args.model
    if args.confidence:
        detector_kwargs['confidence'] = args.confidence
    detector = YOLODetector(**detector_kwargs)

    # Tracking module
    tracker = VehicleTracker()

    # Parking detection module
    parking_kwargs = {}
    if args.threshold:
        parking_kwargs['time_threshold'] = args.threshold
    parking_detector = ParkingDetector(**parking_kwargs)

    # Alert module
    alert_manager = AlertManager()

    print("\n[Main] All modules initialized successfully!\n")

    # ----------------------------------------------------------
    # STEP 2: Open video source
    # ----------------------------------------------------------
    source = args.source
    # Check if source is a camera index (integer string)
    try:
        source = int(source)
        print(f"[Main] Opening camera: {source}")
    except ValueError:
        print(f"[Main] Opening video file: {source}")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        print("[ERROR] Please check the file path or camera index.")
        sys.exit(1)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Main] Video: {frame_width}x{frame_height} @ {video_fps:.1f} FPS")
    if total_frames > 0:
        print(f"[Main] Total frames: {total_frames}")

    # ----------------------------------------------------------
    # STEP 3: Setup output video writer (optional)
    # ----------------------------------------------------------
    writer = None
    save_output = SAVE_OUTPUT and not args.no_save

    if save_output:
        output_path = OUTPUT_VIDEO
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, video_fps,
                                 (frame_width, frame_height))
        print(f"[Main] Saving output to: {output_path}")

    # ----------------------------------------------------------
    # STEP 4: Interactive ROI drawing (if requested)
    # ----------------------------------------------------------
    if args.draw_roi:
        ret, first_frame = cap.read()
        if ret:
            roi = select_roi(first_frame)
            if roi:
                parking_detector.set_roi(*roi)
            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ----------------------------------------------------------
    # STEP 5: Main processing loop
    # ----------------------------------------------------------
    print("\n[Main] Starting detection pipeline...")
    print("[Main] Press 'q' to quit, 'r' to draw ROI, 'p' to pause\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps = 0.0
    prev_time = time.time()
    paused = False

    while True:
        # Handle pause
        if paused:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('p'):
                paused = False
                print("[Main] Resumed")
            elif key == ord('q'):
                break
            continue

        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("[Main] End of video reached")
            break

        frame_count += 1

        # --- PIPELINE START ---

        # Step A: Detect vehicles using YOLO
        detections = detector.detect(frame)

        # Step B: Track vehicles using DeepSORT
        tracked_vehicles = tracker.update(detections, frame)

        # Step C: Check parking violations
        vehicles_with_status = parking_detector.update(tracked_vehicles)

        # Step D: Draw annotations
        # Draw ROI zone
        frame = alert_manager.draw_roi(frame, parking_detector.get_roi())

        # Draw vehicle bounding boxes and labels
        frame = alert_manager.draw_vehicles(frame, vehicles_with_status)

        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed > 0:
            fps = 1.0 / elapsed
        prev_time = current_time

        # Draw statistics
        stats = parking_detector.get_stats()
        frame = alert_manager.draw_stats(frame, fps, stats)

        # Draw instructions
        frame = alert_manager.draw_instructions(frame)

        # --- PIPELINE END ---

        # Display frame
        cv2.imshow(WINDOW_NAME, frame)

        # Save frame to output video
        if writer is not None:
            writer.write(frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit
            print("[Main] Quit requested")
            break

        elif key == ord('r'):
            # Draw new ROI
            roi = select_roi(frame)
            if roi:
                parking_detector.set_roi(*roi)

        elif key == ord('p'):
            # Pause
            paused = True
            print("[Main] Paused - press 'p' to resume")

        elif key == ord('s'):
            # Screenshot
            screenshot_path = f"data/screenshot_{frame_count}.png"
            os.makedirs("data", exist_ok=True)
            cv2.imwrite(screenshot_path, frame)
            print(f"[Main] Screenshot saved: {screenshot_path}")

    # ----------------------------------------------------------
    # STEP 6: Cleanup
    # ----------------------------------------------------------
    print("\n[Main] Shutting down...")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[Main] Output video saved: {OUTPUT_VIDEO}")

    cv2.destroyAllWindows()

    # Print final statistics
    stats = parking_detector.get_stats()
    print(f"\n{'=' * 60}")
    print(f"  Final Statistics")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total violations detected: {stats['violations']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
