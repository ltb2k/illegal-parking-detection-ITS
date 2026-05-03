"""
Alert Module - Visualization & Logging
=======================================
Handles all visual annotation on video frames and logging of
violation events. Responsible for:

- Drawing bounding boxes (color-coded by status)
- Displaying tracking IDs and class names
- Drawing the No-Parking Zone ROI
- Showing "ILLEGAL PARKING" alerts
- Displaying FPS and statistics
- Logging violations to file
"""

import cv2
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.config import (
    COLOR_NORMAL, COLOR_WARNING, COLOR_VIOLATION, COLOR_ROI, COLOR_TEXT,
    FONT_SCALE, FONT_THICKNESS, LOG_FILE, ENABLE_LOGGING, SHOW_FPS
)


class AlertManager:
    """
    Visual alert and logging manager.

    Draws annotations on video frames to visualize detection results
    and parking violations. Also handles logging of violation events
    to a file for record-keeping.

    Attributes:
        show_fps (bool): Whether to display FPS counter.
        enable_logging (bool): Whether to log violations to file.
        log_file (str): Path to the violation log file.
        logged_violations (set): Set of track IDs already logged
                                  (prevents duplicate log entries).
    """

    def __init__(self, show_fps=SHOW_FPS, enable_logging=ENABLE_LOGGING, log_file=LOG_FILE):
        """
        Initialize the alert manager.

        Args:
            show_fps (bool): Display FPS on video frames.
            enable_logging (bool): Enable violation logging to file.
            log_file (str): Path to the log file.
        """
        self.show_fps = show_fps
        self.enable_logging = enable_logging
        self.log_file = log_file
        self.logged_violations = set()  # Avoid duplicate logs

        # Ensure log directory exists
        if self.enable_logging:
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            print(f"[Alert] Logging enabled -> {self.log_file}")

        print(f"[Alert] Alert manager initialized")

    def draw_roi(self, frame, roi):
        """
        Draw the No-Parking Zone on the frame.

        Draws a dashed rectangle with a semi-transparent overlay
        and "NO PARKING ZONE" label.

        Args:
            frame (numpy.ndarray): Video frame to annotate.
            roi (tuple): (x1, y1, x2, y2) ROI coordinates.

        Returns:
            numpy.ndarray: Annotated frame.
        """
        x1, y1, x2, y2 = [int(v) for v in roi]

        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_ROI, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ROI, 2)

        # Draw label
        label = "NO PARKING ZONE"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_x = x1 + 5
        label_y = y1 - 10 if y1 - 10 > 20 else y1 + label_size[1] + 10

        # Background for text
        cv2.rectangle(frame,
                      (label_x - 2, label_y - label_size[1] - 5),
                      (label_x + label_size[0] + 5, label_y + 5),
                      COLOR_ROI, -1)
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame

    def draw_vehicles(self, frame, vehicles):
        """
        Draw bounding boxes and labels for all tracked vehicles.

        Color-coding:
        - Green: Normal (outside ROI or just entered)
        - Orange: Warning (in ROI, not yet violation)
        - Red: Violation (illegally parked)

        Args:
            frame (numpy.ndarray): Video frame to annotate.
            vehicles (list[dict]): Vehicle list from ParkingDetector.update().

        Returns:
            numpy.ndarray: Annotated frame.
        """
        for vehicle in vehicles:
            bbox = vehicle['bbox']
            track_id = vehicle['track_id']
            class_name = vehicle['class_name']
            status = vehicle.get('status', 'normal')
            duration = vehicle.get('duration', 0)
            is_violation = vehicle.get('is_violation', False)

            # Convert bbox to integers
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Select color based on status
            if status == 'violation':
                color = COLOR_VIOLATION
                thickness = 3
            elif status == 'warning':
                color = COLOR_WARNING
                thickness = 2
            else:
                color = COLOR_NORMAL
                thickness = 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Build label text
            label = f"ID:{track_id} {class_name}"
            if status == 'warning':
                label += f" ({duration}s)"
            elif status == 'violation':
                label += f" ({duration}s)"

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                            FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(frame,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0] + 5, y1),
                          color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                        COLOR_TEXT, FONT_THICKNESS)

            # Draw ILLEGAL PARKING alert for violations
            if is_violation:
                self._draw_violation_alert(frame, x1, y1, x2, y2)
                self._log_violation(track_id, class_name, duration)

        return frame

    def _draw_violation_alert(self, frame, x1, y1, x2, y2):
        """
        Draw a prominent "ILLEGAL PARKING" alert on a violated vehicle.

        Args:
            frame (numpy.ndarray): Video frame.
            x1, y1, x2, y2 (int): Bounding box coordinates.
        """
        alert_text = "ILLEGAL PARKING"
        text_size, _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # Position alert below the bounding box
        text_x = x1
        text_y = y2 + text_size[1] + 10

        # Ensure text stays within frame
        frame_h, frame_w = frame.shape[:2]
        if text_y > frame_h - 5:
            text_y = y1 - 30  # Place above if no room below

        # Draw alert background (red)
        cv2.rectangle(frame,
                      (text_x - 2, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5),
                      COLOR_VIOLATION, -1)

        # Draw alert text (white on red)
        cv2.putText(frame, alert_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

    def _log_violation(self, track_id, class_name, duration):
        """
        Log a violation event to file (once per vehicle).

        Args:
            track_id (int): Vehicle tracking ID.
            class_name (str): Vehicle class name.
            duration (float): Time parked in violation zone.
        """
        if not self.enable_logging:
            return

        # Only log once per track ID
        if track_id in self.logged_violations:
            return

        self.logged_violations.add(track_id)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"[{timestamp}] VIOLATION - "
            f"Track ID: {track_id}, "
            f"Vehicle: {class_name}, "
            f"Duration: {duration}s\n"
        )

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            print(f"[Alert] Violation logged: ID {track_id} ({class_name})")
        except IOError as e:
            print(f"[Alert] Failed to write log: {e}")

    def draw_stats(self, frame, fps, stats):
        """
        Draw statistics overlay on the frame.

        Displays FPS, vehicle count, and violation count in the
        top-left corner of the frame.

        Args:
            frame (numpy.ndarray): Video frame to annotate.
            fps (float): Current frames per second.
            stats (dict): Statistics from ParkingDetector.get_stats().

        Returns:
            numpy.ndarray: Annotated frame.
        """
        frame_h, frame_w = frame.shape[:2]

        # Build stats text lines
        lines = []
        if self.show_fps:
            lines.append(f"FPS: {fps:.1f}")
        lines.append(f"Vehicles: {stats.get('total_tracked', 0)}")
        lines.append(f"In Zone: {stats.get('in_roi', 0)}")
        lines.append(f"Violations: {stats.get('violations', 0)}")

        # Draw background panel
        panel_w = 200
        panel_h = len(lines) * 30 + 15
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h),
                      COLOR_ROI, 1)

        # Draw each line
        for i, line in enumerate(lines):
            y_pos = 35 + i * 30
            # Color violations count in red if > 0
            color = COLOR_TEXT
            if 'Violations' in line and stats.get('violations', 0) > 0:
                color = COLOR_VIOLATION

            cv2.putText(frame, line, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def draw_instructions(self, frame):
        """
        Draw keyboard instructions on the frame.

        Args:
            frame (numpy.ndarray): Video frame.

        Returns:
            numpy.ndarray: Annotated frame.
        """
        frame_h, frame_w = frame.shape[:2]
        instructions = [
            "Press 'q' - Quit",
            "Press 'r' - Draw ROI",
            "Press 'p' - Pause/Resume",
        ]

        for i, text in enumerate(instructions):
            y_pos = frame_h - 20 - i * 25
            cv2.putText(frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)

        return frame
