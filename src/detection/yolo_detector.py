"""
Detection Module - YOLOv8 Vehicle Detector
==========================================
Uses Ultralytics YOLOv8 to detect vehicles in video frames.
Supports: Car, Motorbike, Bus, Truck.

Returns bounding boxes, confidence scores, and class labels
for downstream tracking and parking logic.
"""

import sys
import os

# Add project root to path for config imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ultralytics import YOLO
from config.config import (
    YOLO_MODEL,
    CONFIDENCE_THRESHOLD,
    VEHICLE_CLASSES,
    CLASS_NAMES
)


class YOLODetector:
    """
    Vehicle detector using YOLOv8.

    This class wraps the Ultralytics YOLO model to perform
    object detection on video frames, filtering results to
    only include vehicle classes of interest.

    Attributes:
        model (YOLO): Loaded YOLOv8 model instance.
        confidence (float): Minimum confidence threshold for detections.
        vehicle_classes (list): COCO class IDs to detect.
        class_names (dict): Mapping from class ID to human-readable name.
    """

    def __init__(self, model_path=YOLO_MODEL, confidence=CONFIDENCE_THRESHOLD):
        """
        Initialize the YOLOv8 detector.

        Args:
            model_path (str): Path to YOLOv8 model weights.
                              Defaults to config value. Will auto-download
                              if not found locally.
            confidence (float): Minimum confidence threshold (0.0 - 1.0).
        """
        print(f"[Detection] Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_classes = VEHICLE_CLASSES
        self.class_names = CLASS_NAMES
        print(f"[Detection] Model loaded. Confidence threshold: {self.confidence}")
        print(f"[Detection] Detecting classes: {[self.class_names[c] for c in self.vehicle_classes]}")

    def detect(self, frame):
        """
        Run YOLOv8 detection on a single frame.

        Processes the input frame through the YOLO model and filters
        results to only include vehicle detections above the confidence
        threshold.

        Args:
            frame (numpy.ndarray): Input video frame (BGR format from OpenCV).

        Returns:
            list[dict]: List of detection dictionaries, each containing:
                - 'bbox' (list[float]): [x1, y1, x2, y2] bounding box coordinates
                - 'confidence' (float): Detection confidence score
                - 'class_id' (int): COCO class ID
                - 'class_name' (str): Human-readable class name

        Example:
            >>> detector = YOLODetector()
            >>> detections = detector.detect(frame)
            >>> for det in detections:
            ...     print(f"{det['class_name']}: {det['confidence']:.2f}")
            Car: 0.92
            Truck: 0.87
        """
        # Run inference with verbose=False to suppress per-frame logs
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []

        # Process results - YOLOv8 returns a list of Results objects
        for result in results:
            boxes = result.boxes  # Boxes object containing detection results

            for box in boxes:
                # Extract class ID
                class_id = int(box.cls[0])

                # Filter: only keep vehicle classes
                if class_id not in self.vehicle_classes:
                    continue

                # Extract bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Extract confidence score
                confidence = float(box.conf[0])

                # Build detection dictionary
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, "Unknown")
                }

                detections.append(detection)

        return detections

    def get_detection_summary(self, detections):
        """
        Generate a summary string of current detections.

        Args:
            detections (list[dict]): List of detection dictionaries from detect().

        Returns:
            str: Summary string like "3 vehicles (2 Car, 1 Truck)"
        """
        if not detections:
            return "No vehicles detected"

        # Count by class
        counts = {}
        for det in detections:
            name = det['class_name']
            counts[name] = counts.get(name, 0) + 1

        total = len(detections)
        breakdown = ", ".join(f"{count} {name}" for name, count in counts.items())
        return f"{total} vehicles ({breakdown})"
