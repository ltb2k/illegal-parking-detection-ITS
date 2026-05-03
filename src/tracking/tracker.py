"""
Tracking Module - DeepSORT Vehicle Tracker
==========================================
Uses DeepSORT algorithm to assign persistent unique IDs to
detected vehicles and track them across consecutive frames.

This enables the parking logic to monitor individual vehicles
over time and determine if they remain stationary.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from deep_sort_realtime.deepsort_tracker import DeepSort
from config.config import MAX_AGE, N_INIT, MAX_IOU_DISTANCE


class VehicleTracker:
    """
    Multi-object tracker using DeepSORT.

    Wraps the deep-sort-realtime library to provide persistent
    tracking IDs for vehicles detected by YOLO. Each vehicle
    receives a unique ID that persists across frames even with
    brief occlusions.

    Attributes:
        tracker (DeepSort): DeepSORT tracker instance.
    """

    def __init__(self, max_age=MAX_AGE, n_init=N_INIT, max_iou_distance=MAX_IOU_DISTANCE):
        """
        Initialize the DeepSORT tracker.

        Args:
            max_age (int): Maximum number of frames to keep a track
                           alive without matching detections.
            n_init (int): Number of consecutive detections before a
                          track is considered "confirmed".
            max_iou_distance (float): Maximum IOU distance threshold
                                      for associating detections to tracks.
        """
        print(f"[Tracking] Initializing DeepSORT tracker")
        print(f"[Tracking] max_age={max_age}, n_init={n_init}, max_iou_dist={max_iou_distance}")

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder=None  # We provide our own embeddings to avoid downloading large models
        )

    def _generate_embeddings(self, detections, frame):
        """
        Generate simple appearance embeddings from cropped bounding box regions.

        Uses a lightweight approach: resize each crop to a fixed size,
        flatten, and normalize to create a feature vector.

        Args:
            detections (list[dict]): YOLO detections with 'bbox' key.
            frame (numpy.ndarray): Current video frame.

        Returns:
            list[numpy.ndarray]: List of embedding vectors.
        """
        import numpy as np

        embeddings = []
        h, w = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Clamp coordinates to frame boundaries
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            if x2 <= x1 or y2 <= y1:
                # Invalid crop, use zero embedding
                embeddings.append(np.zeros(128))
                continue

            # Crop and resize to fixed size
            crop = frame[y1:y2, x1:x2]
            try:
                import cv2
                crop_resized = cv2.resize(crop, (32, 32))
                # Flatten and normalize to create a simple feature vector
                feature = crop_resized.flatten().astype(np.float32)
                # Reduce dimensionality by averaging blocks
                feature = feature.reshape(-1, feature.shape[0] // 128).mean(axis=1)
                # L2 normalize
                norm = np.linalg.norm(feature)
                if norm > 0:
                    feature = feature / norm
                embeddings.append(feature)
            except Exception:
                embeddings.append(np.zeros(128))

        return embeddings

    def update(self, detections, frame):
        """
        Update tracker with new detections from current frame.

        Converts YOLO detections into the format required by DeepSORT,
        generates simple appearance embeddings, performs the tracking
        update, and returns tracked objects with persistent IDs.

        Args:
            detections (list[dict]): Detections from YOLODetector.detect().
                Each dict has: 'bbox', 'confidence', 'class_id', 'class_name'.
            frame (numpy.ndarray): Current video frame (used for
                appearance feature extraction).

        Returns:
            list[dict]: List of tracked vehicle dictionaries, each containing:
                - 'track_id' (int): Unique persistent tracking ID
                - 'bbox' (list[float]): [x1, y1, x2, y2] bounding box
                - 'class_name' (str): Vehicle class name
                - 'confirmed' (bool): Whether the track is confirmed

        Example:
            >>> tracker = VehicleTracker()
            >>> tracked = tracker.update(detections, frame)
            >>> for obj in tracked:
            ...     print(f"ID {obj['track_id']}: {obj['class_name']}")
            ID 1: Car
            ID 3: Truck
        """
        if not detections:
            # Still need to update tracker with empty detections
            # so it can age out old tracks
            tracks = self.tracker.update_tracks([], frame=frame)
            return self._format_tracks(tracks)

        # Generate embeddings for each detection
        embeds = self._generate_embeddings(detections, frame)

        # Convert detections to DeepSORT format
        # DeepSORT expects: list of ([x1, y1, w, h], confidence, class_name)
        deepsort_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            confidence = det['confidence']
            class_name = det['class_name']

            # Format: ([left, top, width, height], confidence, detection_class)
            deepsort_detections.append(
                ([x1, y1, w, h], confidence, class_name)
            )

        # Update tracker with detections and embeddings
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame, embeds=embeds)

        return self._format_tracks(tracks)

    def _format_tracks(self, tracks):
        """
        Convert DeepSORT track objects to standardized dictionaries.

        Args:
            tracks: List of Track objects from DeepSORT.

        Returns:
            list[dict]: Formatted track dictionaries.
        """
        tracked_vehicles = []

        for track in tracks:
            # Only process confirmed tracks (seen for >= n_init frames)
            if not track.is_confirmed():
                continue

            # Get track ID
            track_id = track.track_id

            # Get bounding box in [x1, y1, x2, y2] format
            ltrb = track.to_ltrb()  # left, top, right, bottom
            bbox = [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]

            # Get class name from detection
            class_name = track.det_class if track.det_class else "Vehicle"

            tracked_vehicles.append({
                'track_id': track_id,
                'bbox': bbox,
                'class_name': class_name,
                'confirmed': True
            })

        return tracked_vehicles

    def get_active_track_count(self):
        """
        Get the number of currently active (confirmed) tracks.

        Returns:
            int: Number of active tracked vehicles.
        """
        if not hasattr(self.tracker, 'tracks'):
            return 0
        return sum(1 for t in self.tracker.tracks if t.is_confirmed())
