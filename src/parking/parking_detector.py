"""
Parking Detection Module - Illegal Parking Logic
=================================================
Implements the core logic to determine if a tracked vehicle
is illegally parked. A vehicle is considered illegally parked if:

1. It is inside a defined "No Parking Zone" (ROI)
2. It has minimal movement (stationary)
3. It has remained stationary for longer than the time threshold

This module maintains a history of vehicle positions and timestamps
to make accurate determinations.
"""

import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.config import (
    NO_PARKING_ZONE,
    PARKING_TIME_THRESHOLD,
    MOVEMENT_THRESHOLD,
)


class ParkingDetector:
    """
    Illegal parking detection engine.

    Monitors tracked vehicles and determines if any are illegally
    parked based on their position (inside ROI), movement (stationary),
    and duration (exceeds time threshold).

    Attributes:
        roi (tuple): No-parking zone coordinates (x1, y1, x2, y2).
        time_threshold (float): Seconds a vehicle must be stationary to trigger.
        movement_threshold (float): Max pixel movement to be "stationary".
        vehicle_history (dict): Maps track_id -> vehicle state history.
    """

    def __init__(self, roi=NO_PARKING_ZONE, time_threshold=PARKING_TIME_THRESHOLD,
                 movement_threshold=MOVEMENT_THRESHOLD):
        """
        Initialize the parking detector.

        Args:
            roi (tuple): (x1, y1, x2, y2) defining the No-Parking Zone.
            time_threshold (float): Time in seconds before flagging violation.
            movement_threshold (float): Maximum pixel displacement to
                                         consider a vehicle stationary.
        """
        self.roi = roi
        self.time_threshold = time_threshold
        self.movement_threshold = movement_threshold

        # Vehicle history: {track_id: {
        #     'first_seen': timestamp,
        #     'last_position': (cx, cy),
        #     'stationary_since': timestamp or None,
        #     'is_violation': bool,
        #     'in_roi': bool
        # }}
        self.vehicle_history = {}

        print(f"[Parking] Initialized parking detector")
        print(f"[Parking] ROI: {self.roi}")
        print(f"[Parking] Time threshold: {self.time_threshold}s")
        print(f"[Parking] Movement threshold: {self.movement_threshold}px")

    def set_roi(self, x1, y1, x2, y2):
        """
        Update the No-Parking Zone ROI.

        Args:
            x1, y1 (int): Top-left corner coordinates.
            x2, y2 (int): Bottom-right corner coordinates.
        """
        self.roi = (x1, y1, x2, y2)
        print(f"[Parking] ROI updated to: {self.roi}")

    def get_roi(self):
        """
        Get the current ROI coordinates.

        Returns:
            tuple: (x1, y1, x2, y2) of the No-Parking Zone.
        """
        return self.roi

    def _get_center(self, bbox):
        """
        Calculate the center point of a bounding box.

        Args:
            bbox (list): [x1, y1, x2, y2] bounding box coordinates.

        Returns:
            tuple: (cx, cy) center coordinates.
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return (cx, cy)

    def _is_inside_roi(self, bbox):
        """
        Check if the center of a bounding box is inside the ROI.

        Uses the center point of the vehicle's bounding box to determine
        if the vehicle is within the No-Parking Zone.

        Args:
            bbox (list): [x1, y1, x2, y2] bounding box coordinates.

        Returns:
            bool: True if vehicle center is inside ROI.
        """
        cx, cy = self._get_center(bbox)
        rx1, ry1, rx2, ry2 = self.roi
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def _calculate_displacement(self, pos1, pos2):
        """
        Calculate Euclidean distance between two positions.

        Args:
            pos1 (tuple): (x, y) first position.
            pos2 (tuple): (x, y) second position.

        Returns:
            float: Euclidean distance in pixels.
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx ** 2 + dy ** 2) ** 0.5

    def update(self, tracked_vehicles):
        """
        Update parking detection state with current tracked vehicles.

        For each tracked vehicle, this method:
        1. Checks if it's inside the ROI
        2. Calculates movement since last frame
        3. Determines if it's stationary
        4. Flags as violation if stationary for too long in ROI

        Args:
            tracked_vehicles (list[dict]): Tracked vehicles from VehicleTracker.update().
                Each dict has: 'track_id', 'bbox', 'class_name', 'confirmed'.

        Returns:
            list[dict]: Enhanced vehicle list with parking status added:
                - 'in_roi' (bool): Whether vehicle is in No-Parking Zone
                - 'is_violation' (bool): Whether illegal parking is detected
                - 'duration' (float): Time spent stationary in ROI (seconds)
                - 'status' (str): 'normal', 'warning', or 'violation'
        """
        current_time = time.time()
        results = []

        # Track which IDs are still active
        active_ids = set()

        for vehicle in tracked_vehicles:
            track_id = vehicle['track_id']
            bbox = vehicle['bbox']
            active_ids.add(track_id)

            center = self._get_center(bbox)
            in_roi = self._is_inside_roi(bbox)

            # Initialize history for new vehicles
            if track_id not in self.vehicle_history:
                self.vehicle_history[track_id] = {
                    'first_seen': current_time,
                    'last_position': center,
                    'stationary_since': current_time if in_roi else None,
                    'is_violation': False,
                    'in_roi': in_roi
                }
            else:
                history = self.vehicle_history[track_id]
                last_pos = history['last_position']

                # Calculate movement
                displacement = self._calculate_displacement(center, last_pos)

                if in_roi:
                    if displacement < self.movement_threshold:
                        # Vehicle is stationary in ROI
                        if history['stationary_since'] is None:
                            history['stationary_since'] = current_time
                    else:
                        # Vehicle moved significantly - reset stationary timer
                        history['stationary_since'] = current_time
                else:
                    # Vehicle left ROI - reset all parking state
                    history['stationary_since'] = None
                    history['is_violation'] = False

                # Update position
                history['last_position'] = center
                history['in_roi'] = in_roi

            # Calculate duration and status
            history = self.vehicle_history[track_id]
            duration = 0.0
            status = 'normal'

            if in_roi and history['stationary_since'] is not None:
                duration = current_time - history['stationary_since']

                if duration >= self.time_threshold:
                    # VIOLATION: Stationary in ROI beyond threshold
                    history['is_violation'] = True
                    status = 'violation'
                else:
                    # WARNING: In ROI but not yet a violation
                    status = 'warning'

            # Build result with parking info
            result = {
                **vehicle,
                'in_roi': in_roi,
                'is_violation': history['is_violation'],
                'duration': round(duration, 1),
                'status': status
            }
            results.append(result)

        # Clean up history for vehicles no longer tracked
        stale_ids = set(self.vehicle_history.keys()) - active_ids
        for stale_id in stale_ids:
            del self.vehicle_history[stale_id]

        return results

    def get_violation_count(self):
        """
        Get the number of currently active violations.

        Returns:
            int: Number of vehicles currently flagged as violations.
        """
        return sum(
            1 for v in self.vehicle_history.values()
            if v.get('is_violation', False)
        )

    def get_stats(self):
        """
        Get current parking detection statistics.

        Returns:
            dict: Statistics including total tracked, in ROI, and violations.
        """
        total = len(self.vehicle_history)
        in_roi = sum(1 for v in self.vehicle_history.values() if v.get('in_roi', False))
        violations = self.get_violation_count()

        return {
            'total_tracked': total,
            'in_roi': in_roi,
            'violations': violations
        }
