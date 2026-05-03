"""
Configuration file for Illegal Parking Detection System.
Contains all tunable parameters for detection, tracking, parking logic, and alert.
"""

# ============================================================
# VIDEO INPUT
# ============================================================
VIDEO_SOURCE = "data/video.mp4"       # Path to input video file
OUTPUT_VIDEO = "data/output.mp4"      # Path to save output video (set None to disable)
SAVE_OUTPUT = True                    # Whether to save annotated output video

# ============================================================
# YOLO DETECTION
# ============================================================
YOLO_MODEL = "yolov8n.pt"            # YOLOv8 model variant (n/s/m/l/x)
CONFIDENCE_THRESHOLD = 0.5           # Minimum confidence score to accept a detection
# COCO class IDs for vehicles: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]
# Human-readable names mapping
CLASS_NAMES = {
    2: "Car",
    3: "Motorbike",
    5: "Bus",
    7: "Truck"
}

# ============================================================
# DEEPSORT TRACKING
# ============================================================
MAX_AGE = 30                          # Max frames to keep a lost track alive
N_INIT = 3                            # Min detections before a track is confirmed
MAX_IOU_DISTANCE = 0.7                # Max IOU distance for matching

# ============================================================
# PARKING DETECTION LOGIC
# ============================================================
# Default No-Parking Zone ROI (x1, y1, x2, y2) - top-left and bottom-right
# This will be overridden if user draws ROI with mouse
NO_PARKING_ZONE = (200, 300, 800, 600)

# Time threshold in seconds - vehicle must stay in ROI longer than this
PARKING_TIME_THRESHOLD = 5.0

# Movement threshold in pixels - if vehicle center moves less than this, it's "stationary"
MOVEMENT_THRESHOLD = 15.0

# How often (in seconds) to check movement history for stationary detection
MOVEMENT_CHECK_INTERVAL = 1.0

# ============================================================
# ALERT & VISUALIZATION
# ============================================================
# Colors in BGR format
COLOR_NORMAL = (0, 255, 0)            # Green - normal vehicle
COLOR_WARNING = (0, 165, 255)         # Orange - vehicle in ROI but not yet violation
COLOR_VIOLATION = (0, 0, 255)         # Red - illegal parking detected
COLOR_ROI = (255, 255, 0)             # Cyan - No Parking Zone boundary
COLOR_TEXT = (255, 255, 255)          # White - text color

# Font settings
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Log file for violations
LOG_FILE = "data/violations.log"
ENABLE_LOGGING = True

# ============================================================
# DISPLAY
# ============================================================
SHOW_FPS = True                       # Display FPS counter on video
WINDOW_NAME = "Illegal Parking Detection - ITS"
