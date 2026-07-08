# Dashboard constants
import os

DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Location key for manual coordinates
MANUAL_LOCATION = "Manual (Enter coordinates)"

# Default slider range
DEFAULT_TIME_RANGE = {"start": "2026-06-13T22:00:00", "end": "2026-06-14T23:00:00"}

# Visualization limits
MAP_ZOOM_MIN, MAP_ZOOM_MAX, MAP_ZOOM_DEFAULT = 10, 700, 500
ROI_RADIUS_MIN, ROI_RADIUS_MAX, ROI_RADIUS_DEFAULT = 5, 200, 30
