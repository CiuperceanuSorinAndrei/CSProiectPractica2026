"""Dashboard package: UI structure, state management, and Dash nowcasting application."""
from src.dashboard.frame_store import FrameStore
from src.dashboard.frame_history import FrameHistory
from src.dashboard.dashboard_layout import DashboardLayout
from src.dashboard.nowcasting_dashboard import NowcastingDashboard
from src.dashboard.constants import (
    DATA_DIR, MANUAL_LOCATION, DEFAULT_TIME_RANGE,
    MAP_ZOOM_MIN, MAP_ZOOM_MAX, MAP_ZOOM_DEFAULT,
    ROI_RADIUS_MIN, ROI_RADIUS_MAX, ROI_RADIUS_DEFAULT,
)

__all__ = [
    "NowcastingDashboard", "DashboardLayout", "FrameStore", "FrameHistory",
    "DATA_DIR", "MANUAL_LOCATION", "DEFAULT_TIME_RANGE",
    "MAP_ZOOM_MIN", "MAP_ZOOM_MAX", "MAP_ZOOM_DEFAULT",
    "ROI_RADIUS_MIN", "ROI_RADIUS_MAX", "ROI_RADIUS_DEFAULT",
]
