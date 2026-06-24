"""Constante partajate de UI-ul si callback-urile dashboard-ului."""
import os

# Directorul cu cadrele .nc locale
DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Cheia de locatie care activeaza introducerea manuala a coordonatelor
MANUAL_LOCATION = "Manual (Introducere coordonate)"

# Intervalul implicit pentru slider-ul de cadre (modul istoric)
DEFAULT_TIME_RANGE = {"start": "2026-06-13T22:00:00", "end": "2026-06-14T23:00:00"}

# Limitele controalelor de vizualizare (clamping in update_dashboard)
MAP_ZOOM_MIN, MAP_ZOOM_MAX, MAP_ZOOM_DEFAULT = 100, 700, 500
ROI_RADIUS_MIN, ROI_RADIUS_MAX, ROI_RADIUS_DEFAULT = 5, 200, 30
