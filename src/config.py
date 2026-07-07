import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = str(Path(__file__).resolve().parent.parent)
DATA_RAW_DIR = str(Path(BASE_DIR) / "data" / "raw")
DATA_GEO_DIR = str(Path(BASE_DIR) / "data" / "geofiles")

Path(DATA_RAW_DIR).mkdir(parents=True, exist_ok=True)
Path(DATA_GEO_DIR).mkdir(parents=True, exist_ok=True)

FTP_HOST = "ftphsaf.meteoam.it"
FTP_BASE_FOLDER = "h60/h60_cur_mon_data"
FTP_FILE_FORMAT = "h60_%Y%m%d_%H%M_fdk.nc.gz"
FTP_TIMEOUT = 30        # secunde
FTP_MAX_RETRIES = 3

PREDEFINED_LOCATIONS = {
    "Craiova": {"lat": 44.33, "lon": 23.79},
    "București": {"lat": 44.43, "lon": 26.10},
    "Timișoara": {"lat": 45.75, "lon": 21.22},
    "Bazinul Jiu": {"lat": 44.55, "lon": 23.50}, 
    "Bazinul Olt": {"lat": 44.80, "lon": 24.30},
    "Manual (Introducere coordonate)": None
}

DEFAULT_LOCATION_KEY = "Craiova (Centru)"
DEFAULT_RADIUS_KM = 500

RAIN_THRESHOLD_MIN = 1.0
RAIN_THRESHOLD_TRACKING = 1.0
RAIN_THRESHOLD_SEVERE = 5.0
RAIN_VMAX = 12.0

MAX_TRACKING_DISTANCE_PX = 18

DEFAULT_ANIMATION_SPEED = 0.4
