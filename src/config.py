import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 1. Directory Structure
BASE_DIR = str(Path(__file__).resolve().parent.parent)
DATA_RAW_DIR = str(Path(BASE_DIR) / "data" / "raw")
DATA_GEO_DIR = str(Path(BASE_DIR) / "data" / "geofiles")

Path(DATA_RAW_DIR).mkdir(parents=True, exist_ok=True)
Path(DATA_GEO_DIR).mkdir(parents=True, exist_ok=True)

# 2. FTP Configuration
FTP_HOST = "ftphsaf.meteoam.it"
FTP_BASE_FOLDER = "h60/h60_cur_mon_data"
FTP_FILE_FORMAT = "h60_%Y%m%d_%H%M_fdk.nc.gz"
FTP_TIMEOUT = 30
FTP_MAX_RETRIES = 3

# 3. Location Targets
PREDEFINED_LOCATIONS = {
    "Craiova": {"lat": 44.33, "lon": 23.79},
    "Bucharest": {"lat": 44.43, "lon": 26.10},
    "Timisoara": {"lat": 45.75, "lon": 21.22},
    "Jiu Basin": {"lat": 44.55, "lon": 23.50}, 
    "Olt Basin": {"lat": 44.80, "lon": 24.30},
    "Manual (Enter coordinates)": None
}



# 4. Meteorology Thresholds
RAIN_THRESHOLD_MIN = 1.0
RAIN_THRESHOLD_TRACKING = 1.0

RAIN_VMAX = 12.0
MAX_TRACKING_DISTANCE_PX = 18

# 5. External API Credentials
EDL_USER = os.getenv("EDL_USER")
EDL_PASS = os.getenv("EDL_PASS")
SH_ID = os.getenv("SH_ID")
SH_SECRET = os.getenv("SH_SECRET")

# 6. Hydrology Parameters
RUNOFF_COEFFICIENT = 0.35

RESERVOIRS_SWOT_COVERED_ONLY = True
