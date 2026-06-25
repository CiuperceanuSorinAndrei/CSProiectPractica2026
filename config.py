import os
from dotenv import load_dotenv

load_dotenv()

# --- FOLDERE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_GEO_DIR = os.path.join(BASE_DIR, "data", "geofiles") 
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_GEO_DIR, exist_ok=True)

# --- SETĂRI FTP H-SAF ---
FTP_HOST = "ftphsaf.meteoam.it"
FTP_BASE_FOLDER = "h60/h60_cur_mon_data"

FTP_USER = os.getenv("FTP_USER")
FTP_PASS = os.getenv("FTP_PASS")
FTP_TIMEOUT = 30        # secunde
FTP_MAX_RETRIES = 3

# --- SETĂRI REGIUNE (LOCAȚII PREDEFINITE) ---
PREDEFINED_LOCATIONS = {
    "Craiova ": {"lat": 44.33, "lon": 23.79},
    "București": {"lat": 44.43, "lon": 26.10},
    "Timișoara": {"lat": 45.75, "lon": 21.22},
    "Bazinul Jiu": {"lat": 44.55, "lon": 23.50}, 
    "Bazinul Olt": {"lat": 44.80, "lon": 24.30},
    "Manual (Introducere coordonate)": None
}

# Setări implicite la deschiderea aplicației
DEFAULT_LOCATION_KEY = "Craiova (Centru)"
DEFAULT_RADIUS_KM = 500

# --- SETĂRI METEO & ALGORITM ---
RAIN_THRESHOLD_MIN = 0.1    # mm/h minim pentru a detecta formarea unei celule
RAIN_THRESHOLD_SEVERE = 5.0 # mm/h pentru vizualizarea ploii severe
RAIN_VMAX = 12.0            # Limita maximă pe scala de culori a hărții

# --- SETĂRI TRACKING (CINEMATICĂ) ---
MAX_TRACKING_DISTANCE_PX = 18  # Distanța maximă (pixeli) pentru matching între cadre

# --- SETĂRI STREAMLIT ---
DEFAULT_ANIMATION_SPEED = 0.4