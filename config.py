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
# Sablon strftime pentru numele fisierelor (folosit la generarea numelor de descarcat
# si la citirea datei din fisierele locale). %Y an, %m luna, %d zi, %H ora, %M minut.
FTP_FILE_FORMAT = "h60_%Y%m%d_%H%M_fdk.nc.gz"

FTP_USER = os.getenv("FTP_USER")
FTP_PASS = os.getenv("FTP_PASS")
FTP_TIMEOUT = 30        # secunde
FTP_MAX_RETRIES = 3

# --- CREDENTIALE DESCARCARE DATE VOLUMETRICE ---
# NASA Earthdata — nivelele curente SWOT (via PODAAC)
EDL_USER = os.getenv("EDL_USER")
EDL_PASS = os.getenv("EDL_PASS")
# Copernicus Sentinel Hub OAuth client — nivelele curente Sentinel-2
SH_ID = os.getenv("SH_ID")
SH_SECRET = os.getenv("SH_SECRET")

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
RAIN_THRESHOLD_MIN = 0.1    # mm/h minim pentru afisare si volumetrie
RAIN_THRESHOLD_TRACKING = 0.5 # mm/h minim pentru tracking si metrici cinematice (Core Tracking)
RAIN_THRESHOLD_SEVERE = 5.0 # mm/h pentru vizualizarea ploii severe
RAIN_VMAX = 12.0            # Limita maximă pe scala de culori a hărții

# --- SETĂRI TRACKING (CINEMATICĂ) ---
MAX_TRACKING_DISTANCE_PX = 18  # Distanța maximă (pixeli) pentru matching între cadre

# --- SETĂRI VOLUMETRIE LAC ACUMULARE ---
# Coeficient de scurgere (fractiunea din precipitatie care ajunge in lac ca debit de bazin).
# Valoare constanta, simplificare; o rafinare ulterioara ar folosi SCS Curve Number.
RUNOFF_COEFFICIENT = 0.35

# --- BILANT HIDROLOGIC: iesiri (V_{t+1} = V_t + intrare - evacuare - evaporare) ---
# Evaporare de suprafata (mm/zi). 0 = ignorata (ferestre scurte de nowcast). Se seteaza din ET0
# FAO Penman-Monteith: pentru Romania ~4-5 mm/zi vara, ~0.5-1 mm/zi iarna.
EVAP_MM_PER_DAY = 0.0
# Debit de evacuare asumat la baraj (m^3/s). 0 = ignorat. Releasele operationale (Hidroelectrica,
# Portile de Fier) domina scaderile reale de nivel, dar nu sunt previzibile din vreme.
RESERVOIR_OUTFLOW_M3S = 0.0

# Restrange scopul aplicatiei la lacurile cu nivel curent din SWOT (elimina lacurile neacoperite,
# care ar porni oricum de la NNR). Scripturile de build vad in continuare setul complet.
RESERVOIRS_SWOT_COVERED_ONLY = True

# --- SETĂRI STREAMLIT ---
DEFAULT_ANIMATION_SPEED = 0.4