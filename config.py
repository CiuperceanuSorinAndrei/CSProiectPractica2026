import os
from dotenv import load_dotenv

load_dotenv()

# --- FOLDERE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(DATA_RAW_DIR, exist_ok=True)

# --- SETĂRI FTP H-SAF ---
FTP_HOST = "ftphsaf.meteoam.it"
FTP_BASE_FOLDER = "h60/h60_cur_mon_data"

FTP_USER = os.getenv("FTP_USER")
FTP_PASS = os.getenv("FTP_PASS")