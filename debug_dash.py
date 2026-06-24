"""Harness de debug: apeleaza callback-ul principal direct, fara a porni serverul."""
import sys
import os

# Adaugam directorul curent ca importurile sa functioneze
sys.path.append(os.path.abspath("."))

from config import PREDEFINED_LOCATIONS
from src.dashboard import DEFAULT_TIME_RANGE
from app_dash import dashboard

# _update_dashboard(frame_idx, loc, m_lat, m_lon, map_zoom, radius_km, run_mode, time_range)
loc = list(PREDEFINED_LOCATIONS.keys())[0]

print("Starting debug test...")
try:
    res = dashboard._update_dashboard(0, loc, 44.33, 23.79, 500, 30, "historic", DEFAULT_TIME_RANGE)
    print("Success! Returned", len(res), "items")
    print("Image src starts with:", res[0][:50])
except Exception:
    import traceback
    traceback.print_exc()
