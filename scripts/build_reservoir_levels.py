from __future__ import annotations
import os
import sys
import json
import tempfile
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import EDL_USER, EDL_PASS
from src.geo.reservoir_loader import ReservoirLoader

# Import from scripts/
import swot_service

# Configuration
LAKE_SN = "SWOT_L2_HR_LakeSP_prior_D"
REACH_SN = "SWOT_L2_HR_RiverSP_reach_D"
OUT_DEFAULT = "data/geo/reservoirs/reservoir_levels.json"
CACHE_DEFAULT = "data/geo/swot_cache"

def main():
    # 1. Setup arguments & auth
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    args = ap.parse_args()

    if not EDL_USER or not EDL_PASS:
        sys.exit("Missing Earthdata credentials.")

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    secret = tempfile.mkdtemp()
    netrc, cj = os.path.join(secret, ".netrc"), os.path.join(secret, "cj")
    
    with open(netrc, "w") as fh:
        fh.write(f"machine urs.earthdata.nasa.gov login {EDL_USER} password {EDL_PASS}\n")

    # 2. Collect data
    try:
        lake_obs = swot_service.collect_granules(LAKE_SN, args.days, swot_service.parse_lakes, netrc, cj, args.cache, secret)
        reach_obs = swot_service.collect_granules(REACH_SN, args.days, swot_service.parse_reaches, netrc, cj, args.cache, secret)
    finally:
        for p in (netrc, cj):
            if os.path.exists(p): os.remove(p)

    # 3. Match and extract
    reservoirs = ReservoirLoader.get_all_reservoirs()
    print(f"Observations: {len(lake_obs)} lake + {len(reach_obs)} river; matching...", flush=True)
    
    lake_m = swot_service.match_points_to_polygons(lake_obs, reservoirs)
    reach_m = swot_service.match_reaches_to_polygons(reach_obs, reservoirs)

    levels = {}
    for name in set(lake_m) | set(reach_m):
        cand = [c for c in (lake_m.get(name), reach_m.get(name)) if c]
        o = max(cand, key=lambda c: c["time"])
        levels[name] = {
            "wse_m": round(o["wse"], 2), 
            "wse_u_m": round(o["wse_u"], 2) if o["wse_u"] is not None else None,
            "as_of": o["time"][:19], 
            "product": o["product"], 
            "ref": o["ref"], 
            "source": "swot",
        }

    # 4. Save
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(levels, fh, ensure_ascii=False, indent=0)
        
    print(f"Done: {len(levels)} reservoirs saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
