from __future__ import annotations
import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SH_ID, SH_SECRET
from src.geo import sentinel2_level as s2
from src.geo.reservoir_loader import ReservoirLoader

# Configuration
LEVELS_SWOT = "data/geo/reservoirs/reservoir_levels.json"
OUT_DEFAULT = "data/geo/reservoirs/reservoir_levels_s2.json"

def _measure_latest(token, poly, curve, days):
    dfrom = time.strftime("%Y-%m-%d", time.gmtime(time.time() - days * 86400))
    dto = time.strftime("%Y-%m-%d", time.gmtime())
    return s2.measure_level(token, poly, curve, dfrom, dto)

def main():
    # 1. Setup Arguments & Auth
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-vol", type=float, default=20.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    cid, sec = SH_ID, SH_SECRET
    if not cid or not sec:
        sys.exit("Missing Sentinel Hub credentials.")

    # 2. Filter targets
    reservoirs = ReservoirLoader.get_all_reservoirs()
    swot = set()
    if os.path.exists(LEVELS_SWOT):
        with open(LEVELS_SWOT, encoding="utf-8") as f: swot = set(json.load(f))
            
    if args.only:
        targets = [n.strip() for n in args.only.split(",") if n.strip()]
    else:
        targets = [n for n, r in reservoirs.items() if n not in swot and r["vol_mil_m3"] >= args.min_vol]
        targets.sort(key=lambda n: -reservoirs[n]["vol_mil_m3"])
        
    print(f"{len(targets)} target reservoirs", flush=True)

    # 3. Process loop
    out = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f: out = json.load(f)
            
    token = s2.get_token(cid, sec)
    
    for i, name in enumerate(targets, 1):
        r = reservoirs.get(name)
        if r is None or name in out: continue
            
        try:
            area, vfrac, wse = _measure_latest(token, r["polygon"], r["stage_storage"], 120)
            if vfrac < 0.5:
                area, vfrac, wse = _measure_latest(token, r["polygon"], r["stage_storage"], 270)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                token = s2.get_token(cid, sec)
                area, vfrac, wse = _measure_latest(token, r["polygon"], r["stage_storage"], 120)
            else:
                print(f"  [{i}/{len(targets)}] {name}: HTTP {e.code} {e.read().decode()[:120]}", flush=True)
                continue
        except Exception as e:
            print(f"  [{i}/{len(targets)}] {name}: ERROR {e}", flush=True)
            continue

        if vfrac < 0.3:
            print(f"  [{i}/{len(targets)}] {name}: skipped (valid {vfrac:.0%})", flush=True)
            continue
            
        if area < 0.05 * (r["surface_area_m2"] or 0.0):
            print(f"  [{i}/{len(targets)}] {name}: skipped (water {area/1e6:.3f} km2)", flush=True)
            continue
            
        out[name] = {
            "wse_m": round(wse, 2), "as_of": time.strftime("%Y-%m-%d"), 
            "product": "s2", "s2_area_km2": round(area / 1e6, 4), 
            "valid_frac": round(vfrac, 2), "source": "s2"
        }
        print(f"  [{i}/{len(targets)}] {name:24.24s} area={area/1e6:6.3f} km2 valid={vfrac:.0%} -> wse {wse:.1f} m", flush=True)
        
        with open(args.out, "w", encoding="utf-8") as f: json.dump(out, f, ensure_ascii=False, indent=0)

    print(f"Done: {len(out)} reservoirs -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
