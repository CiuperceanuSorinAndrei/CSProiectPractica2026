from __future__ import annotations
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geo.reservoir_loader import ReservoirLoader
from src.geo.dem_source import DemSource
from src.geo.stage_storage import StageStorageCurve
from src.geo.catchment import delineate_catchment

# Configuration
OUT_DEFAULT = "data/geo/reservoirs/dem_augment.json"
CACHE_DEFAULT = "data/geo/dem_cache"
AREA_MIN_KM2 = 0.1
VOL_MARGIN = 0.05
CATCH_MARGINS = (0.35, 0.7, 1.2)
CATCH_DOWNSAMPLE = 9
MAX_RISE_M = 15.0

def build_one(name: str, r: dict, src: DemSource) -> dict:
    # 1. Stage-Storage Curve Generation
    poly = r["polygon"]
    lon_min, lat_min, lon_max, lat_max = poly.bounds
    res_area_km2 = r["surface_area_m2"] / 1e6
    curve = None

    if res_area_km2 >= AREA_MIN_KM2:
        wv = src.mosaic(lon_min - VOL_MARGIN, lon_max + VOL_MARGIN, lat_min - VOL_MARGIN, lat_max + VOL_MARGIN)
        if wv is not None:
            curve = StageStorageCurve.from_dem(wv, poly, r["max_volume_m3"], max_rise_m=MAX_RISE_M)
    
    # 2. Fallback Curve
    if curve is None:
        wl = r.get("waterline_attr_m") or 0.0
        curve = StageStorageCurve.from_attributes(r["max_volume_m3"], r["surface_area_m2"], wl if wl > 0 else 0.0)

    # 3. Hydrographic Catchment
    catchment_km2 = None
    catchment_wkt = None
    catchment_flag = "skipped_small"
    
    if res_area_km2 >= AREA_MIN_KM2:
        catchment_flag = "no_dem"
        for mg in CATCH_MARGINS:
            wc = src.mosaic(lon_min - mg, lon_max + mg, lat_min - mg, lat_max + mg)
            if wc is None: break
            
            res = delineate_catchment(wc, poly, downsample=CATCH_DOWNSAMPLE)
            catchment_km2 = round(res["catchment_km2"], 3)
            catchment_wkt = res.get("catchment_wkt")
            
            if not res["edge_clipped"]:
                catchment_flag = "ok"
                break
            catchment_flag = "clipped"

    return {
        "stage_storage": curve.to_dict(),
        "catchment_km2": catchment_km2,
        "catchment_flag": catchment_flag,
        "catchment_wkt": catchment_wkt,
        "source": curve.source,
        "res_area_km2": round(res_area_km2, 4),
        "vol_mil_m3": r["vol_mil_m3"],
    }

def main():
    # 4. Initialization
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    src = DemSource(args.cache)
    reservoirs = ReservoirLoader.get_all_reservoirs()

    # 5. Load Cache
    done = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f: done = json.load(f)
        print(f"Resuming: {len(done)} processed")

    names = [n.strip() for n in args.only.split(",") if n.strip()] if args.only else sorted(reservoirs, key=lambda n: -reservoirs[n]["vol_mil_m3"])
    todo = [n for n in names if n not in done]
    if args.limit: todo = todo[:args.limit]
    
    print(f"To process: {len(todo)}")

    # 6. Processing Loop
    t0 = time.time()
    for i, name in enumerate(todo, 1):
        t = time.time()
        try:
            done[name] = build_one(name, reservoirs[name], src)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {name}: ERROR {e}")
            continue
            
        d = done[name]
        print(f"  [{i}/{len(todo)}] {name:28.28s} src={d['source']:10s} catch={d['catchment_km2']} ({d['catchment_flag']}) {time.time()-t:4.1f}s")
              
        if i % 20 == 0:
            with open(args.out, "w", encoding="utf-8") as f: json.dump(done, f, ensure_ascii=False)
            print(f"    ...saved ({len(done)} total, {time.time()-t0:.0f}s elapsed)")

    # 7. Final Save
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False)
    print(f"Done: {len(done)} reservoirs in {args.out} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
