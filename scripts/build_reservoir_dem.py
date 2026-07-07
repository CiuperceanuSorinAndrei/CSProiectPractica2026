"""Precomputa datele DEM per lac de acumulare intr-un JSON mic, comis in repo.

Pentru fiecare lac (>= prag de suprafata) descarca tile-uri Copernicus GLO-30, construieste
curba stage-storage (integrand terenul peste luciul apei) si delimiteaza bazinul hidrografic.
Lacurile mici sau fara acoperire DEM primesc o curba parametrica si bazin nedefinit (la rulare
se cade pe suprafata lacului). Scriptul este reluabil: sare peste lacurile deja prezente in OUT.

Rulare:
    .venv\\Scripts\\python.exe scripts\\build_reservoir_dem.py [--limit N] [--only "Nume,Nume"]

Tile-urile brute (~2 GB) se cache-uiesc in --cache (implicit data/geo/dem_cache, gitignored);
in repo se comite doar JSON-ul rezultat (~cateva sute KB).
"""
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

OUT_DEFAULT = "data/geo/reservoirs/dem_augment.json"
CACHE_DEFAULT = "data/geo/dem_cache"
AREA_MIN_KM2 = 0.1                 # sub acest prag -> curba parametrica, fara bazin DEM
VOL_MARGIN = 0.05                  # ~5.5 km in jurul lacului pentru banda de inundare
CATCH_MARGINS = (0.35, 0.7, 1.2)   # extindem fereastra bazinului daca atinge marginea
CATCH_DOWNSAMPLE = 9               # ~270 m: aria bazinului e ~invarianta la rezolutie, dar ~12x mai rapid
MAX_RISE_M = 15.0


def build_one(name: str, r: dict, src: DemSource) -> dict:
    poly = r["polygon"]
    lon_min, lat_min, lon_max, lat_max = poly.bounds
    res_area_km2 = r["surface_area_m2"] / 1e6

    # --- curba stage-storage ---
    curve = None
    if res_area_km2 >= AREA_MIN_KM2:
        wv = src.mosaic(lon_min - VOL_MARGIN, lon_max + VOL_MARGIN,
                        lat_min - VOL_MARGIN, lat_max + VOL_MARGIN)
        if wv is not None:
            curve = StageStorageCurve.from_dem(wv, poly, r["max_volume_m3"], max_rise_m=MAX_RISE_M)
    if curve is None:
        wl = r.get("waterline_attr_m") or 0.0
        curve = StageStorageCurve.from_attributes(
            r["max_volume_m3"], r["surface_area_m2"], wl if wl > 0 else 0.0)

    # --- bazin hidrografic (doar pentru lacuri semnificative) ---
    catchment_km2 = None
    catchment_flag = "skipped_small"
    if res_area_km2 >= AREA_MIN_KM2:
        catchment_flag = "no_dem"
        for mg in CATCH_MARGINS:
            wc = src.mosaic(lon_min - mg, lon_max + mg, lat_min - mg, lat_max + mg)
            if wc is None:
                break
            res = delineate_catchment(wc, poly, downsample=CATCH_DOWNSAMPLE)
            catchment_km2 = round(res["catchment_km2"], 3)
            if not res["edge_clipped"]:
                catchment_flag = "ok"
                break
            catchment_flag = "clipped"   # atinge marginea; extindem sau (la ultima) ramane subestimat

    return {
        "stage_storage": curve.to_dict(),
        "catchment_km2": catchment_km2,
        "catchment_flag": catchment_flag,
        "source": curve.source,
        "res_area_km2": round(res_area_km2, 4),
        "vol_mil_m3": r["vol_mil_m3"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="proceseaza doar primele N lacuri noi")
    ap.add_argument("--only", default="", help="doar aceste nume (separate prin virgula)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    src = DemSource(args.cache)
    reservoirs = ReservoirLoader.get_all_reservoirs()

    done = {}
    if os.path.exists(args.out):
        done = json.load(open(args.out, encoding="utf-8"))
        print(f"reluare: {len(done)} lacuri deja procesate")

    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
    else:
        # ordonam descrescator dupa volum: cele importante intai
        names = sorted(reservoirs, key=lambda n: -reservoirs[n]["vol_mil_m3"])
    todo = [n for n in names if n not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"de procesat: {len(todo)} lacuri")

    t0 = time.time()
    for i, name in enumerate(todo, 1):
        t = time.time()
        try:
            done[name] = build_one(name, reservoirs[name], src)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {name}: EROARE {e}")
            continue
        d = done[name]
        print(f"  [{i}/{len(todo)}] {name:28.28s} src={d['source']:10s} "
              f"catch={d['catchment_km2']} ({d['catchment_flag']}) {time.time()-t:4.1f}s")
        if i % 20 == 0:
            json.dump(done, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"    ...salvat ({len(done)} total, {time.time()-t0:.0f}s scurs)")

    json.dump(done, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"gata: {len(done)} lacuri in {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
