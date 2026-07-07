"""Nivelul curent al lacurilor din Sentinel-2 (extindere optica a apei) via Copernicus Sentinel Hub.

Pentru lacurile pe care SWOT nu le poate masura (acumulari montane inguste, cu versanti abrupti),
extrage aria luciului apei (masca NDWI) de la Sentinel Hub si o inverseaza prin curba stage-storage
(DEM) intr-o cota a apei. Optica nu sufera de layover radar, deci merge in vaile abrupte.

Scrie reservoir_levels_s2.json (imbinat cu nivelele SWOT de catre ReservoirLoader). Tinta implicita:
lacurile neacoperite de SWOT cu volum >= --min-vol milioane m^3.

Necesita, DOAR la rulare, un client OAuth Sentinel Hub in mediu: SH_ID / SH_SECRET.

Rulare:
    SH_ID=... SH_SECRET=... .venv/Scripts/python.exe scripts/build_reservoir_levels_s2.py [--min-vol 20]
"""
from __future__ import annotations

import os
import sys
import io
import json
import time
import argparse
import urllib.request
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SH_ID, SH_SECRET

from src.geo import sentinel2_level as s2
from src.geo.reservoir_loader import ReservoirLoader

LEVELS_SWOT = "data/geo/reservoirs/reservoir_levels.json"
OUT_DEFAULT = "data/geo/reservoirs/reservoir_levels_s2.json"


def _measure_latest(token, poly, curve, days):
    """Cea mai recenta observatie: fereastra de `days` zile care se termina azi."""
    dfrom = time.strftime("%Y-%m-%d", time.gmtime(time.time() - days * 86400))
    dto = time.strftime("%Y-%m-%d", time.gmtime())
    return s2.measure_level(token, poly, curve, dfrom, dto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-vol", type=float, default=20.0, help="prag volum (mil m^3) pentru lacurile tinta")
    ap.add_argument("--only", default="", help="doar aceste nume (separate prin virgula)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    cid, sec = SH_ID, SH_SECRET
    if not cid or not sec:
        sys.exit("Lipsesc credentialele Sentinel Hub (SH_ID / SH_SECRET in .env).")

    reservoirs = ReservoirLoader.get_all_reservoirs()
    swot = set(json.load(open(LEVELS_SWOT, encoding="utf-8"))) if os.path.exists(LEVELS_SWOT) else set()
    if args.only:
        targets = [n.strip() for n in args.only.split(",") if n.strip()]
    else:
        targets = [n for n, r in reservoirs.items()
                   if n not in swot and r["vol_mil_m3"] >= args.min_vol]
        targets.sort(key=lambda n: -reservoirs[n]["vol_mil_m3"])
    print(f"{len(targets)} lacuri tinta (neacoperite de SWOT, vol >= {args.min_vol} mil m3)", flush=True)

    out = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    token = s2.get_token(cid, sec)
    for i, name in enumerate(targets, 1):
        r = reservoirs.get(name)
        if r is None or name in out:
            continue
        try:
            area, vfrac, wse = _measure_latest(token, r["polygon"], r["stage_storage"], 120)
            if vfrac < 0.5:      # prea mult nor peste lac -> incearca o fereastra mai larga
                area, vfrac, wse = _measure_latest(token, r["polygon"], r["stage_storage"], 270)
        except urllib.error.HTTPError as e:
            if e.code == 401:    # token expirat -> reautentificare
                token = s2.get_token(cid, sec)
                area, vfrac, wse = _measure_latest(token, r["polygon"], r["stage_storage"], 120)
            else:
                print(f"  [{i}/{len(targets)}] {name}: HTTP {e.code} {e.read().decode()[:120]}", flush=True); continue
        except Exception as e:
            print(f"  [{i}/{len(targets)}] {name}: EROARE {e}", flush=True); continue

        if vfrac < 0.3:
            print(f"  [{i}/{len(targets)}] {name}: prea innorat (valid {vfrac:.0%}), sarit", flush=True); continue
        if area < 0.05 * (r["surface_area_m2"] or 0.0):
            # arie < 5% din NNR: fie polder secat, fie NDWI ratat pe apa vegetata/turbida a campiei
            # -> nu e o citire de nivel de incredere, o lasam neacoperita (calitate > cantitate).
            print(f"  [{i}/{len(targets)}] {name}: arie apa implauzibil de mica ({area/1e6:.3f} km2), sarit", flush=True)
            continue
        out[name] = {"wse_m": round(wse, 2), "as_of": time.strftime("%Y-%m-%d"), "product": "s2",
                     "s2_area_km2": round(area / 1e6, 4), "valid_frac": round(vfrac, 2), "source": "s2"}
        print(f"  [{i}/{len(targets)}] {name:24.24s} area={area/1e6:6.3f} km2 valid={vfrac:.0%} -> wse {wse:.1f} m", flush=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)

    print(f"gata: {len(out)} lacuri cu nivel S2 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
