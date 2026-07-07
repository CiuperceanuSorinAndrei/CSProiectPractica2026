"""Construieste nivelul curent al lacurilor din SWOT (Copernicus/NASA) intr-un JSON mic.

Descarca de pe PODAAC (autentificare NASA Earthdata) produsele SWOT recente peste Romania:
  - LakeSP (Prior): lacuri -> potrivire punct-in-poligon;
  - RiverSP (Reach): tronsoane de rau -> potrivire linie-intersecteaza-poligon (prinde lacurile
    de acumulare pe rauri mari: Portile de Fier pe Dunare, cascadele Olt/Siret etc.).
Pastreaza cota luciului apei (`wse`, m, geoid ~EGM2008) cea mai recenta per lac.

Rezultatul (data/geo/reservoirs/reservoir_levels.json) e mic si se comite; granulele brute se
cache-uiesc si sunt gitignored. Necesita, DOAR la rulare, EDL_USER / EDL_PASS in mediu.

Rulare:
    EDL_USER=... EDL_PASS=... .venv/Scripts/python.exe scripts/build_reservoir_levels.py [--days 21]
"""
from __future__ import annotations

import os
import sys
import json
import glob
import time
import tempfile
import zipfile
import argparse
import subprocess
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EDL_USER, EDL_PASS

import numpy as np
import shapefile
from shapely.geometry import Point, shape
from shapely import STRtree
from src.geo.reservoir_loader import ReservoirLoader

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
LAKE_SN = "SWOT_L2_HR_LakeSP_prior_D"
REACH_SN = "SWOT_L2_HR_RiverSP_reach_D"
RO_BBOX = (20.0, 43.5, 30.0, 48.5)
OUT_DEFAULT = "data/geo/reservoirs/reservoir_levels.json"
CACHE_DEFAULT = "data/geo/swot_cache"


def _safe(rec, idx, key):
    try:
        return float(rec[idx[key]])
    except (ValueError, TypeError, KeyError):
        return None


def cmr_granules(short_name: str, days: int) -> list[str]:
    """URL-urile .zip (deduplicate) pentru granulele SWOT peste Romania in ultimele `days` zile."""
    since = time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(time.time() - days * 86400))
    bbox = ",".join(map(str, RO_BBOX))
    urls, seen, page = [], set(), 1
    while True:
        url = (f"{CMR}?short_name={short_name}&bounding_box={bbox}"
               f"&temporal={since},&sort_key=-start_date&page_size=200&page_num={page}")
        with urllib.request.urlopen(url, timeout=60) as r:
            ents = json.load(r).get("feed", {}).get("entry", [])
        if not ents:
            break
        for g in ents:
            z = next((l["href"] for l in g.get("links", [])
                      if l.get("href", "").endswith(".zip") and "podaac" in l.get("href", "")), None)
            if z and z not in seen:
                seen.add(z); urls.append(z)
        page += 1
        if len(ents) < 200:
            break
    return urls


def download(url: str, dest: str, netrc: str, cj: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    rc = subprocess.run(["curl", "-sL", "--netrc-file", netrc, "-b", cj, "-c", cj,
                         "--max-time", "300", url, "-o", dest]).returncode
    return rc == 0 and os.path.exists(dest) and os.path.getsize(dest) > 1000


def _open_shp(zip_path: str, work: str):
    ex = os.path.join(work, os.path.basename(zip_path) + "_x")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ex)
    except zipfile.BadZipFile:
        return None
    shp = glob.glob(ex + "/**/*.shp", recursive=True)
    return shapefile.Reader(shp[0], encoding="latin-1") if shp else None


def parse_lakes(zip_path: str, work: str) -> list[dict]:
    """Observatii-lac (punct + wse) in bbox RO, filtrate pe calitate."""
    sf = _open_shp(zip_path, work)
    if sf is None:
        return []
    idx = {n: i for i, n in enumerate(f[0] for f in sf.fields[1:])}
    out = []
    for rec in sf.records():
        wse = _safe(rec, idx, "wse")
        lon, lat = _safe(rec, idx, "p_lon"), _safe(rec, idx, "p_lat")
        if wse is None or lon is None or not (-1000 < wse < 9000):
            continue
        if int(rec[idx["quality_f"]] or 0) not in (0, 1):
            continue
        if not (RO_BBOX[0] <= lon <= RO_BBOX[2] and RO_BBOX[1] <= lat <= RO_BBOX[3]):
            continue
        out.append({"geom": Point(lon, lat), "wse": wse, "wse_u": _safe(rec, idx, "wse_u"),
                    "time": str(rec[idx["time_str"]]) if "time_str" in idx else "", "product": "lake",
                    "ref": str(rec[idx["lake_id"]]) if "lake_id" in idx else ""})
    sf.close()
    return out


def parse_reaches(zip_path: str, work: str) -> list[dict]:
    """Observatii-tronson (linie + wse) in bbox RO."""
    sf = _open_shp(zip_path, work)
    if sf is None:
        return []
    idx = {n: i for i, n in enumerate(f[0] for f in sf.fields[1:])}
    out = []
    for rec, shp_ in zip(sf.records(), sf.shapes()):
        wse = _safe(rec, idx, "wse")
        lon, lat = _safe(rec, idx, "p_lon"), _safe(rec, idx, "p_lat")
        if wse is None or lon is None or not (-1000 < wse < 9000) or not shp_.points:
            continue
        if not (RO_BBOX[0] <= lon <= RO_BBOX[2] and RO_BBOX[1] <= lat <= RO_BBOX[3]):
            continue
        try:
            geom = shape(shp_.__geo_interface__)
        except Exception:
            continue
        out.append({"geom": geom, "wse": wse, "wse_u": _safe(rec, idx, "wse_u"),
                    "time": str(rec[idx["time_str"]]) if "time_str" in idx else "", "product": "river",
                    "ref": str(rec[idx["river_name"]]).strip() if "river_name" in idx else ""})
    sf.close()
    return out


def collect(short_name: str, days: int, parse_fn, netrc: str, cj: str, cache: str, work: str) -> list[dict]:
    urls = cmr_granules(short_name, days)
    print(f"{short_name}: {len(urls)} granule", flush=True)
    obs = []
    for i, url in enumerate(urls, 1):
        dest = os.path.join(cache, url.rsplit("/", 1)[-1])
        if not download(url, dest, netrc, cj):
            print(f"  [{i}/{len(urls)}] descarcare esuata", flush=True); continue
        got = parse_fn(dest, work)
        obs.extend(got)
        print(f"  [{i}/{len(urls)}] {os.path.basename(dest)[:50]} -> {len(got)}", flush=True)
    return obs


def match_points(obs: list[dict], reservoirs: dict) -> dict:
    """Fiecare lac -> observatia-punct cea mai recenta din interiorul poligonului."""
    if not obs:
        return {}
    tree = STRtree([o["geom"] for o in obs])
    res = {}
    for name, r in reservoirs.items():
        cont = [int(i) for i in tree.query(r["polygon"], predicate="contains")]
        if cont:
            res[name] = max((obs[i] for i in cont), key=lambda o: o["time"])
    return res


def match_reaches(obs: list[dict], reservoirs: dict) -> dict:
    """Fiecare lac -> mediana wse a tronsoanelor care intersecteaza poligonul, din trecerea
    cea mai recenta."""
    if not obs:
        return {}
    tree = STRtree([o["geom"] for o in obs])
    res = {}
    for name, r in reservoirs.items():
        hit = [int(i) for i in tree.query(r["polygon"], predicate="intersects")]
        if not hit:
            continue
        latest = max(obs[i]["time"] for i in hit)
        same = [obs[i] for i in hit if obs[i]["time"] == latest]
        res[name] = {"geom": None, "wse": float(np.median([o["wse"] for o in same])),
                     "wse_u": np.nanmedian([o["wse_u"] for o in same if o["wse_u"] is not None]) if any(o["wse_u"] is not None for o in same) else None,
                     "time": latest, "product": "river", "ref": ";".join(sorted({o["ref"] for o in same if o["ref"]}))[:60]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    args = ap.parse_args()

    user, pw = EDL_USER, EDL_PASS
    if not user or not pw:
        sys.exit("Lipsesc credentialele Earthdata (EDL_USER / EDL_PASS in .env).")

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    secret = tempfile.mkdtemp()
    netrc, cj = os.path.join(secret, ".netrc"), os.path.join(secret, "cj")
    with open(netrc, "w") as fh:
        fh.write(f"machine urs.earthdata.nasa.gov login {user} password {pw}\n")

    try:
        lake_obs = collect(LAKE_SN, args.days, parse_lakes, netrc, cj, args.cache, secret)
        reach_obs = collect(REACH_SN, args.days, parse_reaches, netrc, cj, args.cache, secret)
    finally:
        for p in (netrc, cj):
            if os.path.exists(p):
                os.remove(p)

    reservoirs = ReservoirLoader.get_all_reservoirs()
    print(f"observatii: {len(lake_obs)} lac + {len(reach_obs)} rau; potrivire...", flush=True)
    lake_match = match_points(lake_obs, reservoirs)
    reach_match = match_reaches(reach_obs, reservoirs)

    levels = {}
    for name in set(lake_match) | set(reach_match):
        cand = [c for c in (lake_match.get(name), reach_match.get(name)) if c]
        o = max(cand, key=lambda c: c["time"])          # cea mai recenta observatie
        levels[name] = {
            "wse_m": round(o["wse"], 2), "wse_u_m": round(o["wse_u"], 2) if o["wse_u"] is not None else None,
            "as_of": o["time"][:19], "product": o["product"], "ref": o["ref"], "source": "swot",
        }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(levels, fh, ensure_ascii=False, indent=0)
    n_lake = sum(1 for v in levels.values() if v["product"] == "lake")
    n_river = sum(1 for v in levels.values() if v["product"] == "river")
    print(f"gata: {len(levels)} lacuri cu nivel SWOT ({n_lake} lac + {n_river} rau) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
