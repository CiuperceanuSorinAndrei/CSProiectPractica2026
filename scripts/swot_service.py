from __future__ import annotations

import os
import json
import glob
import time
import subprocess
import urllib.request
import zipfile
import numpy as np
import shapefile
from shapely.geometry import Point, shape
from shapely import STRtree

# Constants
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
RO_BBOX = (20.0, 43.5, 30.0, 48.5)


def _safe_float(rec, idx_map, key):
    # Extract float safely
    try:
        return float(rec[idx_map[key]])
    except (ValueError, TypeError, KeyError):
        return None


def fetch_cmr_granules(short_name: str, days: int) -> list[str]:
    # Fetch deduplicated .zip URLs from CMR
    since = time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(time.time() - days * 86400))
    bbox = ",".join(map(str, RO_BBOX))
    urls, seen, page = [], set(), 1
    
    while True:
        url = (f"{CMR_URL}?short_name={short_name}&bounding_box={bbox}"
               f"&temporal={since},&sort_key=-start_date&page_size=200&page_num={page}")
        
        with urllib.request.urlopen(url, timeout=60) as r:
            ents = json.load(r).get("feed", {}).get("entry", [])
            
        if not ents: break
            
        for g in ents:
            z = next((l["href"] for l in g.get("links", [])
                      if l.get("href", "").endswith(".zip") and "podaac" in l.get("href", "")), None)
            if z and z not in seen:
                seen.add(z)
                urls.append(z)
                
        page += 1
        if len(ents) < 200: break
            
    return urls


def download_curl(url: str, dest: str, netrc: str, cj: str) -> bool:
    # Curl download with auth
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
        
    res = subprocess.run(["curl", "-sL", "--netrc-file", netrc, "-b", cj, "-c", cj,
                         "--max-time", "300", url, "-o", dest], capture_output=True, text=True)
                         
    if res.returncode != 0 or not os.path.exists(dest) or os.path.getsize(dest) < 1000:
        print(f"curl failed. RC: {res.returncode}, STDERR: {res.stderr}")
        return False
        
    return True


def _read_shp_from_zip(zip_path: str, work_dir: str):
    # Extract and open shapefile
    ex = os.path.join(work_dir, os.path.basename(zip_path) + "_x")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ex)
    except zipfile.BadZipFile:
        return None
        
    shp = glob.glob(ex + "/**/*.shp", recursive=True)
    return shapefile.Reader(shp[0], encoding="latin-1") if shp else None


def parse_lakes(zip_path: str, work_dir: str) -> list[dict]:
    # Parse lake points
    sf = _read_shp_from_zip(zip_path, work_dir)
    if not sf: return []
        
    idx = {n: i for i, n in enumerate(f[0] for f in sf.fields[1:])}
    out = []
    
    for rec in sf.records():
        wse = _safe_float(rec, idx, "wse")
        lon, lat = _safe_float(rec, idx, "p_lon"), _safe_float(rec, idx, "p_lat")
        
        if wse is None or lon is None or not (-1000 < wse < 9000): continue
        if int(rec[idx["quality_f"]] or 0) not in (0, 1): continue
        if not (RO_BBOX[0] <= lon <= RO_BBOX[2] and RO_BBOX[1] <= lat <= RO_BBOX[3]): continue
            
        out.append({
            "geom": Point(lon, lat), "wse": wse, "wse_u": _safe_float(rec, idx, "wse_u"),
            "time": str(rec[idx["time_str"]]) if "time_str" in idx else "", 
            "product": "lake",
            "ref": str(rec[idx["lake_id"]]) if "lake_id" in idx else ""
        })
        
    sf.close()
    return out


def parse_reaches(zip_path: str, work_dir: str) -> list[dict]:
    # Parse reach lines
    sf = _read_shp_from_zip(zip_path, work_dir)
    if not sf: return []
        
    idx = {n: i for i, n in enumerate(f[0] for f in sf.fields[1:])}
    out = []
    
    for rec, shp_ in zip(sf.records(), sf.shapes()):
        wse = _safe_float(rec, idx, "wse")
        lon, lat = _safe_float(rec, idx, "p_lon"), _safe_float(rec, idx, "p_lat")
        
        if wse is None or lon is None or not (-1000 < wse < 9000) or not shp_.points: continue
        if not (RO_BBOX[0] <= lon <= RO_BBOX[2] and RO_BBOX[1] <= lat <= RO_BBOX[3]): continue
            
        try:
            geom = shape(shp_.__geo_interface__)
        except Exception:
            continue
            
        out.append({
            "geom": geom, "wse": wse, "wse_u": _safe_float(rec, idx, "wse_u"),
            "time": str(rec[idx["time_str"]]) if "time_str" in idx else "", 
            "product": "river",
            "ref": str(rec[idx["river_name"]]).strip() if "river_name" in idx else ""
        })
        
    sf.close()
    return out


def collect_granules(short_name: str, days: int, parse_fn, netrc: str, cj: str, cache: str, work: str) -> list[dict]:
    # Collect and parse all granules
    urls = fetch_cmr_granules(short_name, days)
    print(f"{short_name}: {len(urls)} granules", flush=True)
    obs = []
    
    for i, url in enumerate(urls, 1):
        dest = os.path.join(cache, url.rsplit("/", 1)[-1])
        if not download_curl(url, dest, netrc, cj):
            print(f"  [{i}/{len(urls)}] download failed", flush=True)
            continue
            
        got = parse_fn(dest, work)
        obs.extend(got)
        print(f"  [{i}/{len(urls)}] {os.path.basename(dest)[:50]} -> {len(got)}", flush=True)
        
    return obs


def match_points_to_polygons(obs: list[dict], reservoirs: dict) -> dict:
    # Match most recent points
    if not obs: return {}
        
    tree = STRtree([o["geom"] for o in obs])
    res = {}
    
    for name, r in reservoirs.items():
        cont = [int(i) for i in tree.query(r["polygon"], predicate="contains")]
        if cont:
            res[name] = max((obs[i] for i in cont), key=lambda o: o["time"])
            
    return res


def match_reaches_to_polygons(obs: list[dict], reservoirs: dict) -> dict:
    # Match median reaches
    if not obs: return {}
        
    tree = STRtree([o["geom"] for o in obs])
    res = {}
    
    for name, r in reservoirs.items():
        hit = [int(i) for i in tree.query(r["polygon"], predicate="intersects")]
        if not hit: continue
            
        latest = max(obs[i]["time"] for i in hit)
        same = [obs[i] for i in hit if obs[i]["time"] == latest]
        
        valid_wse_u = [o["wse_u"] for o in same if o["wse_u"] is not None]
        median_wse_u = float(np.nanmedian(valid_wse_u)) if valid_wse_u else None
        
        res[name] = {
            "geom": None, 
            "wse": float(np.median([o["wse"] for o in same])),
            "wse_u": median_wse_u,
            "time": latest, 
            "product": "river", 
            "ref": ";".join(sorted({o["ref"] for o in same if o["ref"]}))[:60]
        }
        
    return res
