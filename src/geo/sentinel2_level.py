# Fetches Sentinel-2 NDWI lake area via Sentinel Hub and inverts it to WSE using stage-storage curve.
from __future__ import annotations

import io
import json
import urllib.request
import urllib.parse

import numpy as np
import tifffile
import shapely
from shapely.ops import transform as shp_transform
from pyproj import Transformer

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
MAX_DIM = 2500  # Sentinel Hub limit for output dimension

_EVALSCRIPT = """//VERSION=3
function setup(){return{input:["B03","B08","SCL","dataMask"],output:{bands:2,sampleType:"UINT8"}}}
function evaluatePixel(s){
  let valid = (s.dataMask==1 && ![8,9,10].includes(s.SCL)) ? 1 : 0;   // exclude clouds and nodata
  let water = 0;
  if(valid==1){ let ndwi=(s.B03-s.B08)/(s.B03+s.B08); water = ndwi>0.0 ? 1 : 0; }
  return [water, valid];
}"""


def get_token(cid: str, sec: str) -> str:
    data = urllib.parse.urlencode({"grant_type": "client_credentials",
                                   "client_id": cid, "client_secret": sec}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=data), timeout=30))["access_token"]


def utm_epsg(lon: float) -> int:
    return 32634 if lon < 24.0 else 32635   # Romania: UTM zones 34N / 35N


def _process(token: str, bbox, epsg: int, W: int, H: int, date_from: str, date_to: str,
             timeout: float = 180) -> np.ndarray:
    body = {
        "input": {"bounds": {"bbox": list(bbox),
                             "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
                  "data": [{"type": "sentinel-2-l2a",
                            "dataFilter": {"timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"},
                                           "maxCloudCoverage": 60, "mosaickingOrder": "leastCC"}}]},
        "output": {"width": W, "height": H, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": _EVALSCRIPT,
    }
    req = urllib.request.Request(PROCESS_URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                                          "Accept": "image/tiff"})
    arr = tifffile.imread(io.BytesIO(urllib.request.urlopen(req, timeout=timeout).read()))
    return arr if arr.ndim == 3 else arr[..., None]


def measure_level(token: str, poly, curve, date_from: str, date_to: str, timeout: float = 180):
    # Returns (area_m2, valid_fraction, absolute_wse) for given time window
    epsg = utm_epsg(poly.centroid.x)
    poly_u = shp_transform(Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform, poly)
    xmin, ymin, xmax, ymax = poly_u.bounds
    xmin -= 300; ymin -= 300; xmax += 300; ymax += 300
    bw, bh = xmax - xmin, ymax - ymin
    W, H = int(bw / 10), int(bh / 10)
    scale = min(1.0, MAX_DIM / max(W, H))
    W, H = max(int(W * scale), 1), max(int(H * scale), 1)
    px, py = bw / W, bh / H

    arr = _process(token, (xmin, ymin, xmax, ymax), epsg, W, H, date_from, date_to, timeout=timeout)
    water, valid = arr[..., 0], arr[..., 1]

    xs = xmin + (np.arange(W) + 0.5) * px
    ys = ymax - (np.arange(H) + 0.5) * py
    X, Y = np.meshgrid(xs, ys)
    inpoly = shapely.contains_xy(poly_u, X.ravel(), Y.ravel()).reshape(water.shape)
    cell = px * py
    valid_frac = float((valid[inpoly] == 1).mean()) if inpoly.any() else 0.0
    area_m2 = float(((water == 1) & inpoly).sum()) * cell

    lv, vol = curve.levels_m, curve.volumes_m3
    area_at_level = np.maximum.accumulate(np.gradient(vol, lv))     # area(level), monotonic
    wse = curve.waterline_m + float(np.interp(area_m2, area_at_level, lv))
    return area_m2, valid_frac, wse


def best_scene_date(token: str, poly, date_from: str, date_to: str) -> str | None:
    # Returns date (YYYY-MM-DD) of least cloudy Sentinel-2 scene in window
    b = poly.bounds
    body = {"collections": ["sentinel-2-l2a"], "datetime": f"{date_from}T00:00:00Z/{date_to}T23:59:59Z",
            "bbox": [b[0], b[1], b[2], b[3]], "limit": 100}
    req = urllib.request.Request(CATALOG_URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        feats = json.load(urllib.request.urlopen(req, timeout=30)).get("features", [])
    except Exception:
        return None
    best = None
    for f in feats:
        p = f.get("properties", {})
        cc = p.get("eo:cloud_cover", 100)
        dt = p.get("datetime", "")[:10]
        if dt and (best is None or cc < best[0]):
            best = (cc, dt)
    return best[1] if best else None
