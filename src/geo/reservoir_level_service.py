"""Serviciu care furnizeaza cota curenta a unui lac pentru intervalul de simulare selectat.

Cand utilizatorul alege un alt interval de timp, in loc de snapshot-ul static (cea mai recenta
observatie), interogheaza Sentinel-2 (Sentinel Hub) pentru observatia cea mai apropiata de data
intervalului, o inverseaza prin curba DEM si suprascrie nivelul. Rezultatul e memoizat per
(lac, data). Fara credentiale sau la esec/nori, se cade pe nivelul static.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import config
from src.geo import sentinel2_level as s2

_cache: dict[tuple, dict | None] = {}
_token: str | None = None
_token_ts: float = 0.0

_TIMEOUT = 90.0            # secunde per cerere Sentinel Hub (nu bloca dashboard-ul)
_WINDOWS = ((30, 10), (60, 30))   # (zile inainte, dupa) tinta; se largeste daca e innorat


def _get_token() -> str:
    global _token, _token_ts
    if _token is None or (time.time() - _token_ts) > 480:   # reinnoire la ~8 min
        _token = s2.get_token(config.SH_ID, config.SH_SECRET)
        _token_ts = time.time()
    return _token


def _fetch(reservoir: dict, target: str) -> dict | None:
    curve = reservoir.get("stage_storage")
    v_nnr = reservoir.get("max_volume_m3") or 0.0
    if curve is None or v_nnr <= 0.0:
        return None
    poly = reservoir["polygon"]
    t = datetime.strptime(target, "%Y-%m-%d")

    area = vfrac = wse = None
    dfrom = dto = None
    for before, after in _WINDOWS:
        dfrom = (t - timedelta(days=before)).strftime("%Y-%m-%d")
        dto = (t + timedelta(days=after)).strftime("%Y-%m-%d")
        try:
            area, vfrac, wse = s2.measure_level(_get_token(), poly, curve, dfrom, dto, timeout=_TIMEOUT)
        except Exception:
            return None
        if vfrac >= 0.5:
            break
    if wse is None or vfrac < 0.3 or area < 0.05 * (reservoir.get("surface_area_m2") or 0.0):
        return None                     # innorat / arie implauzibila / fara scena -> fara citire

    try:
        as_of = s2.best_scene_date(_get_token(), poly, dfrom, dto)
    except Exception:
        as_of = None
    v0 = max(curve.volume_for_wse(wse), 0.0)
    return {"current_wse_m": round(wse, 2), "current_volume_m3": v0, "current_fill_frac": v0 / v_nnr,
            "level_source": "s2", "level_product": "s2", "level_as_of": as_of or target}


def with_interval_level(reservoir: dict | None, name: str, time_range: dict | None) -> dict | None:
    """Copie a `reservoir` cu nivelul din intervalul selectat (Sentinel-2 la cerere), memoizat per
    (lac, data). Fallback la reservoir-ul original (nivel static) daca lipsesc credentialele, nu e
    selectat un interval, sau observatia nu e de incredere. Non-blocant pe erori."""
    if not reservoir or not name:
        return reservoir
    if not config.SH_ID or not config.SH_SECRET:
        return reservoir
    start = (time_range or {}).get("start")
    if not start:
        return reservoir
    target = start[:10]

    key = (name, target)
    if key not in _cache:
        _cache[key] = _fetch(reservoir, target)
    rec = _cache[key]
    if not rec:
        return reservoir
    merged = dict(reservoir)
    merged.update(rec)
    return merged
