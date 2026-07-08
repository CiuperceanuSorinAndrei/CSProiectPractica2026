# Service providing the current lake level for the selected simulation interval.
# Queries Sentinel-2 for closest observation, calculates level via DEM curve, and memoizes.
from __future__ import annotations

import time
from datetime import datetime, timedelta

from src import config
from src.geo import sentinel2_level as s2
from src.geo.meteo_service import MeteoService
from src.geo.reservoir_fill import ReservoirFillEstimator

_cache: dict[tuple, dict | None] = {}
_token: str | None = None
_token_ts: float = 0.0

_TIMEOUT = 90.0            # seconds per Sentinel Hub request
_WINDOWS = ((30, 0), (60, 0), (90, 0))   # DO NOT look into the future! (days before, days after)


def _get_token() -> str:
    global _token, _token_ts
    if _token is None or (time.time() - _token_ts) > 480:   # renew after ~8 min
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
        return None                     # cloudy / implausible area / no scene -> skip

    try:
        as_of = s2.best_scene_date(_get_token(), poly, dfrom, dto)
    except Exception:
        as_of = None
    v0 = max(curve.volume_for_wse(wse), 0.0)
    return {"current_wse_m": round(wse, 2), "current_volume_m3": v0, "current_fill_frac": v0 / v_nnr,
            "level_source": "s2", "level_product": "s2", "level_as_of": as_of or target}


def with_interval_level(reservoir: dict | None, name: str, time_range: dict | None) -> dict | None:
    # Returns a copy of reservoir with level for selected interval (memoized Sentinel-2 fetch)
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
    
    merged = dict(reservoir)
    if rec:
        merged.update(rec)
        
    # Fast-forward volume if there's a gap between level_as_of and target
    as_of_str = merged.get("level_as_of")
    if as_of_str:
        try:
            # Parse dates
            as_of_dt = datetime.fromisoformat(as_of_str.replace("Z", "+00:00")).replace(tzinfo=None)
            target_dt = datetime.strptime(target, "%Y-%m-%d")
            
            # If the measurement is strictly before the target date, fast forward
            if as_of_dt.date() < target_dt.date():
                gap_data = MeteoService.fetch_historical_gap(
                    lat=merged["center"][0], lon=merged["center"][1],
                    start_dt=as_of_dt, end_dt=target_dt
                )
                precip_mm = gap_data["precipitation_mm"]
                evap_mm = gap_data["evaporation_mm"]
                
                # Estimate new volume using ReservoirFillEstimator
                days = (target_dt.date() - as_of_dt.date()).days
                # Simulate over the gap (duration_hours = days * 24)
                v_start = merged.get("current_volume_m3") or merged.get("max_volume_m3", 0.0)
                catch_km2 = merged.get("catchment_km2") or 0.0
                area_m2 = merged.get("surface_area_m2") or 0.0
                
                # Inflow from precipitation
                if catch_km2 > 0:
                    inflow_m3 = config.RUNOFF_COEFFICIENT * (precip_mm * 0.001) * (catch_km2 * 1e6)
                else:
                    inflow_m3 = (precip_mm * 0.001) * area_m2
                    
                # Outflow from base flow over the gap days
                base_outflow_m3s = (catch_km2 * 0.005) if catch_km2 else 10.0
                outflow_m3 = base_outflow_m3s * (days * 24 * 3600)
                
                # Evaporation from Open-Meteo
                evap_m3 = (evap_mm * 0.001) * area_m2
                
                v_new = max(v_start + inflow_m3 - outflow_m3 - evap_m3, 0.0)
                merged["current_volume_m3"] = v_new
                
                curve = merged.get("stage_storage")
                v_nnr = merged.get("max_volume_m3", 0.0)
                if curve and v_nnr > 0:
                    merged["current_wse_m"] = round(curve.level_for_volume(v_new), 2)
                    merged["current_fill_frac"] = v_new / v_nnr
                merged["level_as_of"] = target + "T00:00:00 (Fast-forwarded)"
                merged["fast_forward_days"] = days
        except Exception as e:
            print(f"Warning: Failed to fast-forward volume: {e}")

    return merged
