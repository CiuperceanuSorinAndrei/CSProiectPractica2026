import os
import threading
import shapefile
from shapely.geometry import shape
from shapely.ops import transform
from shapely import wkt as shapely_wkt
from pyproj import Transformer

class ReservoirLoader:
    _cache = None
    _lock = threading.Lock()
    _DEM_AUGMENT_PATH = "data/geo/reservoirs/dem_augment.json"
    _LEVELS_PATH = "data/geo/reservoirs/reservoir_levels.json"          # SWOT (lake + river)
    _LEVELS_S2_PATH = "data/geo/reservoirs/reservoir_levels_s2.json"    # Sentinel-2 (optical)

    @staticmethod
    def get_all_reservoirs(shapefile_path="data/geo/reservoirs/LacuriAcumulare.shp") -> dict:
        # Parses the shapefile and extracts geometries, returning dictionary of lakes
        if ReservoirLoader._cache is not None:
            return ReservoirLoader._cache
            
        with ReservoirLoader._lock:
            if ReservoirLoader._cache is not None:
                return ReservoirLoader._cache

            if not os.path.exists(shapefile_path):
                return {}

        reservoirs = {}
        try:
            sf = shapefile.Reader(shapefile_path)
            records = sf.records()
            shapes = sf.shapes()
            
            # Project from Stereo 70 (EPSG:31700) to Lat/Lon (EPSG:4326)
            proj_transformer = Transformer.from_crs("EPSG:31700", "EPSG:4326", always_xy=True)

            for rec, shp in zip(records, shapes):
                entry = ReservoirLoader._build_reservoir_entry(rec, shp, proj_transformer, reservoirs)
                if entry is not None:
                    name, data = entry
                    reservoirs[name] = data

            sf.close()
            ReservoirLoader._augment_with_dem(reservoirs)
            ReservoirLoader._cache = reservoirs
        except Exception as e:
            print(f"Error loading reservoirs: {e}")

        return ReservoirLoader._cache if ReservoirLoader._cache else {}

    @staticmethod
    def get_covered_reservoirs(shapefile_path="data/geo/reservoirs/LacuriAcumulare.shp") -> dict:
        # Returns subset of reservoirs covered by SWOT (or all if fallback)
        from src.config import RESERVOIRS_SWOT_COVERED_ONLY
        all_res = ReservoirLoader.get_all_reservoirs(shapefile_path)
        if not RESERVOIRS_SWOT_COVERED_ONLY:
            return all_res
        covered = {n: r for n, r in all_res.items() if r.get("level_source") in ("swot", "s2")}
        return covered if covered else all_res

    @staticmethod
    def _load_json(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            import json
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            print(f"Could not read {path}: {e}")
            return {}

    @staticmethod
    def _augment_with_dem(reservoirs: dict) -> None:
        # Attaches stage-storage curve, catchment area, and current level to reservoirs
        from src.geo.stage_storage import StageStorageCurve

        aug = ReservoirLoader._load_json(ReservoirLoader._DEM_AUGMENT_PATH)
        # Current levels: SWOT (primary) + Sentinel-2 (gap fill for mountain lakes)
        # SWOT takes priority where both exist
        levels = {**ReservoirLoader._load_json(ReservoirLoader._LEVELS_S2_PATH),
                  **ReservoirLoader._load_json(ReservoirLoader._LEVELS_PATH)}

        for name, r in reservoirs.items():
            entry = aug.get(name)
            if entry:
                curve = StageStorageCurve.from_dict(entry["stage_storage"])
                r["catchment_km2"] = entry.get("catchment_km2")
                # Parse catchment boundary polygon from WKT (used for MAP calculation)
                cwkt = entry.get("catchment_wkt")
                r["catchment_polygon"] = shapely_wkt.loads(cwkt) if cwkt else None
                r["catchment_flag"] = entry.get("catchment_flag")
            else:
                wl = r.get("waterline_attr_m") or 0.0
                curve = StageStorageCurve.from_attributes(
                    r["max_volume_m3"], r["surface_area_m2"], wl if wl > 0 else 0.0)
                r["catchment_km2"] = None
                r["catchment_polygon"] = None
                r["catchment_flag"] = "not_built"

            curve = curve.with_submerged_branch(r["surface_area_m2"])
            r["stage_storage"] = curve
            ReservoirLoader._attach_current_level(r, curve, levels.get(name))

    @staticmethod
    def _attach_current_level(r: dict, curve, lvl: dict | None) -> None:
        # Sets starting volume from SWOT level or None if no observation
        v_nnr = r.get("max_volume_m3") or 0.0
        if lvl and lvl.get("wse_m") is not None and v_nnr > 0:
            wse = float(lvl["wse_m"])
            # Cap volume at NNR: max operation state is "full" (100%).
            # Submerged curve gives true fraction; above NNR is not a plausible starting state from SWOT.
            v0 = min(max(curve.volume_for_wse(wse), 0.0), v_nnr)
            r["current_volume_m3"] = v0
            r["current_fill_frac"] = v0 / v_nnr
            r["current_wse_m"] = wse
            r["level_source"] = lvl.get("source", "swot")
            r["level_product"] = lvl.get("product")     # "lake" | "river"
            r["level_as_of"] = lvl.get("as_of")
        else:
            r["current_volume_m3"] = None      # -> estimator starts from NNR
            r["current_fill_frac"] = None
            r["current_wse_m"] = None
            r["level_source"] = "assumed_nnr"
            r["level_as_of"] = None

    @staticmethod
    def _build_reservoir_entry(rec, shp, transformer, existing: dict):
        # Builds a reservoir entry from record and shape, handling reprojection and duplicates
        denumire = rec.denumire
        if not denumire or str(denumire).strip() == "":
            return None

        name = str(denumire).strip().title()

        # Avoid duplicates (if multiple polygons have the same name)
        original_name = name
        idx = 2
        while name in existing:
            name = f"{original_name} {idx}"
            idx += 1

        geom_stereo = shape(shp)
        if geom_stereo.is_empty:
            return None

        # Water surface area: polygon area in native CRS (Stereo 70, meters) -> m^2
        surface_area_m2 = float(geom_stereo.area)

        # Max volume (at Normal Retention Level), stored in shapefile in million m^3
        vol_mil_m3 = ReservoirLoader._safe_float(rec, "vol_mil_m3")
        max_volume_m3 = vol_mil_m3 * 1.0e6 if vol_mil_m3 else 0.0

        # Water surface elevation from attribute; used only as fallback when DEM is missing
        waterline_attr_m = ReservoirLoader._safe_float(rec, "elevatie")

        # Transform polygon from meters (Stereo 70) to degrees (WGS84)
        geom = transform(transformer.transform, geom_stereo)

        bounds = geom.bounds  # (min_lon, min_lat, max_lon, max_lat)

        # Calculate approximate bounding radius (width/height in km)
        delta_lon = bounds[2] - bounds[0]
        delta_lat = bounds[3] - bounds[1]
        radius_km = max(delta_lon * 80.0, delta_lat * 111.0) / 2.0 + 5.0  # +5km padding

        data = {
            "name": name,
            "polygon": geom,
            "bounds": bounds,
            "center": (geom.centroid.y, geom.centroid.x),  # (lat, lon)
            "radius_km": radius_km,
            "surface_area_m2": surface_area_m2,
            "vol_mil_m3": vol_mil_m3,
            "max_volume_m3": max_volume_m3,
            "waterline_attr_m": waterline_attr_m,
        }
        return name, data

    @staticmethod
    def _safe_float(rec, field: str) -> float:
        # Reads a numeric field from a shapefile record safely
        try:
            value = rec[field]
        except (KeyError, IndexError, TypeError):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
