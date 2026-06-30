"""Preprocesare stateless a unui cadru cu netCDF4 brut (fara overhead-ul xarray).

Aceleasi operatii ca vechea cale xarray (crop -> proiectie -> detectie -> filtrare), dar
deschidem fisierul direct cu netCDF4 si citim DOAR fereastra (hyperslab) ceruta. xarray
construia tot Dataset-ul si decoda CF la fiecare deschidere (~280 ms/cadru); aici costul
scade la ~10-20 ms/cadru. Valorile rezultate sunt identice cu cele din calea xarray.

Functiile sunt stateless, folosite atat de foreground (work-steal la cache-miss) cat si de
thread-ul de warm-up din fundal. Modulul traieste in radacina (langa orchestrator.py) ca sa
nu antreneze __init__-ul pachetului dashboard (import circular).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import netCDF4

from src.geo.projection import GeoProjection
from src.core.detection.storm_cell_detector import StormCellDetector
from config import RAIN_THRESHOLD_MIN, RAIN_THRESHOLD_TRACKING


@dataclass
class FrameGeometry:
    """Geometrie derivata din (bbox, centru, raza) + grila fixa a satelitului. Identica pentru
    toate cadrele cu aceeasi vizualizare, deci se calculeaza o singura data (grid-reuse).
    Contine si slice-urile de crop, ca sa citim doar fereastra de interes din fiecare fisier."""
    lon_grid: np.ndarray
    lat_grid: np.ndarray
    pixel_area_km2: np.ndarray
    roi_mask: np.ndarray
    y_slice: slice
    x_slice: slice
    roi_mask_fractional: np.ndarray = None


from src.core.domain import StormCell

@dataclass
class FramePrep:
    """Rezultatul etapei stateless de preprocesare a unui cadru (memoizabil per fisier)."""
    rain_rate: np.ndarray
    filtered_cells: list[StormCell]
    max_rain: float


# Detector stateless, reutilizat pentru toate cadrele.
_detector = StormCellDetector(threshold=RAIN_THRESHOLD_TRACKING, min_size=2)


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Distanta Haversine (km) intre un punct fix si un grid de puncte."""
    R = 6371.0
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _read_grid_and_proj(file_path: str):
    """Citeste coordonatele grilei (nx, ny) si atributele proiectiei geostationare."""
    ds = netCDF4.Dataset(file_path)
    try:
        nx = np.asarray(ds.variables["nx"][:])
        ny = np.asarray(ds.variables["ny"][:])
        gp = ds.variables["geostationary_projection"]
        proj = {k: gp.getncattr(k) for k in gp.ncattrs()}
        return nx, ny, proj
    finally:
        ds.close()


def compute_geometry(file_path: str, bbox: tuple, center: tuple, radius_km: float, polygon=None) -> FrameGeometry | None:
    """Construieste geometria (slice-uri crop, grile Lon/Lat, ROI, arie pixel) dintr-un fisier
    exemplu. Grila satelitului e fixa, deci e valabila pentru toate cadrele cu aceeasi geometrie.
    Intoarce None daca bbox-ul cade in afara imaginii satelitului."""
    lon_min, lon_max, lat_min, lat_max = bbox
    center_lat, center_lon = center
    nx, ny, proj = _read_grid_and_proj(file_path)
    h = proj["perspective_point_height"]

    # bbox geografic -> limite (metri) pe proiectia geostationara (ca DatasetCropper)
    transformer = GeoProjection.latlon_to_satellite(proj)
    xs, ys = transformer.transform([lon_min, lon_max, lon_min, lon_max],
                                   [lat_min, lat_min, lat_max, lat_max])
    x_min_m, x_max_m, y_min_m, y_max_m = min(xs), max(xs), min(ys), max(ys)

    x_vals = GeoProjection.scale_grid_values(nx, h)
    y_vals = GeoProjection.scale_grid_values(ny, h)
    x_idx = np.where((x_vals >= x_min_m) & (x_vals <= x_max_m))[0]
    y_idx = np.where((y_vals >= y_min_m) & (y_vals <= y_max_m))[0]
    if len(x_idx) == 0 or len(y_idx) == 0:
        return None  # in afara imaginii satelitului

    x_slice = slice(int(x_idx[0]), int(x_idx[-1]) + 1)
    y_slice = slice(int(y_idx[0]), int(y_idx[-1]) + 1)

    lon_grid, lat_grid = GeoProjection.grid_to_latlon(nx[x_slice], ny[y_slice], proj)
    
    # Phase 2: Exact pixel area using geodesic gradients instead of 1/cos(lat) approximation
    lat_diff_y = np.gradient(lat_grid, axis=0)
    lon_diff_y = np.gradient(lon_grid, axis=0)
    lat_diff_x = np.gradient(lat_grid, axis=1)
    lon_diff_x = np.gradient(lon_grid, axis=1)
    
    dy_km = _haversine_km(lat_grid, lon_grid, lat_grid + lat_diff_y, lon_grid + lon_diff_y)
    dx_km = _haversine_km(lat_grid, lon_grid, lat_grid + lat_diff_x, lon_grid + lon_diff_x)
    pixel_area_km2 = dx_km * dy_km
    if polygon is not None:
        from src.geo.intersection import PolygonIntersection
        roi_mask_fractional = PolygonIntersection.create_fractional_mask(polygon, lon_grid, lat_grid)
        roi_mask = roi_mask_fractional > 0.0
    else:
        roi_mask = _haversine_km(center_lat, center_lon, lat_grid, lon_grid) <= radius_km
        roi_mask_fractional = roi_mask.astype(np.float32)
        
    return FrameGeometry(lon_grid, lat_grid, pixel_area_km2, roi_mask, y_slice, x_slice, roi_mask_fractional)


def _read_rain_window(file_path: str, y_slice: slice, x_slice: slice) -> np.ndarray | None:
    """Citeste doar fereastra de ploaie (rr[y_slice, x_slice]), decodand fill-ul ca NaN
    (identic cu xarray)."""
    ds = netCDF4.Dataset(file_path)
    try:
        data = ds.variables["rr"][y_slice, x_slice]  # masked array (auto mask+scale, default)
        return np.ma.filled(data, np.nan).astype(np.float64, copy=False)
    except Exception:
        return None
    finally:
        ds.close()


def preprocess(file_path: str, geom: FrameGeometry, bbox: tuple) -> FramePrep | None:
    """Citeste fereastra de ploaie + detectie celule + filtrare la BBox -> FramePrep.
    Intoarce None daca fisierul nu poate fi citit."""
    rr = _read_rain_window(file_path, geom.y_slice, geom.x_slice)
    if rr is None:
        return None
        
    # Phase 2: Preserve NaNs as missing data instead of coercing to 0.0
    rr[np.isinf(rr)] = 0.0
    # Using np.where to safely check < 0 even with NaNs present
    rr[np.where((rr < 0) & (~np.isnan(rr)))] = 0.0

    max_rain = float(np.nanmax(rr[geom.roi_mask])) if np.any(geom.roi_mask) and not np.all(np.isnan(rr[geom.roi_mask])) else 0.0

    lon_grid, lat_grid = geom.lon_grid, geom.lat_grid
    lon_min, lon_max, lat_min, lat_max = bbox
    filtered_cells = []
    for cell in _detector.extract_cells(rr):
        y_idx = int(cell.centroid_y)
        x_idx = int(cell.centroid_x)
        if 0 <= y_idx < lat_grid.shape[0] and 0 <= x_idx < lon_grid.shape[1]:
            cell_lon = lon_grid[y_idx, x_idx]
            cell_lat = lat_grid[y_idx, x_idx]
            if (
                np.isfinite(cell_lon) and np.isfinite(cell_lat)
                and lon_min <= cell_lon <= lon_max
                and lat_min <= cell_lat <= lat_max
            ):
                cell.geo_lon = float(cell_lon)
                cell.geo_lat = float(cell_lat)
                filtered_cells.append(cell)
    return FramePrep(rain_rate=rr, filtered_cells=filtered_cells, max_rain=max_rain)
