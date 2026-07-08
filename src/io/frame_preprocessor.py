from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import netCDF4

from src.geo.projection import GeoProjection
from src.core.detection.storm_cell_detector import StormCellDetector
from src.config import RAIN_THRESHOLD_TRACKING
from src.core.domain import StormCell

@dataclass
class FrameGeometry:
    lon_grid: np.ndarray
    lat_grid: np.ndarray
    pixel_area_km2: np.ndarray
    roi_mask: np.ndarray
    y_slice: slice
    x_slice: slice
    roi_mask_fractional: np.ndarray = None

@dataclass
class FramePrep:
    rain_rate: np.ndarray
    filtered_cells: list[StormCell]
    max_rain: float

_detector = StormCellDetector(threshold=RAIN_THRESHOLD_TRACKING, min_size=2)

def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    # 1. Geodesic Math
    R = 6371.0
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2r - lat1r) / 2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin((lon2r - lon1r) / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def _read_grid_and_proj(file_path: str):
    # 2. NetCDF Grid Extraction
    with netCDF4.Dataset(file_path) as ds:
        nx, ny = np.asarray(ds.variables["nx"][:]), np.asarray(ds.variables["ny"][:])
        gp = ds.variables["geostationary_projection"]
        return nx, ny, {k: gp.getncattr(k) for k in gp.ncattrs()}

def compute_geometry(file_path: str, bbox: tuple, center: tuple, radius_km: float, polygon=None, catchment_polygon=None) -> FrameGeometry | None:
    # 3. Geometry Computation
    nx, ny, proj = _read_grid_and_proj(file_path)
    h = proj["perspective_point_height"]

    transformer = GeoProjection.latlon_to_satellite(proj)
    xs, ys = transformer.transform([bbox[0], bbox[1], bbox[0], bbox[1]], [bbox[2], bbox[2], bbox[3], bbox[3]])
    
    x_vals, y_vals = GeoProjection.scale_grid_values(nx, h), GeoProjection.scale_grid_values(ny, h)
    x_idx, y_idx = np.where((x_vals >= min(xs)) & (x_vals <= max(xs)))[0], np.where((y_vals >= min(ys)) & (y_vals <= max(ys)))[0]
    if not len(x_idx) or not len(y_idx): return None

    x_slice, y_slice = slice(int(x_idx[0]), int(x_idx[-1]) + 1), slice(int(y_idx[0]), int(y_idx[-1]) + 1)
    lon_grid, lat_grid = GeoProjection.grid_to_latlon(nx[x_slice], ny[y_slice], proj)
    
    dy_km = _haversine_km(lat_grid, lon_grid, lat_grid + np.gradient(lat_grid, axis=0), lon_grid + np.gradient(lon_grid, axis=0))
    dx_km = _haversine_km(lat_grid, lon_grid, lat_grid + np.gradient(lat_grid, axis=1), lon_grid + np.gradient(lon_grid, axis=1))
    
    roi_polygon = catchment_polygon if catchment_polygon is not None else polygon
    if roi_polygon:
        from src.geo.intersection import PolygonIntersection
        roi_mask_fractional = PolygonIntersection.create_fractional_mask(roi_polygon, lon_grid, lat_grid)
        roi_mask = roi_mask_fractional > 0.0
    else:
        roi_mask = _haversine_km(center[0], center[1], lat_grid, lon_grid) <= radius_km
        roi_mask_fractional = roi_mask.astype(np.float32)
        
    return FrameGeometry(lon_grid, lat_grid, dx_km * dy_km, roi_mask, y_slice, x_slice, roi_mask_fractional)

def _read_rain_window(file_path: str, y_slice: slice, x_slice: slice) -> np.ndarray | None:
    # 4. Data Extraction
    try:
        with netCDF4.Dataset(file_path) as ds:
            return np.ma.filled(ds.variables["rr"][y_slice, x_slice], np.nan).astype(np.float64, copy=False)
    except Exception: return None

def preprocess(file_path: str, geom: FrameGeometry, bbox: tuple) -> FramePrep | None:
    # 5. Core Preprocessing Pipeline
    rr = _read_rain_window(file_path, geom.y_slice, geom.x_slice)
    if rr is None: return None
        
    rr[np.isinf(rr)] = 0.0
    rr[np.where((rr < 0) & (~np.isnan(rr)))] = 0.0

    max_rain = float(np.nanmax(rr[geom.roi_mask])) if np.any(geom.roi_mask) and not np.all(np.isnan(rr[geom.roi_mask])) else 0.0

    lon_min, lon_max, lat_min, lat_max = bbox
    filtered_cells = []
    
    for cell in _detector.extract_cells(rr):
        y_idx, x_idx = int(cell.centroid_y), int(cell.centroid_x)
        if 0 <= y_idx < geom.lat_grid.shape[0] and 0 <= x_idx < geom.lon_grid.shape[1]:
            cell_lon, cell_lat = geom.lon_grid[y_idx, x_idx], geom.lat_grid[y_idx, x_idx]
            if np.isfinite(cell_lon) and np.isfinite(cell_lat) and lon_min <= cell_lon <= lon_max and lat_min <= cell_lat <= lat_max:
                cell.geo_lon, cell.geo_lat = float(cell_lon), float(cell_lat)
                filtered_cells.append(cell)
                
    return FramePrep(rain_rate=rr, filtered_cells=filtered_cells, max_rain=max_rain)
