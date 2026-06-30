from __future__ import annotations

from functools import lru_cache
import threading
import numpy as np
from pyproj import CRS, Transformer




class GeoProjection:
    """Transformari intre proiectia geostationary a satelitului si Lat/Lon (WGS84).

    Transformer-ele pyproj sunt cache-uite (lru_cache) deoarece sunt costisitoare
    de construit si se reutilizeaza identic pentru fiecare cadru cu aceeasi proiectie.
    """

    @staticmethod
    def scale_grid_values(vals: np.ndarray, perspective_height: float) -> np.ndarray:
        """Normalizeaza valorile nx/ny: daca sunt in radiani (<1), inmulteste cu h."""
        if np.max(np.abs(vals)) < 1.0:
            return vals * perspective_height
        return vals

    @staticmethod
    def satellite_to_latlon(proj_info: dict) -> Transformer:
        """Transformer: din coordonate geostationary (m) in Lat/Lon WGS84. Cached."""
        return GeoProjection._cached_transformer(GeoProjection._make_proj4(proj_info), "4326")

    @staticmethod
    def latlon_to_satellite(proj_info: dict) -> Transformer:
        """Transformer: din Lat/Lon WGS84 in coordonate geostationary (m). Cached."""
        return GeoProjection._cached_transformer("4326", GeoProjection._make_proj4(proj_info))

    @staticmethod
    def grid_to_latlon(x_vals: np.ndarray, y_vals: np.ndarray, proj_info: dict):
        """Converteste grila geostationary (nx, ny) in grile Lon/Lat WGS84."""
        h = proj_info["perspective_point_height"]
        x_scaled = GeoProjection.scale_grid_values(x_vals, h)
        y_scaled = GeoProjection.scale_grid_values(y_vals, h)
        x_grid, y_grid = np.meshgrid(x_scaled, y_scaled)
        return GeoProjection.satellite_to_latlon(proj_info).transform(x_grid, y_grid)

    # -----------------------------------------------------------------------
    # Intern
    # -----------------------------------------------------------------------
    @staticmethod
    def _make_proj4(proj_info: dict) -> str:
        """Construieste string-ul proj4 din atributele proiectiei geostationary."""
        h = proj_info["perspective_point_height"]
        return (
            f"+proj=geos +h={h} +lon_0={proj_info['longitude_of_projection_origin']} "
            f"+sweep={proj_info['sweep_angle_axis']} +a={proj_info['semi_major_axis']} "
            f"+b={proj_info['semi_minor_axis']} +units=m"
        )

    @staticmethod
    @lru_cache(maxsize=32)
    def _cached_transformer(source: str, target: str) -> Transformer:
        """Cache intern pentru Transformer-e pyproj (Thread-Safe via lru_cache)."""
        def _to_crs(val: str) -> CRS:
            if val.startswith("+proj"):
                return CRS.from_proj4(val)
            return CRS.from_epsg(int(val))
        return Transformer.from_crs(_to_crs(source), _to_crs(target), always_xy=True)
