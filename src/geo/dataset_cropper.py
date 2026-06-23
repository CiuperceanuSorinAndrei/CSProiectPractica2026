from __future__ import annotations

import numpy as np
import xarray as xr

from src.geo.projection import GeoProjection


class DatasetCropper:
    _lon_min: float = None
    _lon_max: float = None
    _lat_min: float = None
    _lat_max: float = None

    def __init__(self, lon_min: float, lon_max: float, lat_min: float, lat_max: float):
        self.set_bbox(lon_min, lon_max, lat_min, lat_max)

    def set_bbox(self, lon_min: float, lon_max: float, lat_min: float, lat_max: float):
        self._lon_min = lon_min
        self._lon_max = lon_max
        self._lat_min = lat_min
        self._lat_max = lat_max

    # Crop bazat pe coordonate geografice (Lat/Lon)
    def crop(self, ds: xr.Dataset) -> xr.Dataset | None:
        proj_info = ds['geostationary_projection'].attrs
        h = proj_info['perspective_point_height']

        x_min_m, x_max_m, y_min_m, y_max_m = self._bbox_to_sat_meters(proj_info)

        x_vals = GeoProjection.scale_grid_values(ds['nx'].values, h)
        y_vals = GeoProjection.scale_grid_values(ds['ny'].values, h)

        # Identificare indici corespondenti in matrice
        x_indices = np.where((x_vals >= x_min_m) & (x_vals <= x_max_m))[0]
        y_indices = np.where((y_vals >= y_min_m) & (y_vals <= y_max_m))[0]

        if len(x_indices) == 0 or len(y_indices) == 0:
            print("Eroare: Coordonatele cerute sunt in afara imaginii satelitului.")
            return None

        # Decupare prin indexare (isel)
        return ds.isel(
            nx=slice(int(x_indices[0]), int(x_indices[-1]) + 1),
            ny=slice(int(y_indices[0]), int(y_indices[-1]) + 1),
        )

    # Transforma bbox-ul geografic in limite (metri) pe proiectia geostationara
    def _bbox_to_sat_meters(self, proj_info: dict) -> tuple[float, float, float, float]:
        transformer = GeoProjection.latlon_to_satellite(proj_info)
        xs, ys = transformer.transform(
            [self._lon_min, self._lon_max, self._lon_min, self._lon_max],
            [self._lat_min, self._lat_min, self._lat_max, self._lat_max],
        )
        return min(xs), max(xs), min(ys), max(ys)
