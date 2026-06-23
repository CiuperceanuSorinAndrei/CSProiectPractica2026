import os
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


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
    def crop(self, ds: xr.Dataset) -> xr.Dataset:
        proj_info = ds['geostationary_projection'].attrs
        h = proj_info['perspective_point_height']

        x_min_m, x_max_m, y_min_m, y_max_m = self._bbox_to_sat_meters(proj_info, h)

        x_vals = ds['nx'].values
        y_vals = ds['ny'].values
        if np.max(np.abs(x_vals)) < 1.0:
            x_vals = x_vals * h
            y_vals = y_vals * h

        # Identificare indici corespondenti in matrice
        x_indices = np.where((x_vals >= x_min_m) & (x_vals <= x_max_m))[0]
        y_indices = np.where((y_vals >= y_min_m) & (y_vals <= y_max_m))[0]

        if len(x_indices) == 0 or len(y_indices) == 0:
            print("Eroare: Coordonatele cerute sunt in afara imaginii satelitului.")
            return None

        # Decupare prin indexare (isel)
        idx_x_min, idx_x_max = min(x_indices), max(x_indices)
        idx_y_min, idx_y_max = min(y_indices), max(y_indices)

        return ds.isel(nx=slice(idx_x_min, idx_x_max + 1), ny=slice(idx_y_min, idx_y_max + 1))

    # Transforma bbox-ul geografic in limite (metri) pe proiectia geostationara a satelitului
    def _bbox_to_sat_meters(self, proj_info: dict, h: float) -> tuple[float, float, float, float]:
        proj4_str = (
            f"+proj=geos +h={h} +lon_0={proj_info['longitude_of_projection_origin']} "
            f"+sweep={proj_info['sweep_angle_axis']} +a={proj_info['semi_major_axis']} "
            f"+b={proj_info['semi_minor_axis']} +units=m"
        )
        crs_sat = CRS.from_proj4(proj4_str)
        crs_latlon = CRS.from_epsg(4326)

        transformer = Transformer.from_crs(crs_latlon, crs_sat, always_xy=True)
        xs, ys = transformer.transform(
            [self._lon_min, self._lon_max, self._lon_min, self._lon_max],
            [self._lat_min, self._lat_min, self._lat_max, self._lat_max]
        )

        return min(xs), max(xs), min(ys), max(ys)


# --- Testing ---
if __name__ == "__main__":
    nume_fisier = "h60_20260613_1400_fdk.nc"
    cale_test = os.path.join("data", "raw", nume_fisier)

    ds = xr.open_dataset(cale_test, engine='netcdf4')

    cropper = DatasetCropper(lon_min=22.0, lon_max=26.0, lat_min=43.0, lat_max=46.0)
    ds_cropped = cropper.crop(ds)

    if ds_cropped is not None:
        print("Decupare reusita.")
        print("Dimensiuni:", ds_cropped.dims)
