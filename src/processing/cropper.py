import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

def crop_dataset_to_bbox(ds: xr.Dataset, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> xr.Dataset:
    """Crop bazat pe coordonate geografice (Lat/Lon)."""
    proj_info = ds['geostationary_projection'].attrs
    h = proj_info['perspective_point_height']
    
    # Definiere proiectii pentru transformare
    proj4_str = (
        f"+proj=geos +h={h} +lon_0={proj_info['longitude_of_projection_origin']} "
        f"+sweep={proj_info['sweep_angle_axis']} +a={proj_info['semi_major_axis']} "
        f"+b={proj_info['semi_minor_axis']} +units=m"
    )
    crs_sat = CRS.from_proj4(proj4_str)
    crs_latlon = CRS.from_epsg(4326)
    
    transformer = Transformer.from_crs(crs_latlon, crs_sat, always_xy=True)
    xs, ys = transformer.transform([lon_min, lon_max, lon_min, lon_max], [lat_min, lat_min, lat_max, lat_max])
    
    x_min_m, x_max_m = min(xs), max(xs)
    y_min_m, y_max_m = min(ys), max(ys)
    
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