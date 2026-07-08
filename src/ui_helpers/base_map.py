from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo
import shapely.geometry as sgeom
import matplotlib.colors as mcolors

from src.config import RAIN_THRESHOLD_MIN, RAIN_VMAX

def resolve_scale(vmin: float | None, vmax: float | None) -> tuple[float, float]:
    # 1. Scale Resolution
    return (RAIN_THRESHOLD_MIN if vmin is None else vmin, RAIN_VMAX if vmax is None else vmax)

def setup_basemap(ax) -> None:
    # 2. Cartopy Layers
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='none', facecolor='#1a1c20')
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='#111315')
    ax.add_feature(land, zorder=0)
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=1.0, edgecolor="#343a40", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linestyle="-", linewidth=1.2, edgecolor="#495057", zorder=1)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#495057", alpha=0.5, zorder=1)
    gl.xlabel_style, gl.ylabel_style = {'color': '#adb5bd', 'size': 9}, {'color': '#adb5bd', 'size': 9}

def draw_rain(ax, lon, lat, rain_masked, vmin: float, vmax: float):
    # 3. Radar Colormap & Field
    radar_colors = [
        "#00ECEC", "#01A0F6", "#0000F6", "#00FF00", "#00C800", "#009000",
        "#FFFF00", "#E7C000", "#FF9000", "#FF0000", "#D60000", "#C00000",
        "#FF00FF", "#9955C9"
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("Radar", radar_colors)
    
    im = ax.contourf(lon, lat, rain_masked, levels=np.linspace(vmin, vmax, 20),
                     transform=ccrs.PlateCarree(), cmap=cmap, extend="max", alpha=0.85, zorder=2)
                     
    cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.92)
    cb.set_label("Rain Intensity (mm/h)", color="#adb5bd")
    cb.ax.yaxis.set_tick_params(color="#adb5bd")
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="#adb5bd")
    return im

def draw_roi(ax, center, radius_km, polygon) -> None:
    # 4. Region of Interest Highlighting
    if polygon:
        ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor=(1, 1, 1, 0.25),
                          edgecolor='white', linewidth=3.0, linestyle='-', zorder=5)
    elif center and radius_km:
        c_lat, c_lon = center
        ax.plot(c_lon, c_lat, "wo", markersize=6, transform=ccrs.PlateCarree(), zorder=5)
        circ = cgeo.Geodesic().circle(lon=c_lon, lat=c_lat, radius=radius_km * 1000.0, n_samples=100, endpoint=False)
        ax.add_geometries([sgeom.Polygon(circ)], crs=ccrs.PlateCarree(), facecolor=(1, 1, 1, 0.15),
                          edgecolor='white', linewidth=2.5, linestyle='--', zorder=5)
