from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from .base_map import resolve_scale, setup_basemap, draw_rain, draw_roi
from .overlays import draw_tracked_cells

class StormMapPlotter:
    # 1. Facade Entry Point
    @staticmethod
    def create_figure(
        lon_grid: np.ndarray, lat_grid: np.ndarray, rain_rate_masked: np.ma.MaskedArray,
        extent: tuple[float, float, float, float], vmin: float | None = None, vmax: float | None = None,
        title: str = "", roi_center: tuple[float, float] | None = None,
        roi_radius_km: float | None = None, polygon=None
    ):
        # 2. Main Visualization Initialization
        lon_min, lon_max, lat_min, lat_max = extent
        vmin, vmax = resolve_scale(vmin, vmax)

        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
        fig.patch.set_facecolor('#111315')
        ax.set_facecolor('#111315')
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # 3. Layer Render
        setup_basemap(ax)
        im = draw_rain(ax, lon_grid, lat_grid, rain_rate_masked, vmin, vmax)
        draw_roi(ax, roi_center, roi_radius_km, polygon)

        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", color="#f8f9fa", pad=10)

        fig.tight_layout(pad=1.5)
        return fig, ax, im

    @staticmethod
    def draw_overlays(ax, tracked_cells: list[dict], lon_grid: np.ndarray, lat_grid: np.ndarray) -> None:
        # 4. Proxy Layer
        draw_tracked_cells(ax, tracked_cells, lon_grid, lat_grid)
