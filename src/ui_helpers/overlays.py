from __future__ import annotations

import numpy as np
import cv2
import cartopy.crs as ccrs
import matplotlib.patheffects as patheffects

def draw_tracked_cells(ax, cells: list[dict], lon: np.ndarray, lat: np.ndarray) -> None:
    # 1. Overlay Integration
    proj = ccrs.PlateCarree()

    for c in cells:
        c_lon, c_lat = c["geo_lon"], c["geo_lat"]

        # Marker centroid
        ax.plot(c_lon, c_lat, "o", color="white", markeredgecolor="black",
                markeredgewidth=1.5, markersize=5, transform=proj, zorder=4)
        ax.text(c_lon + 0.02, c_lat + 0.02, f"#{c['cell_id'][:4]}", color="white",
                fontsize=8, fontweight="bold", transform=proj,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="black")])

        # Velocity Arrow
        if c.get("is_tracked", False):
            _draw_velocity(ax, c, lon, lat, proj)


def _draw_velocity(ax, cell: dict, lon: np.ndarray, lat: np.ndarray, proj) -> None:
    # 3. Trajectory Vector
    c_lon, c_lat = cell["geo_lon"], cell["geo_lat"]
    py = int(np.clip(cell.get("predicted_centroid_y", cell["centroid_y"]), 0, lat.shape[0] - 1))
    px = int(np.clip(cell.get("predicted_centroid_x", cell["centroid_x"]), 0, lon.shape[1] - 1))

    dx, dy = lon[py, px] - c_lon, lat[py, px] - c_lat

    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        ax.arrow(c_lon, c_lat, dx, dy, width=0.03, head_width=0.1, head_length=0.1,
                 fc="white", ec="black", transform=proj, zorder=5, linewidth=1.0)
