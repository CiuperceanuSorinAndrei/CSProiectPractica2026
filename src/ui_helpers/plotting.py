from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo
import shapely.geometry as sgeom


class StormMapPlotter:
    """Randare matplotlib/cartopy a hartii de precipitatii si a overlay-urilor de tracking."""

    @staticmethod
    def create_figure(
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        rain_rate_masked: np.ma.MaskedArray,
        extent: tuple[float, float, float, float],
        vmin: float = 0.1,
        vmax: float = 12.0,
        title: str = "",
        roi_center: tuple[float, float] | None = None,
        roi_radius_km: float | None = None,
    ):
        """Creeaza figura principala cu harta de precipitatii.

        Args:
            extent: (lon_min, lon_max, lat_min, lat_max)

        Returns:
            (fig, ax, im) - figura, axa cartopy si imaginea pcolormesh
        """
        lon_min, lon_max, lat_min, lat_max = extent
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=1.5, edgecolor="black")
        ax.add_feature(cfeature.COASTLINE, linestyle="-", linewidth=1)
        ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.3)

        im = ax.pcolormesh(
            lon_grid, lat_grid, rain_rate_masked,
            transform=ccrs.PlateCarree(),
            cmap="Blues", vmin=vmin, vmax=vmax, shading="auto", alpha=0.85,
        )
        plt.colorbar(im, ax=ax, label="Intensitate ploaie (mm/h)", orientation="vertical", pad=0.10, shrink=0.8)

        if roi_center is not None and roi_radius_km is not None:
            c_lat, c_lon = roi_center
            ax.plot(c_lon, c_lat, "ro", markersize=6, transform=ccrs.PlateCarree(), zorder=5)
            circle_points = cgeo.Geodesic().circle(
                lon=c_lon, lat=c_lat, radius=roi_radius_km * 1000.0, n_samples=100, endpoint=False,
            )
            geom = sgeom.Polygon(circle_points)
            ax.add_geometries(
                [geom], crs=ccrs.PlateCarree(),
                facecolor='none', edgecolor='red', linewidth=2.5, linestyle='--', zorder=5,
            )

        if title:
            ax.set_title(title, fontsize=11, fontweight="bold")

        return fig, ax, im

    @staticmethod
    def draw_overlays(ax, tracked_cells: list[dict], lon_grid: np.ndarray, lat_grid: np.ndarray) -> None:
        """Deseneaza centroizii, contururile predictive si vectorii de deplasare."""
        proj = ccrs.PlateCarree()

        for cell in tracked_cells:
            cell_lon = cell["geo_lon"]
            cell_lat = cell["geo_lat"]

            # Marker centroid
            ax.plot(cell_lon, cell_lat, "kx", markersize=8, mew=2.5, transform=proj, zorder=4)
            ax.text(
                cell_lon + 0.02, cell_lat + 0.02,
                f"#{cell['cell_id'][:4]}",
                color="black", fontsize=8, fontweight="bold", transform=proj,
            )

            # Contur prezis din masca
            if "predicted_mask" in cell and cell.get("is_tracked", False):
                StormMapPlotter._draw_predicted_contour(ax, cell["predicted_mask"], lon_grid, lat_grid, proj)

            # Vector de deplasare al centroidului
            if cell.get("is_tracked", False):
                StormMapPlotter._draw_velocity_arrow(ax, cell, lon_grid, lat_grid, proj)

    @staticmethod
    def _draw_predicted_contour(ax, predicted_mask: np.ndarray, lon_grid: np.ndarray, lat_grid: np.ndarray, proj) -> None:
        """Deseneaza conturul verde al mastii predictive."""
        mask_uint8 = predicted_mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:
                continue

            contour_points = contour[:, 0, :]
            contour_lon = []
            contour_lat = []

            for px, py in contour_points:
                if 0 <= py < lon_grid.shape[0] and 0 <= px < lon_grid.shape[1]:
                    contour_lon.append(float(lon_grid[py, px]))
                    contour_lat.append(float(lat_grid[py, px]))

            if len(contour_lon) >= 3:
                contour_lon.append(contour_lon[0])
                contour_lat.append(contour_lat[0])
                ax.plot(
                    contour_lon, contour_lat,
                    color="#00FF00", linestyle="--", linewidth=1.8,
                    transform=proj, zorder=3,
                )

    @staticmethod
    def _draw_velocity_arrow(ax, cell: dict, lon_grid: np.ndarray, lat_grid: np.ndarray, proj) -> None:
        """Deseneaza sageata de deplasare a centroidului."""
        cell_lon = cell["geo_lon"]
        cell_lat = cell["geo_lat"]

        pred_y = int(np.clip(cell.get("predicted_centroid_y", cell["centroid_y"]), 0, lat_grid.shape[0] - 1))
        pred_x = int(np.clip(cell.get("predicted_centroid_x", cell["centroid_x"]), 0, lon_grid.shape[1] - 1))

        dx = lon_grid[pred_y, pred_x] - cell_lon
        dy = lat_grid[pred_y, pred_x] - cell_lat

        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            ax.arrow(
                cell_lon, cell_lat, dx, dy,
                head_width=0.03, head_length=0.03,
                fc="magenta", ec="magenta",
                transform=proj, zorder=5, linewidth=1.5,
            )
