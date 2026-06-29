from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo
import shapely.geometry as sgeom
import matplotlib.patheffects as patheffects


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
        polygon=None
    ) -> tuple[Figure, Axes, Any]:
        """Creeaza figura principala cu harta de precipitatii.

        Args:
            extent: (lon_min, lon_max, lat_min, lat_max)

        Returns:
            (fig, ax, im) - figura, axa cartopy si imaginea pcolormesh
        """
        lon_min, lon_max, lat_min, lat_max = extent
        import matplotlib.colors as mcolors
        
        # Fundal intunecat (Cinematic Dark Mode)
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
        fig.patch.set_facecolor('#111315') # Culoare margini figura
        ax.set_facecolor('#111315')
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Diferentiere pamant si apa folosind rezolutia 50m (care se descarca foarte rapid)
        land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='none', facecolor='#1a1c20')
        ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='#111315')
        ax.add_feature(land_50m, zorder=0)
        ax.add_feature(ocean_50m, zorder=0)
        ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=1.0, edgecolor="#343a40", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linestyle="-", linewidth=1.2, edgecolor="#495057", zorder=1)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#495057", alpha=0.5, zorder=1)
        gl.xlabel_style = {'color': '#adb5bd', 'size': 9}
        gl.ylabel_style = {'color': '#adb5bd', 'size': 9}

        # Paleta de culori specifica pentru radar meteo
        radar_colors = [
            "#00ECEC", "#01A0F6", "#0000F6",  # Precipitatii usoare (Bleu -> Albastru)
            "#00FF00", "#00C800", "#009000",  # Moderate (Verde deschis -> Verde inchis)
            "#FFFF00", "#E7C000",             # Moderate spre grele (Galben -> Portocaliu)
            "#FF9000", "#FF0000", "#D60000",  # Grele (Portocaliu inchis -> Rosu)
            "#C00000", "#FF00FF", "#9955C9"   # Extreme (Rosu inchis -> Magenta / Violet)
        ]
        radar_cmap = mcolors.LinearSegmentedColormap.from_list("Radar", radar_colors)

        levels = np.linspace(vmin, vmax, 20)
        im = ax.contourf(
            lon_grid, lat_grid, rain_rate_masked,
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap=radar_cmap, extend="max", alpha=0.85, zorder=2
        )
        cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.10, shrink=0.8)
        cb.set_label("Intensitate ploaie (mm/h)", color="#adb5bd")
        cb.ax.yaxis.set_tick_params(color="#adb5bd")
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="#adb5bd")

        if polygon is not None:
            # Daca avem poligon, adaugam un fundal alb transparent si bordura mai groasa pentru a face lacurile mici mai vizibile
            ax.add_geometries(
                [polygon], crs=ccrs.PlateCarree(),
                facecolor=(1.0, 1.0, 1.0, 0.25), edgecolor='white', linewidth=3.0, linestyle='-', zorder=5,
            )
        elif roi_center is not None and roi_radius_km is not None:
            c_lat, c_lon = roi_center
            ax.plot(c_lon, c_lat, "wo", markersize=6, transform=ccrs.PlateCarree(), zorder=5)
            circle_points = cgeo.Geodesic().circle(
                lon=c_lon, lat=c_lat, radius=roi_radius_km * 1000.0, n_samples=100, endpoint=False,
            )
            geom = sgeom.Polygon(circle_points)
            ax.add_geometries(
                [geom], crs=ccrs.PlateCarree(),
                facecolor=(1.0, 1.0, 1.0, 0.15), edgecolor='white', linewidth=2.5, linestyle='--', zorder=5,
            )

        if title:
            ax.set_title(title, fontsize=11, fontweight="bold", color="#f8f9fa")

        return fig, ax, im

    @staticmethod
    def draw_overlays(ax, tracked_cells: list[dict], lon_grid: np.ndarray, lat_grid: np.ndarray) -> None:
        """Deseneaza centroizii, contururile predictive si vectorii de deplasare."""
        proj = ccrs.PlateCarree()

        for cell in tracked_cells:
            cell_lon = cell["geo_lon"]
            cell_lat = cell["geo_lat"]

            # Marker centroid (Punct alb cu contur negru)
            ax.plot(cell_lon, cell_lat, "o", color="white", markeredgecolor="black", markeredgewidth=1.5, markersize=5, transform=proj, zorder=4)
            ax.text(
                cell_lon + 0.02, cell_lat + 0.02,
                f"#{cell['cell_id'][:4]}",
                color="white", fontsize=8, fontweight="bold", transform=proj,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="black")]
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
                line = ax.plot(
                    contour_lon, contour_lat,
                    color="white", linestyle="-", linewidth=1.5,
                    transform=proj, zorder=3,
                )
                line[0].set_path_effects([patheffects.withStroke(linewidth=3.5, foreground="black"), patheffects.Normal()])

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
                width=0.03,
                head_width=0.1, head_length=0.1,
                fc="white", ec="black",
                transform=proj, zorder=5, linewidth=1.0,
            )
