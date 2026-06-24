"""Orchestrator: coordoneaza procesarea cadrelor si pastreaza starea intre ele.

Encapsuleaza toata logica de procesare a unui cadru (citire, crop, detectie,
tracking) intr-o singura metoda process_frame(). Starea cinematica (Kalman)
este detinuta de StormTracker; aici pastram doar masca globala prezisa pentru
metricile globale CSI/FAR/POD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.storm_cell_detector import StormCellDetector
from src.core.storm_tracker import StormTracker
from src.core.forecast_metrics import ForecastMetrics
from src.geo.dataset_cropper import DatasetCropper
from src.geo.projection import GeoProjection
from src.io.netcdf_reader import NetCdfReader

from config import RAIN_THRESHOLD_MIN, MAX_TRACKING_DISTANCE_PX


@dataclass
class FrameResult:
    """Rezultatul procesarii unui cadru complet."""
    tracked_cells: list[dict[str, Any]]
    rain_rate: np.ndarray
    rain_rate_masked: np.ma.MaskedArray
    lon_grid: np.ndarray
    lat_grid: np.ndarray
    max_rain: float
    mean_centroid_error: float
    mean_size_error: float
    num_tracked: int
    roi_volume_m3: float  # Volumul de precipitatii in ROI in metri cubi / ora
    predicted_roi_volume_m3: float  # Volumul prezis in ROI pentru cadrul urmator
    global_csi: float | None
    global_far: float | None
    global_pod: float | None


class Orchestrator:
    """Coordoneaza procesarea cadrelor si pastreaza starea cinematica intre ele."""

    def __init__(self) -> None:
        self._detector = StormCellDetector(threshold=RAIN_THRESHOLD_MIN, min_size=2)
        self._tracker = StormTracker(max_dist_pixels=MAX_TRACKING_DISTANCE_PX)
        self._previous_global_predicted_mask: Any = None

    def reset_tracking(self) -> None:
        """Goleste complet starea de tracking (Kalman + masca globala prezisa)."""
        self._tracker.reset()
        self._previous_global_predicted_mask = None

    def process_frame(
        self,
        file_path: str,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float,
    ) -> FrameResult | None:
        """Proceseaza un cadru complet: citire -> crop -> detectie -> tracking.

        Returns:
            FrameResult cu toate datele necesare pentru vizualizare, sau None daca esueaza.
        """
        ds = NetCdfReader(file_path).load_data()
        if ds is None:
            return None

        try:
            ds_cropped = DatasetCropper(lon_min, lon_max, lat_min, lat_max).crop(ds)
            if ds_cropped is None:
                return None

            # Proiectie geostationary -> Lat/Lon (transformer cached)
            proj_info = ds_cropped["geostationary_projection"].attrs
            lon_grid, lat_grid = GeoProjection.grid_to_latlon(
                ds_cropped["nx"].values, ds_cropped["ny"].values, proj_info,
            )

            # Pregatire matrice precipitatii
            rain_rate = ds_cropped["rr"].values.copy()
            rain_rate = np.nan_to_num(rain_rate, nan=0.0)
            rain_rate[rain_rate < 0] = 0.0
            rain_rate_masked = np.ma.masked_where(rain_rate < 0.1, rain_rate)

            max_rain = float(np.max(rain_rate))

            # Detectie celule convective
            raw_storm_cells = self._detector.extract_cells(rain_rate)

            # Filtrare: pastram doar celulele din BBox cu coordonate valide
            filtered_cells = []
            for cell in raw_storm_cells:
                y_idx = int(cell["centroid_y"])
                x_idx = int(cell["centroid_x"])
                if 0 <= y_idx < lat_grid.shape[0] and 0 <= x_idx < lon_grid.shape[1]:
                    cell_lon = lon_grid[y_idx, x_idx]
                    cell_lat = lat_grid[y_idx, x_idx]
                    if (
                        np.isfinite(cell_lon) and np.isfinite(cell_lat)
                        and lon_min <= cell_lon <= lon_max
                        and lat_min <= cell_lat <= lat_max
                    ):
                        cell["geo_lon"] = float(cell_lon)
                        cell["geo_lat"] = float(cell_lat)
                        filtered_cells.append(cell)

            # Tracking cinematic hibrid (Kalman + Optical Flow). StormTracker gestioneaza
            # intern resetarea la schimbarea rezolutiei grilei (zoom in/out pe harta).
            tracked_cells = self._tracker.track(filtered_cells, rain_rate)

            # Calcul Volum Precipitatii in ROI circular (Raza de la centru)
            dist_grid = self._haversine_dist_grid(center_lat, center_lon, lat_grid, lon_grid)
            roi_mask = dist_grid <= radius_km

            # --- CALCUL METRICI GLOBALE (doar in ROI) ---
            global_csi, global_far, global_pod = None, None, None
            prev_global = self._previous_global_predicted_mask
            if prev_global is not None and prev_global.shape == rain_rate.shape:
                obs_mask = (rain_rate >= 0.1) & roi_mask
                pred_mask = prev_global & roi_mask

                # Inregistram metrici doar daca exista activitate in ROI (ploaie observata sau prezisa)
                if np.any(obs_mask) or np.any(pred_mask):
                    global_csi = ForecastMetrics.csi(obs_mask, pred_mask)
                    global_far = ForecastMetrics.far(obs_mask, pred_mask)
                    global_pod = ForecastMetrics.pod(obs_mask, pred_mask)

            # Aproximare aria unui pixel (~3km * 3km / cos(lat)). Folosim latitudinile grilei.
            pixel_area_km2 = 3.0 * (3.0 / np.cos(np.radians(lat_grid)))

            # 1 mm/h ploaie = 0.25 mm/15min = 250 m^3/km^2 acumulati in 15 minute
            valid_rain_mask = (rain_rate >= 0.1) & roi_mask
            roi_volume_m3 = float(np.sum(rain_rate[valid_rain_mask] * pixel_area_km2[valid_rain_mask] * 250.0))

            # Calcul Volum Prezis Acumulat in ROI pentru urmatoarele N cadre (Max 3 ore = 12 cadre)
            MAX_FORECAST_FRAMES = 12
            total_predicted_volume_m3 = 0.0

            for cell in tracked_cells:
                # Verificam daca centroizii curenti sunt in ROI
                c_dist = self._haversine_dist_grid(
                    center_lat, center_lon, np.array([cell["geo_lat"]]), np.array([cell["geo_lon"]]),
                )[0]
                cell["in_roi"] = bool(c_dist <= radius_km)

                # Simulam si integram volumul pe traiectoria viitoare
                if cell.get("is_tracked", False) and "coords" in cell:
                    mean_int = cell.get("mean_intensity", 0.0)
                    vx = cell.get("v_x", 0.0)
                    vy = cell.get("v_y", 0.0)

                    # Reconstruim masca curenta
                    c_mask = np.zeros(rain_rate.shape, dtype=bool)
                    for cy, cx in cell["coords"]:
                        c_mask[int(cy), int(cx)] = True

                    # Daca sta pe loc complet (viteza aproape 0), previne bucla artificiala pt 3 ore
                    if abs(vx) < 0.1 and abs(vy) < 0.1:
                        frames_to_sim = 1
                    else:
                        frames_to_sim = MAX_FORECAST_FRAMES

                    for step in range(1, frames_to_sim + 1):
                        # translate_mask(mask, shift_y, shift_x): vy pe axa y (randuri), vx pe axa x (coloane)
                        future_mask = StormTracker.translate_mask(c_mask, step * vy, step * vx)
                        # astype(bool) evita fancy-indexing pe pixel_area_km2 (masca e uint8)
                        overlap_mask = future_mask.astype(bool) & roi_mask
                        num_pixels_overlap = np.sum(overlap_mask)

                        if num_pixels_overlap > 0:
                            area_overlap_km2 = np.sum(pixel_area_km2[overlap_mask])
                            vol_step_m3 = float(area_overlap_km2 * mean_int * 250.0)
                            total_predicted_volume_m3 += vol_step_m3
                        elif step > 1 and total_predicted_volume_m3 > 0:
                            # Am iesit deja din ROI, n-are rost sa simulam restul cadrelor
                            break

            # Construim noua masca globala prezisa pentru T+1
            new_global_pred_mask = np.zeros(rain_rate.shape, dtype=bool)
            for cell in tracked_cells:
                if "predicted_mask" in cell and cell.get("is_tracked", False):
                    new_global_pred_mask |= cell["predicted_mask"].astype(bool)

            # Salvam starea globala pentru cadrul urmator
            self._previous_global_predicted_mask = new_global_pred_mask

            # Calcul metrici de eroare pe celulele urmarite
            valid_errors = [
                c["prediction_error_pixels"]
                for c in tracked_cells
                if c.get("is_tracked", False)
            ]
            size_errors = [
                c.get("size_error_percent", 0.0)
                for c in tracked_cells
                if c.get("is_tracked", False)
            ]

            return FrameResult(
                tracked_cells=tracked_cells,
                rain_rate=rain_rate,
                rain_rate_masked=rain_rate_masked,
                lon_grid=lon_grid,
                lat_grid=lat_grid,
                max_rain=max_rain,
                mean_centroid_error=float(np.mean(valid_errors)) if valid_errors else 0.0,
                mean_size_error=float(np.mean(size_errors)) if size_errors else 0.0,
                num_tracked=len(valid_errors),
                roi_volume_m3=roi_volume_m3,
                predicted_roi_volume_m3=total_predicted_volume_m3,
                global_csi=global_csi,
                global_far=global_far,
                global_pod=global_pod,
            )
        finally:
            ds.close()

    # Distanta Haversine (km) intre un punct fix si un grid de puncte
    @staticmethod
    def _haversine_dist_grid(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c
