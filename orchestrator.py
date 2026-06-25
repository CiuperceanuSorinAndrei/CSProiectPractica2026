"""Orchestrator: coordoneaza procesarea cadrelor si pastreaza starea intre ele.

Encapsuleaza toata logica de procesare a unui cadru (citire, crop, detectie,
tracking) intr-o singura metoda process_frame(). Starea cinematica (Kalman)
este detinuta de StormTracker; aici pastram doar masca globala prezisa pentru
metricile globale CSI/FAR/POD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import threading

import numpy as np
import scipy.sparse as sp

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
    global_csi: dict[str, float]
    global_far: dict[str, float]
    global_pod: dict[str, float]


class Orchestrator:
    """Coordoneaza procesarea cadrelor si pastreaza starea cinematica intre ele."""

    def __init__(self) -> None:
        self._detector = StormCellDetector(threshold=RAIN_THRESHOLD_MIN, min_size=2)
        self._tracker = StormTracker(max_dist_pixels=MAX_TRACKING_DISTANCE_PX)
        self._lock = threading.Lock()
        self._predictions_queue = []

    def reset_tracking(self) -> None:
        """Goleste complet starea de tracking (Kalman + masca globala prezisa)."""
        with self._lock:
            self._tracker.reset()
            self._predictions_queue.clear()

    def process_frame(
        self,
        file_path: str,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float,
    ) -> FrameResult | None:
        """Proceseaza un cadru complet: citire -> crop -> detectie -> tracking."""
        if not self._lock.acquire(blocking=False):
            return None
            
        ds = NetCdfReader(file_path).load_data()
        if ds is None:
            self._lock.release()
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
            rain_rate[rain_rate < 0] = 0.0
            rain_rate_masked = np.ma.masked_where(rain_rate < RAIN_THRESHOLD_MIN, rain_rate)

            # Calculam distanta pana la centru pentru a defini aria de interes (ROI) imediat
            dist_grid = self._haversine_dist_grid(center_lat, center_lon, lat_grid, lon_grid)
            roi_mask = dist_grid <= radius_km

            # Rata maxima se afiseaza doar pentru aria de interes, nu global
            if np.any(roi_mask):
                max_rain = float(np.max(rain_rate[roi_mask]))
            else:
                max_rain = 0.0

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

            # (roi_mask a fost deja calculat mai sus)

            # --- CALCUL METRICI GLOBALE (doar in ROI) ---
            global_csi, global_far, global_pod = {}, {}, {}
            obs_mask = (rain_rate >= RAIN_THRESHOLD_MIN) & roi_mask

            # Evaluare multi-orizont (15m, 1h, 2h, 5h)
            horizons = [(1, "15m"), (4, "1h"), (8, "2h"), (20, "5h")]
            for steps_back, name in horizons:
                if len(self._predictions_queue) >= steps_back:
                    past_pred_sparse = self._predictions_queue[-steps_back].get(name)
                    if past_pred_sparse is not None:
                        past_pred = past_pred_sparse.toarray()
                        if past_pred.shape == rain_rate.shape:
                            pred_mask = (past_pred >= RAIN_THRESHOLD_MIN) & roi_mask
                            if np.any(obs_mask) or np.any(pred_mask):
                                global_csi[name] = ForecastMetrics.csi(obs_mask, pred_mask)
                                global_far[name] = ForecastMetrics.far(obs_mask, pred_mask)
                                global_pod[name] = ForecastMetrics.pod(obs_mask, pred_mask)

            # Aproximare aria unui pixel (~3km * 3km / cos(lat)). Folosim latitudinile grilei.
            pixel_area_km2 = 3.0 * (3.0 / np.cos(np.radians(lat_grid)))

            # Calculam volumul ACUMULAT din toate precipitațiile (inclusiv burnița de 0.1 mm/h)
            roi_volume_m3 = float(np.sum(rain_rate[roi_mask] * pixel_area_km2[roi_mask] * 250.0))

            # Calcul Volum Prezis Acumulat in ROI pentru următorul pas (15 minute)
            # Evităm supra-numărarea (multi-counting) evaluând strict T+15m
            MAX_FORECAST_FRAMES = 1
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

                    coords = np.asarray(cell["coords"])
                    if len(coords) == 0:
                        continue
                        
                    grid_h, grid_w = rain_rate.shape

                    # Daca sta pe loc complet (viteza aproape 0), previne bucla artificiala pt 3 ore
                    if abs(vx) < 0.1 and abs(vy) < 0.1:
                        frames_to_sim = 1
                    else:
                        frames_to_sim = MAX_FORECAST_FRAMES

                    for step in range(1, frames_to_sim + 1):
                        shifted_y = np.rint(coords[:, 0] + step * vy).astype(int)
                        shifted_x = np.rint(coords[:, 1] + step * vx).astype(int)

                        # Filtram coordonatele care ies in afara grilei
                        valid = (shifted_y >= 0) & (shifted_y < grid_h) & (shifted_x >= 0) & (shifted_x < grid_w)
                        if not np.any(valid):
                            if step > 1 and total_predicted_volume_m3 > 0:
                                break
                            continue
                            
                        val_y = shifted_y[valid]
                        val_x = shifted_x[valid]

                        # Verificam suprapunerea directa cu bazinul
                        in_roi = roi_mask[val_y, val_x]
                        num_pixels_overlap = np.sum(in_roi)

                        if num_pixels_overlap > 0:
                            # Insumam strict aria pixelilor mutati care au cazut in ROI
                            area_overlap_km2 = np.sum(pixel_area_km2[val_y[in_roi], val_x[in_roi]])
                            # Aplicam factor de decadere (decay) pt volum: scade cu ~8% la fiecare cadru
                            decay_factor = 0.92 ** step
                            vol_step_m3 = float(area_overlap_km2 * mean_int * 250.0) * decay_factor
                            total_predicted_volume_m3 += vol_step_m3
                        elif step > 1 and total_predicted_volume_m3 > 0:
                            # Am iesit complet din ROI, n-are rost sa simulam restul pasilor din viitor
                            break

            # Construim noile masti globale pentru T+N
            new_global_preds = {"15m": np.zeros(rain_rate.shape, dtype=np.float32), 
                                "1h": np.zeros(rain_rate.shape, dtype=np.float32),
                                "2h": np.zeros(rain_rate.shape, dtype=np.float32),
                                "5h": np.zeros(rain_rate.shape, dtype=np.float32)}
                                
            for cell in tracked_cells:
                if cell.get("is_tracked", False) and "predicted_masks" in cell:
                    for name in new_global_preds.keys():
                        if cell["predicted_masks"].get(name) is not None:
                            new_global_preds[name] = np.maximum(new_global_preds[name], cell["predicted_masks"][name])

            # Convertim in sparse matrix pentru a economisi memorie (salvam 95% RAM)
            sparse_preds = {name: sp.csr_matrix(mask) for name, mask in new_global_preds.items()}
            self._predictions_queue.append(sparse_preds)
            if len(self._predictions_queue) > 25:
                self._predictions_queue.pop(0)

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
            self._lock.release()
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
