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
import scipy.ndimage as ndimage

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
    predicted_volumes_horizons: dict[str, float] # Volumele prezise pe toate orizonturile
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
        if not self._lock.acquire(blocking=True):
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
            # Mascam doar valorile < 0.1 pentru a lasa burnita vizibila pe harta
            rain_rate_masked = np.ma.masked_where(rain_rate < 0.1, rain_rate)

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

            # Aproximare grosiera a ariei: unghiul de vizualizare oblic la poli dilata fizic pixelul
            pixel_area_km2 = 3.0 * (3.0 / np.cos(np.radians(lat_grid)))

            # Calculam volumul ACUMULAT din toate precipitațiile (inclusiv burnița de 0.1 mm/h)
            roi_volume_m3 = float(np.sum(rain_rate[roi_mask] * pixel_area_km2[roi_mask] * 250.0))

            # Faza 9: Global Advection via Sparse Interpolation
            import cv2
            flow_x, flow_y = StormTracker.generate_global_flow_field(rain_rate.shape, tracked_cells)
            
            horizons = [(1, "15m"), (4, "1h"), (8, "2h"), (20, "5h")]
            new_global_preds = {}
            predicted_volumes_horizons = {}
            total_predicted_volume_m3 = 0.0
            
            grid_h, grid_w = rain_rate.shape
            y_grid, x_grid = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
            
            # Advecție globală (toată harta dintr-o mișcare)
            for steps, name in horizons:
                map_x = x_grid - flow_x * steps
                map_y = y_grid - flow_y * steps
                
                # Mutăm precipitațiile
                shifted_rain = cv2.remap(rain_rate, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                
                # Decadere termodinamica
                decay_factor = 0.98 ** steps
                shifted_rain *= decay_factor
                
                # Estompare Gaussiana (Covariance Blurring) din Faza 7
                if steps > 1:
                    sigma = min(steps * 0.2, 3.0)
                    shifted_rain = ndimage.gaussian_filter(shifted_rain, sigma=sigma)
                    
                # Masca booleană pentru calcul CSI
                new_global_preds[name] = (shifted_rain >= RAIN_THRESHOLD_MIN).astype(np.float32)
                
                # Calcul Volum (toată ploaia din ROI, inclusiv aia fină > 0.0)
                vol = float(np.sum(shifted_rain[roi_mask] * pixel_area_km2[roi_mask] * 250.0))
                predicted_volumes_horizons[name] = vol
                
                if steps == 1:
                    total_predicted_volume_m3 = vol

            # Convertim in sparse matrix pentru a economisi memorie (salvam 95% RAM)
            sparse_preds = {name: sp.csr_matrix(mask) for name, mask in new_global_preds.items()}
            self._predictions_queue.append(sparse_preds)
            if len(self._predictions_queue) > 25:
                self._predictions_queue.pop(0)

            # Calcul metrici de eroare pe celulele urmarite
            valid_errors = [
                c.get("prediction_error_pixels", 0.0)
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
                num_tracked=len([c for c in tracked_cells if c.get("is_tracked", False)]),
                roi_volume_m3=roi_volume_m3,
                predicted_roi_volume_m3=total_predicted_volume_m3,
                predicted_volumes_horizons=predicted_volumes_horizons,
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
