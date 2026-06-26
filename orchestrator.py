"""Orchestrator: coordoneaza procesarea cadrelor.

Encapsuleaza toata logica de procesare a unui cadru intr-o singura metoda process_frame(),
actionand ca un Dispatcher curat catre modulele specializate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import threading

import numpy as np

from src.core.storm_cell_detector import StormCellDetector
from src.core.storm_tracker import StormTracker
from src.core.advection_engine import AdvectionEngine
from src.core.evaluator import Evaluator
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
    roi_volume_m3: float
    predicted_roi_volume_m3: float
    predicted_volumes_horizons: dict[str, float]
    global_csi: dict[str, float]
    global_far: dict[str, float]
    global_pod: dict[str, float]
    global_fss: dict[str, float]


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
        """Proceseaza un cadru complet: citire -> crop -> detectie -> tracking -> advectie -> evaluare."""
        if not self._lock.acquire(blocking=True):
            return None
            
        ds = NetCdfReader(file_path).load_data()
        if ds is None:
            self._lock.release()
            return None

        try:
            # 1. IO & Geo Projection
            ds_cropped = DatasetCropper(lon_min, lon_max, lat_min, lat_max).crop(ds)
            if ds_cropped is None:
                return None

            proj_info = ds_cropped["geostationary_projection"].attrs
            lon_grid, lat_grid = GeoProjection.grid_to_latlon(
                ds_cropped["nx"].values, ds_cropped["ny"].values, proj_info,
            )

            rain_rate = ds_cropped["rr"].values.copy()
            rain_rate[rain_rate < 0] = 0.0
            rain_rate_masked = np.ma.masked_where(rain_rate < 0.1, rain_rate)

            # Masca de arie de interes (ROI)
            dist_grid = self._haversine_dist_grid(center_lat, center_lon, lat_grid, lon_grid)
            roi_mask = dist_grid <= radius_km
            max_rain = float(np.max(rain_rate[roi_mask])) if np.any(roi_mask) else 0.0

            # 2. Extractie Centroizi
            raw_storm_cells = self._detector.extract_cells(rain_rate)
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

            # 3. Tracking (Kalman + Dense Optical Flow)
            tracked_cells, flow = self._tracker.track(filtered_cells, rain_rate)

            # 4. Evaluare Metrici Trecut
            horizons = [(1, "15m"), (4, "1h"), (8, "2h"), (20, "5h")]
            csi, far, pod, fss = Evaluator.calculate_global_metrics(
                rain_rate, roi_mask, self._predictions_queue, horizons
            )

            # 5. Advectie Viitor
            sparse_preds, float_preds = AdvectionEngine.extrapolate(
                rain_rate, flow, tracked_cells, horizons
            )

            # Salvam in istoric pentru viitor
            self._predictions_queue.append(sparse_preds)
            if len(self._predictions_queue) > 25:
                self._predictions_queue.pop(0)

            # 6. Evaluare Volume Viitor
            roi_volume_m3, predicted_volumes = Evaluator.calculate_volumes(
                rain_rate, float_preds, roi_mask, lat_grid
            )

            # Calcul metrici de eroare pe celulele urmarite
            valid_errors = [c.get("prediction_error_pixels", 0.0) for c in tracked_cells if c.get("is_tracked", False)]
            size_errors = [c.get("size_error_percent", 0.0) for c in tracked_cells if c.get("is_tracked", False)]

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
                predicted_roi_volume_m3=predicted_volumes.get("15m", 0.0),
                predicted_volumes_horizons=predicted_volumes,
                global_csi=csi,
                global_far=far,
                global_pod=pod,
                global_fss=fss,
            )
        finally:
            self._lock.release()
            ds.close()

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
