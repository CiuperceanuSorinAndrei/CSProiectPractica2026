from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from src.core.storm_tracker import StormTracker
from src.core.advection_engine import AdvectionEngine
from src.core.evaluator import Evaluator
from frame_preprocessor import FramePrep, FrameGeometry
from config import RAIN_THRESHOLD_MIN

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
    instant_predicted_volumes: dict[str, float]
    global_csi: dict[str, float]
    global_far: dict[str, float]
    global_pod: dict[str, float]
    global_fss: dict[str, float]

class FrameProcessor:
    """Serviciu de domeniu stateless. Primeste input-uri decodate si intoarce FrameResult."""
    
    @staticmethod
    def process(
        prep: FramePrep, 
        geom: FrameGeometry, 
        tracker: StormTracker, 
        predictions_queue: list
    ) -> FrameResult:
        rain_rate = prep.rain_rate
        roi_mask = geom.roi_mask

        # Copii superficiale pentru a nu muta celulele memoizate
        cells_for_tracking = [dict(c) for c in prep.filtered_cells]
        tracked_cells, flow = tracker.track(cells_for_tracking, rain_rate)

        horizons = [(2, "30m"), (4, "1h"), (8, "2h")]
        csi, far, pod, fss = Evaluator.calculate_global_metrics(
            rain_rate, roi_mask, predictions_queue, horizons
        )

        sparse_preds, float_preds = AdvectionEngine.extrapolate(
            rain_rate, flow, tracked_cells, horizons
        )

        predictions_queue.append(sparse_preds)
        if len(predictions_queue) > 25:
            predictions_queue.pop(0)

        roi_volume_m3, predicted_volumes, instant_predicted_volumes = Evaluator.calculate_volumes(
            rain_rate, float_preds, roi_mask, geom.pixel_area_km2, horizons
        )

        valid_errors = [c.get("prediction_error_pixels", 0.0) for c in tracked_cells if c.get("is_tracked", False)]
        size_errors = [c.get("size_error_percent", 0.0) for c in tracked_cells if c.get("is_tracked", False)]

        rain_rate_masked = np.ma.masked_where(rain_rate < RAIN_THRESHOLD_MIN, rain_rate)
        
        return FrameResult(
            tracked_cells=tracked_cells,
            rain_rate=rain_rate,
            rain_rate_masked=rain_rate_masked,
            lon_grid=geom.lon_grid,
            lat_grid=geom.lat_grid,
            max_rain=prep.max_rain,
            mean_centroid_error=float(np.mean(valid_errors)) if valid_errors else 0.0,
            mean_size_error=float(np.mean(size_errors)) if size_errors else 0.0,
            num_tracked=len([c for c in tracked_cells if c.get("is_tracked", False)]),
            roi_volume_m3=roi_volume_m3,
            predicted_roi_volume_m3=predicted_volumes.get("1h", 0.0),
            predicted_volumes_horizons=predicted_volumes,
            instant_predicted_volumes=instant_predicted_volumes,
            global_csi=csi,
            global_far=far,
            global_pod=pod,
            global_fss=fss,
        )
