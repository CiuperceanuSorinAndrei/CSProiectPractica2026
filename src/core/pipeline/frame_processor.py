from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from src.core.tracking.storm_tracker import StormTracker
from src.core.nowcast.advection_engine import AdvectionEngine
from src.core.metrics.evaluator import Evaluator
from src.io.frame_preprocessor import FramePrep, FrameGeometry
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
    roi_map_mm: float
    predicted_roi_map_mm: float
    predicted_volumes_horizons: dict[str, float]
    instant_predicted_volumes: dict[str, float]
    raw_tracked_cells: list[Any] = None
    raw_predicted_cells: dict[str, list[Any]] = None
    sparse_preds: Any = None

class FrameProcessor:
    """Serviciu de domeniu stateless. Primeste input-uri decodate si intoarce FrameResult."""
    
    @staticmethod
    def process(
        prep: FramePrep, 
        geom: FrameGeometry, 
        tracker: StormTracker, 
        predictions_queue: list,
        advection_engine: AdvectionEngine
    ) -> FrameResult:
        rain_rate = prep.rain_rate
        roi_mask = geom.roi_mask

        # Copii superficiale cu istoric profund pentru a nu muta celulele memoizate
        cells_for_tracking = [c.clone() for c in prep.filtered_cells]
        tracked_cells, flow = tracker.track(cells_for_tracking, rain_rate)

        # ponytail: 15m delay calibration. Step 2 (30m from image) = 15m real-time forecast.
        # Step 5 (1h15m from image) = 1h real-time forecast. Step 9 = 2h real-time.
        horizons = [(2, "15m"), (5, "1h"), (9, "2h")]

        sparse_preds, float_preds, predicted_cells_dict = advection_engine.extrapolate(
            rain_rate, flow, tracked_cells, horizons
        )

        roi_map_mm, predicted_volumes, instant_predicted_volumes = Evaluator.calculate_volumes(
            rain_rate, float_preds, roi_mask, geom.pixel_area_km2, horizons,
            getattr(geom, 'roi_mask_fractional', None)
        )

        valid_errors = [getattr(c, "prediction_error_pixels", 0.0) for c in tracked_cells if getattr(c, "is_tracked", False)]
        size_errors = [getattr(c, "size_error_percent", 0.0) for c in tracked_cells if getattr(c, "is_tracked", False)]

        rain_rate_masked = np.ma.masked_where(rain_rate < RAIN_THRESHOLD_MIN, rain_rate)
        
        # V27: DTO Adapter - Convertim obiectele de domeniu în dicționare serializabile pentru Dash
        tracked_cells_dicts = [c.as_dict() for c in tracked_cells]
        
        return FrameResult(
            tracked_cells=tracked_cells_dicts,
            raw_tracked_cells=tracked_cells,
            raw_predicted_cells=predicted_cells_dict,
            rain_rate=rain_rate,
            rain_rate_masked=rain_rate_masked,
            lon_grid=geom.lon_grid,
            lat_grid=geom.lat_grid,
            max_rain=prep.max_rain,
            mean_centroid_error=float(np.mean(valid_errors)) if valid_errors else 0.0,
            mean_size_error=float(np.mean(size_errors)) if size_errors else 0.0,
            num_tracked=len([c for c in tracked_cells if getattr(c, "is_tracked", False)]),
            roi_map_mm=roi_map_mm,
            predicted_roi_map_mm=predicted_volumes.get("1h", 0.0),
            predicted_volumes_horizons=predicted_volumes,
            instant_predicted_volumes=instant_predicted_volumes,
            sparse_preds=sparse_preds,
        )
