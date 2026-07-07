from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from src.core.tracking.storm_tracker import StormTracker
from src.core.nowcast.advection_engine import AdvectionEngine
from src.core.metrics.evaluator import Evaluator
from src.core.constants import DEFAULT_HORIZONS
from src.io.frame_preprocessor import FramePrep, FrameGeometry
from src.config import RAIN_THRESHOLD_MIN

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

class FrameProcessor:
    """Serviciu de domeniu stateless. Primeste input-uri decodate si intoarce FrameResult."""
    
    @staticmethod
    def process(
        prep: FramePrep, 
        geom: FrameGeometry, 
        tracker: StormTracker, 
        advection_engine: AdvectionEngine,
        frame_time=None, run_mode="historic"
    ) -> FrameResult:
        rain_rate = prep.rain_rate
        roi_mask = geom.roi_mask

        # Clone cells to avoid mutating memoized instances
        cells_for_tracking = [c.clone() for c in prep.filtered_cells]
        tracked_cells = tracker.track(cells_for_tracking, rain_rate)

        # Dynamic Horizons based on actual frame delay
        import datetime, math
        if run_mode == "live" and frame_time is not None:
            now = datetime.datetime.utcnow()
            # Prevent negative delay if clock is slightly off
            delay_minutes = max(0.0, (now - frame_time).total_seconds() / 60.0)
            
            # Predict further ahead to compensate for delay. 
            # E.g., if delay is 25m, we are already 25m behind.
            # To predict 15m into the future (from NOW), we need 25+15 = 40m from the FRAME.
            step_15m = int(math.ceil((delay_minutes + 15) / 15.0))
            step_1h  = int(math.ceil((delay_minutes + 60) / 15.0))
            step_2h  = int(math.ceil((delay_minutes + 120) / 15.0))
            
            # Ensure steps are monotonic and > 0
            step_15m = max(1, step_15m)
            step_1h = max(step_15m + 1, step_1h)
            step_2h = max(step_1h + 1, step_2h)
            
            horizons = [(step_15m, "15m"), (step_1h, "1h"), (step_2h, "2h")]
        else:
            # Historic mode: static steps (assuming fixed 15m delay calibration)
            horizons = list(DEFAULT_HORIZONS)

        float_preds = advection_engine.extrapolate(
            rain_rate, tracked_cells, horizons, roi_mask=roi_mask
        )

        roi_map_mm, predicted_volumes, instant_predicted_volumes = Evaluator.calculate_volumes(
            rain_rate, float_preds, roi_mask, geom.pixel_area_km2, horizons,
            getattr(geom, 'roi_mask_fractional', None)
        )
        
        # Apply recent error feedback loop
        advection_engine.update_feedback(
            actual_map=roi_map_mm,
            preds={}
        )
        predicted_volumes = advection_engine.correct_cumulative_volumes(predicted_volumes)
        advection_engine.record_current_forecast(predicted_volumes)

        valid_errors = [c.prediction_error_pixels for c in tracked_cells if c.is_tracked]
        size_errors = [c.size_error_percent for c in tracked_cells if c.is_tracked]

        rain_rate_masked = np.ma.masked_where(rain_rate < RAIN_THRESHOLD_MIN, rain_rate)
        
        # Convert domain objects to serializable dictionaries for the UI
        tracked_cells_dicts = [c.as_dict() for c in tracked_cells]
        
        roi_mask = getattr(geom, 'roi_mask_fractional', None)
        if roi_mask is not None:
            max_rain_lm2 = float(np.max(rain_rate * (roi_mask > 0))) * 0.25
        else:
            max_rain_lm2 = float(np.max(rain_rate)) * 0.25
        
        
        return FrameResult(
            tracked_cells=tracked_cells_dicts,
            raw_tracked_cells=tracked_cells,
            rain_rate=rain_rate,
            rain_rate_masked=rain_rate_masked,
            lon_grid=geom.lon_grid,
            lat_grid=geom.lat_grid,
            max_rain=max_rain_lm2,
            mean_centroid_error=float(np.mean(valid_errors)) if valid_errors else 0.0,
            mean_size_error=float(np.mean(size_errors)) if size_errors else 0.0,
            num_tracked=sum(1 for c in tracked_cells if c.is_tracked),
            roi_map_mm=roi_map_mm,
            predicted_roi_map_mm=predicted_volumes.get("1h", 0.0),
            predicted_volumes_horizons=predicted_volumes,
            instant_predicted_volumes=instant_predicted_volumes,
        )
