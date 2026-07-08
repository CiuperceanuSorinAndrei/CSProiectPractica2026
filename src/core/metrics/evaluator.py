# Global metrics and volume estimator.
from __future__ import annotations

import numpy as np

from src.config import RAIN_THRESHOLD_MIN


class Evaluator:
    @staticmethod
    def calculate_volumes(
        rain_rate: np.ndarray,
        float_preds: dict[int, np.ndarray],
        roi_mask: np.ndarray,
        pixel_area_km2: np.ndarray,
        horizons: list[tuple[int, str]],
        roi_mask_fractional: np.ndarray = None
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        # Calculate MAP in ROI and accumulated MAP per horizon.
        
        frac_mask = roi_mask_fractional if roi_mask_fractional is not None else roi_mask.astype(np.float32)
        
        # Precompute area weights for 15-minute accumulation (0.25 hours)
        weights = pixel_area_km2 * frac_mask * 0.25
        
        # Convert m3 to Mean Areal Precipitation (MAP) in mm.
        area_km2 = float(np.nansum(pixel_area_km2 * frac_mask))
        if area_km2 < 1e-6:
            return 0.0, {}, {}
            
        valid_rain = rain_rate >= RAIN_THRESHOLD_MIN
        roi_map_mm = float(np.nansum(rain_rate[valid_rain] * weights[valid_rain]) / area_km2)
        
        # Estimate MAP for future 15-minute steps.
        step_volumes = {}
        for step, pred_matrix in float_preds.items():
            valid_pred = pred_matrix >= RAIN_THRESHOLD_MIN
            step_volumes[step] = float(np.nansum(pred_matrix[valid_pred] * weights[valid_pred]) / area_km2)
            
        # Accumulate MAP for each horizon.
        predicted_volumes_accumulation = {}
        instant_predicted_volumes = {}
        for target_step, name in horizons:
            accumulated_vol = 0.0
            for step in range(1, target_step + 1):
                accumulated_vol += step_volumes.get(step, 0.0)
            predicted_volumes_accumulation[name] = accumulated_vol
            instant_predicted_volumes[name] = step_volumes.get(target_step, 0.0)
            
        return roi_map_mm, predicted_volumes_accumulation, instant_predicted_volumes
