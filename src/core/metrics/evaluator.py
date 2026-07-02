"""Evaluator pentru metrici globale si estimari de volum."""
from __future__ import annotations

import numpy as np

from config import RAIN_THRESHOLD_MIN, RAIN_THRESHOLD_TRACKING


class Evaluator:
    # Funcția de calcul metrici globale a fost ștearsă (Hydrological Nowcasting Pivot)

    @staticmethod
    def calculate_volumes(
        rain_rate: np.ndarray,
        float_preds: dict[int, np.ndarray],
        roi_mask: np.ndarray,
        pixel_area_km2: np.ndarray,
        horizons: list[tuple[int, str]],
        roi_mask_fractional: np.ndarray = None
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Calculeaza volumul curent in ROI si volumul acumulat pe orizonturi."""
        
        frac_mask = roi_mask_fractional if roi_mask_fractional is not None else roi_mask.astype(np.float32)
        
        # ponytail: Hydrological conversion from absolute m3 to Mean Areal Precipitation (MAP) in L/m2 (mm)
        area_km2 = float(np.nansum(pixel_area_km2 * frac_mask))
        if area_km2 < 1e-6:
            return 0.0, {}, {}
            
        rain_rate_filtered = np.where(rain_rate >= RAIN_THRESHOLD_MIN, rain_rate, 0.0)
        # MAP = average rain depth over the area in 15 mins (rain_rate * 0.25)
        roi_map_mm = float(np.nansum(rain_rate_filtered * pixel_area_km2 * frac_mask * 0.25) / area_km2)
        
        # Calculam volumul estimat pentru fiecare sfert de ora din viitor
        step_volumes = {}
        for step, pred_matrix in float_preds.items():
            pred_filtered = np.where(pred_matrix >= RAIN_THRESHOLD_MIN, pred_matrix, 0.0)
            step_volumes[step] = float(np.nansum(pred_filtered * pixel_area_km2 * frac_mask * 0.25) / area_km2)
            
        # Acumulam volumul pentru fiecare orizont (Ex: 1h = step 1 + step 2 + step 3 + step 4)
        predicted_volumes_accumulation = {}
        instant_predicted_volumes = {}
        for target_step, name in horizons:
            accumulated_vol = 0.0
            for step in range(1, target_step + 1):
                accumulated_vol += step_volumes.get(step, 0.0)
            predicted_volumes_accumulation[name] = accumulated_vol
            instant_predicted_volumes[name] = step_volumes.get(target_step, 0.0)
            
        return roi_map_mm, predicted_volumes_accumulation, instant_predicted_volumes
