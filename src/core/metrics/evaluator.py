"""Evaluator pentru metrici globale si estimari de volum."""
from __future__ import annotations

import numpy as np

from config import RAIN_THRESHOLD_MIN, RAIN_THRESHOLD_TRACKING
from src.core.metrics.forecast_metrics import ForecastMetrics


class Evaluator:
    @staticmethod
    def calculate_global_metrics(
        rain_rate: np.ndarray,
        roi_mask: np.ndarray,
        predictions_queue: list[tuple[dict, dict]],
        horizons: list[tuple[int, str]]
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        """Calculeaza CSI, FAR, POD si FSS pentru orizonturile specificate."""
        global_csi, global_far, global_pod, global_fss = {}, {}, {}, {}
        obs_mask = (rain_rate >= RAIN_THRESHOLD_TRACKING) & roi_mask

        for steps_back, name in horizons:
            if len(predictions_queue) >= steps_back:
                past_pred_sparse = predictions_queue[-steps_back][0].get(name)
                if past_pred_sparse is not None:
                    past_pred = past_pred_sparse.toarray()
                    if past_pred.shape == rain_rate.shape:
                        pred_mask = (past_pred >= RAIN_THRESHOLD_TRACKING) & roi_mask
                        if np.any(obs_mask) or np.any(pred_mask):
                            global_csi[name] = ForecastMetrics.csi(obs_mask, pred_mask)
                            global_far[name] = ForecastMetrics.far(obs_mask, pred_mask)
                            global_pod[name] = ForecastMetrics.pod(obs_mask, pred_mask)
                            global_fss[name] = ForecastMetrics.fss(obs_mask, pred_mask, window_size=5)
                            
        return global_csi, global_far, global_pod, global_fss

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
        
        conversion_factor = 250.0
        frac_mask = roi_mask_fractional if roi_mask_fractional is not None else roi_mask.astype(np.float32)
        
        # V22: Volumul Meteorologic Semnificativ
        rain_rate_filtered = np.where(rain_rate >= RAIN_THRESHOLD_MIN, rain_rate, 0.0)
        roi_volume_m3 = float(np.nansum(rain_rate_filtered * pixel_area_km2 * frac_mask * conversion_factor))
        
        # Calculam volumul estimat pentru fiecare sfert de ora din viitor
        step_volumes = {}
        for step, pred_matrix in float_preds.items():
            pred_filtered = np.where(pred_matrix >= RAIN_THRESHOLD_MIN, pred_matrix, 0.0)
            step_volumes[step] = float(np.nansum(pred_filtered * pixel_area_km2 * frac_mask * conversion_factor))
            
        # Acumulam volumul pentru fiecare orizont (Ex: 1h = step 1 + step 2 + step 3 + step 4)
        predicted_volumes_accumulation = {}
        instant_predicted_volumes = {}
        for target_step, name in horizons:
            accumulated_vol = 0.0
            for step in range(1, target_step + 1):
                accumulated_vol += step_volumes.get(step, 0.0)
            predicted_volumes_accumulation[name] = accumulated_vol
            instant_predicted_volumes[name] = step_volumes.get(target_step, 0.0)
            
        return roi_volume_m3, predicted_volumes_accumulation, instant_predicted_volumes
