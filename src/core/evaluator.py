"""Evaluator pentru metrici globale si estimari de volum."""
from __future__ import annotations

import numpy as np

from config import RAIN_THRESHOLD_MIN
from src.core.forecast_metrics import ForecastMetrics


class Evaluator:
    @staticmethod
    def calculate_global_metrics(
        rain_rate: np.ndarray,
        roi_mask: np.ndarray,
        predictions_queue: list[dict],
        horizons: list[tuple[int, str]]
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        """Calculeaza CSI, FAR, POD si FSS pentru orizonturile specificate."""
        global_csi, global_far, global_pod, global_fss = {}, {}, {}, {}
        obs_mask = (rain_rate >= RAIN_THRESHOLD_MIN) & roi_mask

        for steps_back, name in horizons:
            if len(predictions_queue) >= steps_back:
                past_pred_sparse = predictions_queue[-steps_back].get(name)
                if past_pred_sparse is not None:
                    past_pred = past_pred_sparse.toarray()
                    if past_pred.shape == rain_rate.shape:
                        pred_mask = (past_pred >= RAIN_THRESHOLD_MIN) & roi_mask
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
        lat_grid: np.ndarray,
        horizons: list[tuple[int, str]]
    ) -> tuple[float, dict[str, float]]:
        """Calculeaza volumul curent in ROI si volumul acumulat pe orizonturi.
        
        Suma volumelor pentru un orizont include toate sferturile de ora de la
        momentul T0 pana la momentul orizontului respectiv (integral discret).
        """
        # Aproximare grosiera a ariei: unghiul de vizualizare oblic la poli dilata fizic pixelul
        pixel_area_km2 = 3.0 * (3.0 / np.cos(np.radians(lat_grid)))
        
        # O valoare de 250.0 converteste rata mm/h in metri cubi pentru 1 sfert de ora pe 1 km2
        # (1 mm/h * 0.25 h * 1,000,000 m2 / 1000 = 250 m3)
        conversion_factor = 250.0
        
        # Volumul aportat curent
        roi_volume_m3 = float(np.sum(rain_rate[roi_mask] * pixel_area_km2[roi_mask] * conversion_factor))
        
        # Calculam volumul estimat pentru fiecare sfert de ora din viitor
        step_volumes = {}
        for step, pred_matrix in float_preds.items():
            step_volumes[step] = float(np.sum(pred_matrix[roi_mask] * pixel_area_km2[roi_mask] * conversion_factor))
            
        # Acumulam volumul pentru fiecare orizont (Ex: 1h = step 1 + step 2 + step 3 + step 4)
        predicted_volumes_accumulation = {}
        for target_step, name in horizons:
            accumulated_vol = 0.0
            for step in range(1, target_step + 1):
                accumulated_vol += step_volumes.get(step, 0.0)
            predicted_volumes_accumulation[name] = accumulated_vol
            
        return roi_volume_m3, predicted_volumes_accumulation
