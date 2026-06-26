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
        float_preds: dict[str, np.ndarray],
        roi_mask: np.ndarray,
        lat_grid: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        """Calculeaza volumul curent in ROI si volumele prezise."""
        # Aproximare grosiera a ariei: unghiul de vizualizare oblic la poli dilata fizic pixelul
        pixel_area_km2 = 3.0 * (3.0 / np.cos(np.radians(lat_grid)))
        
        # Volumul curent observat
        roi_volume_m3 = float(np.sum(rain_rate[roi_mask] * pixel_area_km2[roi_mask] * 250.0))
        
        # Volumul prezis pe fiecare orizont
        predicted_volumes = {}
        for name, pred_matrix in float_preds.items():
            vol = float(np.sum(pred_matrix[roi_mask] * pixel_area_km2[roi_mask] * 250.0))
            predicted_volumes[name] = vol
            
        return roi_volume_m3, predicted_volumes
