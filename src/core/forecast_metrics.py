"""Metrici de validare a nowcasting-ului: CSI, FAR, POD si erori de centroid/arie."""
from __future__ import annotations

import numpy as np


class ForecastMetrics:

    @staticmethod
    def csi(observed: np.ndarray, predicted: np.ndarray) -> float:
        """Critical Success Index (Threat Score)."""
        obs = observed.astype(bool)
        pred = predicted.astype(bool)
        hits = np.logical_and(obs, pred).sum()
        misses = np.logical_and(obs, ~pred).sum()
        false_alarms = np.logical_and(~obs, pred).sum()
        denominator = hits + misses + false_alarms
        return float(hits / denominator) if denominator else 0.0

    @staticmethod
    def far(observed: np.ndarray, predicted: np.ndarray) -> float:
        """False Alarm Ratio."""
        obs = observed.astype(bool)
        pred = predicted.astype(bool)
        hits = np.logical_and(obs, pred).sum()
        false_alarms = np.logical_and(~obs, pred).sum()
        denominator = hits + false_alarms
        return float(false_alarms / denominator) if denominator else 0.0

    @staticmethod
    def pod(observed: np.ndarray, predicted: np.ndarray) -> float:
        """Probability of Detection (Hit Rate)."""
        obs = observed.astype(bool)
        pred = predicted.astype(bool)
        hits = np.logical_and(obs, pred).sum()
        misses = np.logical_and(obs, ~pred).sum()
        denominator = hits + misses
        return float(hits / denominator) if denominator else 0.0

    @staticmethod
    def centroid_mae(
        observed_centroids: list[tuple[float, float]],
        predicted_centroids: list[tuple[float, float]],
    ) -> float:
        """Mean Absolute Error intre centroizii observati si cei prezisi."""
        if not observed_centroids or not predicted_centroids:
            return 0.0
        paired = min(len(observed_centroids), len(predicted_centroids))
        errors = []
        for idx in range(paired):
            oy, ox = observed_centroids[idx]
            py, px = predicted_centroids[idx]
            errors.append(float(abs(px - ox) + abs(py - oy)))
        return float(np.mean(errors)) if errors else 0.0

    @staticmethod
    def area_error(observed_area: float, predicted_area: float) -> dict[str, float]:
        """Eroarea absoluta si procentuala intre ariile observata si prezisa."""
        observed_area = max(float(observed_area), 1.0)
        predicted_area = max(float(predicted_area), 0.0)
        abs_error = abs(predicted_area - observed_area)
        pct_error = 100.0 * abs_error / observed_area
        return {"absolute": abs_error, "percent": pct_error}
