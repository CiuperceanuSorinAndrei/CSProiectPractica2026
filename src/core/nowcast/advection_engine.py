# Lagrangian Advection Engine.
from __future__ import annotations

from collections import deque

import numpy as np


from src.core.constants import HORIZON_STEPS
from src.core.domain import StormCell
from src.core.nowcast.kinematic_advector import KinematicAdvector

class AdvectionEngine:
    _MIN_FEEDBACK_MM = 0.02
    _BIAS_MIN = 0.45
    _BIAS_MAX = 1.60
    _BIAS_ALPHA_UP = 0.14
    _BIAS_ALPHA_DOWN = 0.30
    _DRY_DECAY = 0.003
    _WINDOW_SIZE = 9
    _DRY_GUARD_RECENT_MM = 0.03
    _DRY_GUARD_PRED_MAX = 0.20

    def __init__(
        self,
        kinematic_advector: KinematicAdvector,
    ) -> None:
        self.kinematic_advector = kinematic_advector
        self.reset_feedback()

    def reset_feedback(self) -> None:
        # Reset calibration.
        self.dynamic_bias_correction: float = 1.0
        self._pid_bias: float = 1.0
        self._error_history: list[dict] = []
        self._bias_by_step = {step: 1.0 for step in HORIZON_STEPS.values()}
        self._ratio_windows = {
            step: deque(maxlen=self._WINDOW_SIZE)
            for step in HORIZON_STEPS.values()
        }

    def update_feedback(self, actual_map: float, preds: dict) -> None:
        # Update calibration from mature forecasts.
        self._error_history.append({
            "actual": float(actual_map or 0.0),
            "preds": dict(preds or {})
        })

        max_lag = max(HORIZON_STEPS.values())
        while len(self._error_history) > max_lag + 1:
            self._error_history.pop(0)

        for horizon, step in HORIZON_STEPS.items():
            if len(self._error_history) <= step:
                continue
            past = self._error_history[-1 - step]
            pred = float(past["preds"].get(horizon, 0.0) or 0.0)
            actual = sum(item["actual"] for item in self._error_history[-step:])
            self._update_step_bias(step, pred, actual)

    def correct_cumulative_volumes(self, volumes: dict[str, float]) -> dict[str, float]:
        return {
            horizon: float(volumes.get(horizon, 0.0) or 0.0) * self._bias_by_step.get(step, 1.0)
            for horizon, step in HORIZON_STEPS.items()
        }

    def record_current_forecast(self, preds: dict) -> None:
        if self._error_history:
            self._error_history[-1]["preds"] = dict(preds or {})

    def _update_step_bias(self, step: int, pred: float, actual: float) -> None:
        if pred < self._MIN_FEEDBACK_MM and actual < self._MIN_FEEDBACK_MM:
            self._decay_step_bias(step)
            return
        if pred < self._MIN_FEEDBACK_MM:
            ratio = self._BIAS_MAX
        elif actual < self._MIN_FEEDBACK_MM:
            ratio = self._BIAS_MIN
        else:
            ratio = actual / pred
        log_ratio = float(np.clip(
            np.log(max(ratio, 1e-6)),
            np.log(self._BIAS_MIN),
            np.log(self._BIAS_MAX),
        ))
        self._ratio_windows[step].append(log_ratio)
        target = float(np.exp(np.median(self._ratio_windows[step])))
        current = self._bias_by_step[step]
        alpha_up = 0.081 if step == HORIZON_STEPS["2h"] else self._BIAS_ALPHA_UP
        alpha = self._BIAS_ALPHA_DOWN if target < current else alpha_up
        self._bias_by_step[step] = float(np.clip(
            (1.0 - alpha) * current + alpha * target,
            self._BIAS_MIN,
            self._BIAS_MAX,
        ))

    def _decay_step_bias(self, step: int) -> None:
        current = self._bias_by_step[step]
        self._bias_by_step[step] = current + (1.0 - current) * self._DRY_DECAY

    @staticmethod
    def _weighted_median(values: list[float], weights: list[float]) -> float:
        values_arr = np.asarray(values, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        total = float(np.sum(weights_arr))
        if total <= 0.0:
            return float(np.median(values_arr)) if len(values_arr) else 0.0
        order = np.argsort(values_arr)
        sorted_values = values_arr[order]
        cumulative = np.cumsum(weights_arr[order])
        return float(sorted_values[np.searchsorted(cumulative, 0.5 * total, side="left")])

    @staticmethod
    def _roi_center_and_scale(shape: tuple[int, int], roi_mask: np.ndarray | None) -> tuple[float, float, float]:
        if roi_mask is not None:
            y_indices, x_indices = np.where(roi_mask > 0)
            if len(y_indices) > 0:
                cy = float(np.mean(y_indices))
                cx = float(np.mean(x_indices))
                height = float(np.ptp(y_indices) + 1)
                width = float(np.ptp(x_indices) + 1)
                return cy, cx, max((height * width) ** 0.5, 1.0)
        return shape[0] / 2.0, shape[1] / 2.0, max((shape[0] * shape[1]) ** 0.5, 1.0)

    @staticmethod
    def _centroid_confidence(cells: list[StormCell], roi_scale: float) -> float:
        if not cells:
            return 0.0
        centroid_errors = [c.prediction_error_pixels for c in cells]
        size_errors = [abs(c.size_error_percent) for c in cells]
        centroid_penalty = np.median(centroid_errors) / max(roi_scale * 0.35, 1.0)
        size_penalty = np.median(size_errors) / 100.0
        return float(np.clip(1.0 - max(centroid_penalty, size_penalty), 0.0, 1.0))

    @staticmethod
    def _mass_weighted_velocity(cells: list[StormCell]) -> tuple[float, float]:
        if not cells:
            return 0.0, 0.0
        weights = [max(c.volume, 1e-6) for c in cells]
        return (
            float(np.average([c.v_x for c in cells], weights=weights)),
            float(np.average([c.v_y for c in cells], weights=weights)),
        )

    def _recent_actual_is_dry(self) -> bool:
        if not self._error_history:
            return False
        recent = self._error_history[-min(3, len(self._error_history)):]
        return float(np.mean([item["actual"] for item in recent])) < self._DRY_GUARD_RECENT_MM

    def _velocity_for_step(
        self,
        cells: list[StormCell],
        step: int,
        roi_center: tuple[float, float],
        roi_scale: float,
    ) -> tuple[float, float, list[StormCell], list[float]]:
        cy, cx = roi_center
        weights = []
        for c in cells:
            cell_x = c.centroid_x
            cell_y = c.centroid_y
            vel_x = c.v_x
            vel_y = c.v_y
            pred_x = c.centroid_x + c.v_x * step
            pred_y = c.centroid_y + c.v_y * step
            dist = ((pred_x - cx) ** 2 + (pred_y - cy) ** 2) ** 0.5
            proximity = 1.0 / (1.0 + (dist / roi_scale) ** 2)
            to_roi_x = cx - cell_x
            to_roi_y = cy - cell_y
            to_roi_len = max((to_roi_x ** 2 + to_roi_y ** 2) ** 0.5, 1e-6)
            vel_len = max((vel_x ** 2 + vel_y ** 2) ** 0.5, 1e-6)
            direction = (vel_x * to_roi_x + vel_y * to_roi_y) / (vel_len * to_roi_len)
            direction_weight = 0.15 if direction < 0.0 else 0.5 + 0.5 * direction
            mass = max(c.volume, 1e-6)
            weights.append(mass * proximity * direction_weight)

        vx = self._weighted_median([c.v_x for c in cells], weights) if cells else 0.0
        vy = self._weighted_median([c.v_y for c in cells], weights) if cells else 0.0
        return vx, vy, cells, weights

    def extrapolate(
        self,
        rain_rate: np.ndarray,
        tracked_cells: list[StormCell],
        horizons: list[tuple[int, str]],
        roi_mask: np.ndarray = None,
    ) -> dict[int, np.ndarray]:
        # Extrapolate precipitation.
        rain_rate = np.nan_to_num(rain_rate, copy=True, nan=0.0).astype(np.float32, copy=False)
        
        float_preds = {}
        
        max_step = max(h[0] for h in horizons) if horizons else 0
        
        valid_cells = [c for c in tracked_cells if c.is_tracked]
        simulated_cells = [c.clone() for c in valid_cells]
        
        relevant_cells = []
        relevant_weights = []
        if simulated_cells:
            cy, cx, roi_scale = self._roi_center_and_scale(rain_rate.shape, roi_mask)
            _, _, relevant_cells, relevant_weights = self._velocity_for_step(
                simulated_cells, 1, (cy, cx), roi_scale
            )
        else:
            cy, cx, roi_scale = self._roi_center_and_scale(rain_rate.shape, roi_mask)
        centroid_confidence = self._centroid_confidence(relevant_cells, roi_scale)
        
        mean_E = np.average(
            [max(c.volume, 1e-6) for c in relevant_cells],
            weights=relevant_weights,
        ) if relevant_cells else 1.0
        
        mean_dE = np.average([
            c.volume - (c.volume / max(c.volume_trend, 1e-5)) 
            for c in relevant_cells
        ], weights=relevant_weights) if relevant_cells else 0.0
        
        current_E = float(mean_E)
        current_dE = float(mean_dE)
        
        from src.core.nowcast.reaction_diffusion import update_energy
        
        thermo_multiplier = 1.0
        
        for step in range(1, max_step + 1):
            roi_vx, roi_vy, step_cells, step_weights = self._velocity_for_step(
                simulated_cells, step, (cy, cx), roi_scale
            )
            mass_vx, mass_vy = self._mass_weighted_velocity(step_cells)
            count_confidence = min(len(step_cells) / 3.0, 1.0)
            roi_confidence = float(np.clip(np.mean(step_weights), 0.0, 1.0)) if step_weights else 0.0
            tracking_confidence = centroid_confidence * (0.35 + 0.35 * count_confidence + 0.30 * roi_confidence)
            
            # Uniform advection.
            shift_y = step * roi_vy
            shift_x = step * roi_vx
            
            shifted_raw = self.kinematic_advector.advect(
                rain_rate, shift_y, shift_x
            )
            mass_shifted = self.kinematic_advector.advect(
                rain_rate, step * mass_vy, step * mass_vx
            )
            damped_shifted = 0.90 * self.kinematic_advector.advect(
                rain_rate, step * roi_vy * 0.50, step * roi_vx * 0.50
            )
            
            # Blend advected components.
            damped_weight = 0.35 * (1.0 - tracking_confidence)
            mass_weight = 0.25 * tracking_confidence
            roi_weight = max(1.0 - damped_weight - mass_weight, 0.0)
            total_weight = roi_weight + mass_weight + damped_weight
            shifted = (
                shifted_raw * roi_weight
                + mass_shifted * mass_weight
                + damped_shifted * damped_weight
            ) / max(total_weight, 1e-6)
            
            # Correct numerical diffusion/boundary loss before thermodynamics.
            orig_mass = float(np.nansum(rain_rate))
            shifted_mass = float(np.nansum(shifted))
            if orig_mass > 0.01 and shifted_mass > 0.01:
                mass_ratio = np.clip(orig_mass / shifted_mass, 0.90, 1.10)
                shifted = shifted * mass_ratio
                
            # Simulate organic thermodynamic step.
            current_E, current_dE, R_step = update_energy(current_E, np.array([]), current_dE)
            thermo_multiplier *= R_step
            
            # Apply thermodynamics.
            shifted = shifted * thermo_multiplier

            if (
                tracking_confidence < 0.35
                and self._recent_actual_is_dry()
                and float(np.nanmean(shifted)) < self._DRY_GUARD_PRED_MAX
            ):
                shifted = shifted * 0.35
            
            float_preds[step] = shifted
            
        return float_preds
