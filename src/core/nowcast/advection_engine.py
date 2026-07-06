"""Motor de advectie Rigid (Lagrangian Persistence)."""
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
    _BIAS_ALPHA = 0.18
    _DRY_DECAY = 0.003
    _WINDOW_SIZE = 9

    def __init__(
        self,
        kinematic_advector: KinematicAdvector,
    ) -> None:
        self.kinematic_advector = kinematic_advector
        self.reset_feedback()

    def reset_feedback(self) -> None:
        """Reset online calibration state to neutral."""
        self.dynamic_bias_correction: float = 1.0
        self._pid_bias: float = 1.0
        self._error_history: list[dict] = []
        self._bias_by_step = {step: 1.0 for step in HORIZON_STEPS.values()}
        self._ratio_windows = {
            step: deque(maxlen=self._WINDOW_SIZE)
            for step in HORIZON_STEPS.values()
        }
        self._sync_legacy_biases()

    def update_feedback(self, actual_map: float, preds: dict) -> None:
        """Update calibration from issued cumulative forecasts after they mature."""
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

        self._sync_legacy_biases()

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
        self._bias_by_step[step] = float(np.clip(
            (1.0 - self._BIAS_ALPHA) * current + self._BIAS_ALPHA * target,
            self._BIAS_MIN,
            self._BIAS_MAX,
        ))

    def _decay_step_bias(self, step: int) -> None:
        current = self._bias_by_step[step]
        self._bias_by_step[step] = current + (1.0 - current) * self._DRY_DECAY

    def _sync_legacy_biases(self) -> None:
        self._bias_15m = self._bias_by_step[HORIZON_STEPS["15m"]]
        self._bias_1h = self._bias_by_step[HORIZON_STEPS["1h"]]
        self._bias_2h = self._bias_by_step[HORIZON_STEPS["2h"]]
        self.dynamic_bias_correction = 1.0
        self._pid_bias = 1.0

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
            return 1.0
        centroid_errors = [float(getattr(c, "prediction_error_pixels", 0.0) or 0.0) for c in cells]
        size_errors = [abs(float(getattr(c, "size_error_percent", 0.0) or 0.0)) for c in cells]
        centroid_penalty = np.median(centroid_errors) / max(roi_scale * 0.35, 1.0)
        size_penalty = np.median(size_errors) / 100.0
        return float(np.clip(1.0 - max(centroid_penalty, size_penalty), 0.0, 1.0))

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
            cell_x = float(getattr(c, "centroid_x", 0.0))
            cell_y = float(getattr(c, "centroid_y", 0.0))
            vel_x = float(getattr(c, "v_x", 0.0))
            vel_y = float(getattr(c, "v_y", 0.0))
            pred_x = float(getattr(c, "centroid_x", 0.0) + getattr(c, "v_x", 0.0) * step)
            pred_y = float(getattr(c, "centroid_y", 0.0) + getattr(c, "v_y", 0.0) * step)
            dist = ((pred_x - cx) ** 2 + (pred_y - cy) ** 2) ** 0.5
            proximity = 1.0 / (1.0 + (dist / roi_scale) ** 2)
            to_roi_x = cx - cell_x
            to_roi_y = cy - cell_y
            to_roi_len = max((to_roi_x ** 2 + to_roi_y ** 2) ** 0.5, 1e-6)
            vel_len = max((vel_x ** 2 + vel_y ** 2) ** 0.5, 1e-6)
            direction = (vel_x * to_roi_x + vel_y * to_roi_y) / (vel_len * to_roi_len)
            direction_weight = 0.15 if direction < 0.0 else 0.5 + 0.5 * direction
            mass = max(float(getattr(c, "volume", 0.0) or 0.0), 1e-6)
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
        """Extrapoleaza precipitatiile (Hydrological Catchment Nowcasting)."""
        rain_rate = np.nan_to_num(rain_rate, nan=0.0).astype(np.float32)
        
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
        
        # 2. Extragem starea termodinamica curenta folosind Volumele REALE (masa reala de apa)
        # Celulele sunt ponderate dupa masa, distanta si directia catre ROI.
        mean_E = np.average(
            [max(getattr(c, 'volume', 1.0), 1e-6) for c in relevant_cells],
            weights=relevant_weights,
        ) if relevant_cells else 1.0
        
        # dE real = Volumul curent - Volumul precedent (pe baza la volume_trend)
        mean_dE = np.average([
            getattr(c, 'volume', 0.0) - (getattr(c, 'volume', 0.0) / max(getattr(c, 'volume_trend', 1.0), 1e-5)) 
            for c in relevant_cells
        ], weights=relevant_weights) if relevant_cells else 0.0
        
        current_E = float(mean_E)
        current_dE = float(mean_dE)
        
        from src.core.nowcast.reaction_diffusion import update_energy
        
        thermo_multiplier = 1.0
        
        for step in range(1, max_step + 1):
            global_vx, global_vy, _, _ = self._velocity_for_step(
                simulated_cells, step, (cy, cx), roi_scale
            )
            
            # Advectie uniforma prin shiftare scipy
            shift_y = step * global_vy
            shift_x = step * global_vx
            
            shifted_raw = self.kinematic_advector.advect(
                rain_rate, shift_y, shift_x
            )
            
            # Masa aflata efectiv pe ecran (include scaderea naturala la margini)
            
            # Lăsăm pixelii nealterați. Pragul va fi aplicat corect doar de Evaluator, 
            # DUPĂ ce PID-ul și Termodinamica au scalat complet furtuna. 
            # Aceasta previne Efectul de Clichet Asimetric (pierderea sistematică de masă la margini).
            lead_weight = (step - 1) / max(max_step - 1, 1)
            persistence_blend = 0.60 * (1.0 - centroid_confidence) * lead_weight
            damped_persistence = np.minimum(shifted_raw, rain_rate)
            shifted = shifted_raw * (1.0 - persistence_blend) + damped_persistence * persistence_blend
            
            # 1. Conservarea Masei a fost stearsa (Nu dorim compensarea ploilor usoare sterse).
            # Ploaia stearsa la prag ramane stearsa, evitand inflatia artificiala a nucleelor severe.
            
            # 2. Simulăm pasul termodinamic organic (care are limite fizice absolute in reaction_diffusion.py)
            current_E, current_dE, R_step = update_energy(current_E, np.array([]), current_dE)
            thermo_multiplier *= R_step
            
            # Bias-ul online corecteaza doar volumele cumulate afisate, nu campul fizic advectat.
            shifted = shifted * thermo_multiplier
            
            float_preds[step] = shifted
            
            # Pentru compatibilitatea codului (nu le mai folosim în raport)
        return float_preds
