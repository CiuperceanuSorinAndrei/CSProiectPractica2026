"""Motor de advectie Rigid (Lagrangian Persistence)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.core.domain import StormCell
from src.core.nowcast.kinematics import KinematicsEngine
from src.core.nowcast.kinematic_advector import KinematicAdvector

from config import RAIN_THRESHOLD_TRACKING, RAIN_THRESHOLD_MIN

class AdvectionEngine:
    def __init__(
        self,
        kinematic_advector: KinematicAdvector,
    ) -> None:
        self.kinematic_advector = kinematic_advector
        self._ema_trend: float = 1.0  # Exponential Moving Average al trendului global
        
        # State PID Feedback Control
        self.dynamic_bias_correction: float = 1.0
        self._pid_bias: float = 1.0
        self._error_history: list[dict] = []

    def update_feedback(self, actual_map: float, predicted_map: float) -> None:
        """Encapsulated PID feedback update to keep AdvectionEngine state internal."""
        self._error_history.append({
            "actual": actual_map,
            "predicted_15m": predicted_map
        })
        
        # Keep last 5 frames for T-2 calculation
        if len(self._error_history) > 5:
            self._error_history.pop(0)
            
        if len(self._error_history) >= 3:
            past_frame = self._error_history[-3]
            pred_for_now = past_frame["predicted_15m"]
            actual_now = actual_map
            
            if pred_for_now > 1.0 and actual_now > 1.0:
                raw_ratio = float(np.clip(actual_now / pred_for_now, 0.7, 1.4))
                self._pid_bias = 0.8 * self._pid_bias + 0.2 * raw_ratio
            else:
                self._pid_bias = 0.9 * self._pid_bias + 0.1 * 1.0
                
            self.dynamic_bias_correction = self._pid_bias

    def extrapolate(
        self,
        rain_rate: np.ndarray,
        tracked_cells: list[StormCell],
        horizons: list[tuple[int, str]],
    ) -> tuple[dict[str, sp.csr_matrix], dict[int, np.ndarray], dict[int, list[StormCell]]]:
        """Extrapoleaza precipitatiile (Hydrological Catchment Nowcasting)."""
        rain_rate = np.nan_to_num(rain_rate, nan=0.0).astype(np.float32)
        
        sparse_preds = {}
        float_preds = {}
        predicted_cells_dict = {}
        
        max_step = max(h[0] for h in horizons) if horizons else 0
        horizon_map = {h[0]: h[1] for h in horizons}
        
        valid_cells = [c for c in tracked_cells if c.is_tracked]
        simulated_cells = [c.clone() for c in valid_cells]
        
        # 1. Global Kinematics
        if simulated_cells:
            global_vx = float(np.median([c.v_x for c in simulated_cells]))
            global_vy = float(np.median([c.v_y for c in simulated_cells]))
        else:
            global_vx, global_vy = 0.0, 0.0
        
        # 2. EMA-Smoothed Volume Trend
        total_volume = sum(getattr(c, 'volume', 0.0) for c in simulated_cells)
        if total_volume > 0:
            raw_trend = sum(getattr(c, 'volume', 0.0) * getattr(c, 'volume_trend', 1.0) for c in simulated_cells) / total_volume
            # Clamp semnalul brut ÎNAINTE de EMA pentru a preveni contaminarea cu valori aberante
            raw_trend = float(np.clip(raw_trend, 0.7, 1.5))
            # EMA: 75% memorie, 25% semnal nou → convergență rapidă peste ~4 cadre
            self._ema_trend = 0.75 * self._ema_trend + 0.25 * raw_trend
        
        # Clamp pe EMA (trendul fizic brut)
        initial_trend = float(np.clip(self._ema_trend, 0.95, 1.12))
        
        # PID Feedback Loop: ajustam trendul fizic cu eroarea recenta a sistemului
        # dynamic_bias_correction este calculat pe o predictie de 2 pasi (15m).
        # Transformam factorul compus intr-unul per-pas: 1.20 compus -> ~1.10 per pas
        per_step_correction = 1.0 + (self.dynamic_bias_correction - 1.0) / 2.0
        corrected_trend = initial_trend * per_step_correction
        
        # Clamp generalizat de siguranta pe trendul final [0.90, 1.15]
        # (1.15 compus pe 9 pasi cu decay AR-1 da un maxim de ~1.8x, absolut sigur)
        corrected_trend = float(np.clip(corrected_trend, 0.90, 1.15))
        
        # AR-1 Reversion to Mean cu date curate
        current_trend = corrected_trend
        cumulative_multiplier = 1.0
        
        for step in range(1, max_step + 1):
            # Actualizare multiplicator cumulativ
            cumulative_multiplier *= current_trend
            
            # Advectie uniforma prin shiftare scipy
            shift_y = step * global_vy
            shift_x = step * global_vx
            
            shifted_raw = self.kinematic_advector.advect(
                rain_rate, shift_y, shift_x
            )
            
            # Masa aflata efectiv pe ecran (include scaderea naturala la margini)
            mass_in_domain = float(np.sum(shifted_raw))
            
            # Hard-Thresholding pentru zgomotul pur de interpolare
            hard_mask = (shifted_raw >= RAIN_THRESHOLD_MIN).astype(np.float32)
            shifted = shifted_raw * hard_mask
            
            # 1. Conservarea Masei "Safe" (reparăm STRICT ploaia pe care a șters-o threshold-ul)
            mass_after_thresh = float(np.sum(shifted))
            if mass_after_thresh > 0 and mass_in_domain > 0:
                # Cât la sută din ploaie a fost ștearsă de threshold?
                mass_correction = mass_in_domain / mass_after_thresh
                # Cap de siguranță [1.0, 1.08] (Redus de la 1.25 pt a preveni anti-disiparea)
                mass_correction = np.clip(mass_correction, 1.0, 1.08)
                shifted = shifted * mass_correction
            
            # 2. Aplicăm creșterea termodinamică a furtunii (EMA-AR1)
            shifted = shifted * cumulative_multiplier
            
            # Relaxare AR-1 Asimetrică
            if current_trend > 1.0:
                # Daca furtuna crește, o limităm agresiv spre 1.0 (decay 0.70)
                current_trend = 1.0 + (current_trend - 1.0) * 0.70
            else:
                # Daca furtuna moare, o lasam sa moară! (decay slab 0.95)
                current_trend = 1.0 + (current_trend - 1.0) * 0.95
            
            float_preds[step] = shifted
            
            # Pentru compatibilitatea codului (nu le mai folosim în raport)
            predicted_cells_dict[step] = []
            
            if step in horizon_map:
                name = horizon_map[step]
                base_mask = (shifted >= RAIN_THRESHOLD_TRACKING).astype(np.float32)
                sparse_preds[name] = sp.csr_matrix(base_mask)
            
        return sparse_preds, float_preds, predicted_cells_dict
