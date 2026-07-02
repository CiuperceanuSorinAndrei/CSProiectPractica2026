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
        self._cached_grids: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._ema_trend: float = 1.0  # Exponential Moving Average al trendului global

    def _get_grids(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if shape not in self._cached_grids:
            self._cached_grids[shape] = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        return self._cached_grids[shape]

    def extrapolate(
        self,
        rain_rate: np.ndarray,
        flow: np.ndarray | None, # pastrat in semnatura pentru compatibilitate cu apelantul
        tracked_cells: list[StormCell],
        horizons: list[tuple[int, str]],
    ) -> tuple[dict[str, sp.csr_matrix], dict[int, np.ndarray], dict[int, list[StormCell]]]:
        """Extrapoleaza precipitatiile (Hydrological Catchment Nowcasting)."""
        grid_h, grid_w = rain_rate.shape
        rain_rate = np.nan_to_num(rain_rate, nan=0.0).astype(np.float32)
        y_grid, x_grid = self._get_grids((grid_h, grid_w))
        
        sparse_preds = {}
        float_preds = {}
        predicted_cells_dict = {}
        
        max_step = max(h[0] for h in horizons) if horizons else 0
        horizon_map = {h[0]: h[1] for h in horizons}
        
        valid_cells = [c for c in tracked_cells if c.is_tracked]
        
        map_x = x_grid.copy()
        map_y = y_grid.copy()
        
        simulated_cells = [c.clone() for c in valid_cells]
        
        # 1. Global Kinematics
        if simulated_cells:
            global_vx = float(np.median([c.v_x for c in simulated_cells]))
            global_vy = float(np.median([c.v_y for c in simulated_cells]))
        else:
            global_vx, global_vy = 0.0, 0.0
            
        flow_x = np.full((grid_h, grid_w), global_vx, dtype=np.float32)
        flow_y = np.full((grid_h, grid_w), global_vy, dtype=np.float32)
        
        # 2. EMA-Smoothed Volume Trend
        total_volume = sum(getattr(c, 'volume', 0.0) for c in simulated_cells)
        if total_volume > 0:
            raw_trend = sum(getattr(c, 'volume', 0.0) * getattr(c, 'volume_trend', 1.0) for c in simulated_cells) / total_volume
            # Clamp semnalul brut ÎNAINTE de EMA pentru a preveni contaminarea cu valori aberante
            raw_trend = float(np.clip(raw_trend, 0.7, 1.5))
            # EMA: 75% memorie, 25% semnal nou → convergență rapidă peste ~4 cadre
            self._ema_trend = 0.75 * self._ema_trend + 0.25 * raw_trend
        
        # Clamp final pe EMA (1.12 permite corecție suficientă la orizonturi scurte)
        initial_trend = float(np.clip(self._ema_trend, 0.95, 1.12))
        
        # AR-1 Reversion to Mean cu date curate
        current_trend = initial_trend
        cumulative_multiplier = 1.0
        
        # Masa initiala a ploii (pentru conservare)
        base_mass = float(np.sum(rain_rate))
            
        for step in range(1, max_step + 1):
            # Actualizare multiplicator cumulativ
            cumulative_multiplier *= current_trend
            
            # Advectie Semilagrangiana pura folosind Global Flow
            shifted, map_x, map_y = self.kinematic_advector.advect(
                rain_rate, map_x, map_y, x_grid, y_grid, flow_x, flow_y
            )
            
            # Hard-Thresholding pentru zgomotul pur de interpolare
            hard_mask = (shifted >= RAIN_THRESHOLD_MIN).astype(np.float32)
            shifted = shifted * hard_mask
            
            # 1. Conservarea Masei (reparăm scurgerea numerică cauzată de interpolare și thresholding)
            current_mass = float(np.sum(shifted))
            if current_mass > 0 and base_mass > 0:
                mass_correction = base_mass / current_mass
                shifted = shifted * mass_correction
            
            # 2. Aplicăm creșterea termodinamică a furtunii (EMA-AR1)
            shifted = shifted * cumulative_multiplier
            
            # Relaxare AR-1: 30% decay per pas înapoi spre 1.0
            current_trend = 1.0 + (current_trend - 1.0) * 0.70
            
            float_preds[step] = shifted
            
            # Pentru compatibilitatea codului (nu le mai folosim în raport)
            predicted_cells_dict[step] = []
            
            if step in horizon_map:
                name = horizon_map[step]
                base_mask = (shifted >= RAIN_THRESHOLD_TRACKING).astype(np.float32)
                sparse_preds[name] = sp.csr_matrix(base_mask)
            
        return sparse_preds, float_preds, predicted_cells_dict
