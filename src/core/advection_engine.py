"""Motor de advectie Hibrid (Linear Jump + Spatial Growth + Directional Blur).

Aplica advectia semi-Lagrangiana liniara dintr-un singur salt (V6) pentru a
pastra integritatea formei furtunii la 2 ore, si aplica Cresterea Spatiala 
Localizata (V7) strict pe centroidul anticipat pentru corectia volumetrica.
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.sparse as sp
from src.core.domain import StormCell
from src.core.algorithms_config import config as algo_config

from config import RAIN_THRESHOLD_MIN, RAIN_THRESHOLD_TRACKING


class AdvectionEngine:
    _cached_grids: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def _get_grids(cls, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if shape not in cls._cached_grids:
            cls._cached_grids[shape] = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        return cls._cached_grids[shape]
    
    @staticmethod
    def _create_spatial_growth_mask(
        shape: tuple[int, int],
        tracked_cells: list[StormCell],
        steps: int
    ) -> np.ndarray:
        """Creeaza o masca globala de crestere/scadere bazata pe halouri Gaussiene."""
        h, w = shape
        growth_mask = np.ones((h, w), dtype=np.float32)
        decay_mask = np.ones((h, w), dtype=np.float32)
        
        valid_cells = [c for c in tracked_cells if c.is_tracked]
        if not valid_cells:
            return growth_mask
            
        y_grid, x_grid = AdvectionEngine._get_grids((h, w))
        
        for c in valid_cells:
            area = max(1.0, c.predicted_area_kalman)
            d_area = c.d_area_kalman
            
            # V18: Model Asimptotic de Crestere (Fara parabole explozive)
            if d_area > 0:
                tau_growth = algo_config.ADV_TAU_GROWTH
                pred_area = area + d_area * tau_growth * (1.0 - np.exp(-steps / tau_growth))
            else:
                tau_decay = algo_config.ADV_TAU_DECAY
                pred_area = area + d_area * tau_decay * (1.0 - np.exp(-steps / tau_decay))
                
            pred_area = max(1.0, pred_area) 
            
            # V29: Termodinamic Lifecycle Decay (Rezolva alarmele false la orizonturi mari)
            phase = getattr(c, 'lifecycle_phase', 'MATURITY')
            
            if getattr(algo_config, 'ENABLE_THERMODYNAMIC_DECAY', True):
                curve = algo_config.DECAY_CURVES.get(phase, algo_config.DECAY_CURVES["MATURITY"])
                lookup_step = min(max(0, steps), len(curve) - 1)
                max_growth = curve[lookup_step]
                
                if phase == 'BIRTH' and steps <= 3:
                    max_growth = max(max_growth, algo_config.BIRTH_MAX_MULTIPLIER)
            else:
                max_growth = max(0.3, algo_config.ADV_MAX_GROWTH_LIMIT - algo_config.ADV_CLIMATOLOGICAL_DECAY_RATE * max(0, steps - 2))
                if phase == 'BIRTH':
                    max_growth = max(max_growth, 2.0)
                
            cumulative_factor = min(pred_area / area, max_growth)
            
            # 3. Acceleratia Spatiala (Traiectorii Curbate) bazate pe amortizare Singer (gamma=0.8)
            gamma = 0.8
            if gamma == 1.0:
                term_a = 0.5 * (steps ** 2)
            else:
                term_a = (2*steps - (1+gamma)*(1-gamma**steps)/(1-gamma)) / (2*(1-gamma))
            
            cy = c.centroid_y + c.v_y * steps + c.a_y * term_a
            cx = c.centroid_x + c.v_x * steps + c.a_x * term_a
            
            # V19: Raza haloului asimetrica (corecteaza explozia de volum!)
            # Daca furtuna creste, folosim aria extinsa. Daca moare, TRBUIE sa acoperim intreaga furtuna existenta!
            if cumulative_factor >= 1.0:
                radius = max(5.0, np.sqrt(pred_area / np.pi))
            else:
                radius = max(5.0, np.sqrt(area / np.pi))
            
            # Optimizare: Bounding Box local (3-Sigma)
            y_min = max(0, int(cy - 3 * radius))
            y_max = min(h, int(cy + 3 * radius + 1))
            x_min = max(0, int(cx - 3 * radius))
            x_max = min(w, int(cx + 3 * radius + 1))
            
            if y_min >= y_max or x_min >= x_max:
                continue
                
            y_slice = slice(y_min, y_max)
            x_slice = slice(x_min, x_max)
            
            dist_sq = (x_grid[y_slice, x_slice] - cx)**2 + (y_grid[y_slice, x_slice] - cy)**2
            halo = np.exp(-dist_sq / (2.0 * radius**2))
            
            local_multiplier = 1.0 + (cumulative_factor - 1.0) * halo
            
            # Evitam inmultirea multiplicatorilor (explozie) si luam doar impactul maxim
            if cumulative_factor >= 1.0:
                growth_mask[y_slice, x_slice] = np.maximum(growth_mask[y_slice, x_slice], local_multiplier)
            else:
                decay_mask[y_slice, x_slice] = np.minimum(decay_mask[y_slice, x_slice], local_multiplier)
            
        final_mask = growth_mask * decay_mask
        return np.clip(final_mask, 0.5, 2.0)

    @staticmethod
    def _blend_kinematics(
        step: int, 
        flow_x: np.ndarray, 
        flow_y: np.ndarray, 
        valid_cells: list[StormCell], 
        grid_h: int, 
        grid_w: int, 
        x_grid: np.ndarray, 
        y_grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend intre Optical Flow global si predictiile Kalman locale."""
        blended_flow_x = flow_x.copy()
        blended_flow_y = flow_y.copy()
        
        for c in valid_cells:
            gamma = 0.8
            term_a = (2*step - (1+gamma)*(1-gamma**step)/(1-gamma)) / (2*(1-gamma))
            term_v = (1 - gamma**step) / (1 - gamma)
            
            cx = c.centroid_x + c.v_x * step + c.a_x * term_a
            cy = c.centroid_y + c.v_y * step + c.a_y * term_a
            
            vx = c.v_x + c.a_x * term_v
            vy = c.v_y + c.a_y * term_v
            
            area = max(1.0, c.predicted_area_kalman)
            radius = max(5.0, np.sqrt(area / np.pi))
            
            y_min = max(0, int(cy - 3 * radius))
            y_max = min(grid_h, int(cy + 3 * radius + 1))
            x_min = max(0, int(cx - 3 * radius))
            x_max = min(grid_w, int(cx + 3 * radius + 1))
            
            if y_min >= y_max or x_min >= x_max:
                continue
                
            y_slice = slice(y_min, y_max)
            x_slice = slice(x_min, x_max)
            
            kalman_confidence = np.clip(10.0 / (10.0 + c.uncertainty_trace), 0.1, 0.9)
            dist_sq = (x_grid[y_slice, x_slice] - cx)**2 + (y_grid[y_slice, x_slice] - cy)**2
            weight = np.exp(-dist_sq / (2.0 * radius**2)) * kalman_confidence
            
            blended_flow_x[y_slice, x_slice] = blended_flow_x[y_slice, x_slice] * (1 - weight) + vx * weight
            blended_flow_y[y_slice, x_slice] = blended_flow_y[y_slice, x_slice] * (1 - weight) + vy * weight
            
        return blended_flow_x.astype(np.float32), blended_flow_y.astype(np.float32)

    @staticmethod
    def _apply_sprog_diffusion(
        shifted_grown: np.ndarray, 
        step: int, 
        valid_cells: list[StormCell], 
        grid_h: int, 
        grid_w: int, 
        base_uncertainty: float
    ) -> np.ndarray:
        """Aplica blur S-PROG pentru a simula incertitudinea extinderii norilor."""
        if step <= 2:
            return shifted_grown

        base_sigma = float(np.clip(base_uncertainty * 0.1 * step, 0.2, 2.0))
        base_ksize = int(base_sigma * 3) | 1
        
        if base_ksize >= 3:
            shifted_grown_blurred = cv2.GaussianBlur(shifted_grown, (base_ksize, base_ksize), base_sigma)
        else:
            shifted_grown_blurred = shifted_grown.copy()
            
        if not valid_cells:
            return shifted_grown_blurred
            
        y_grid_local, x_grid_local = AdvectionEngine._get_grids((grid_h, grid_w))
        
        for c in valid_cells:
            uncertainty = c.uncertainty_trace
            local_sigma = base_sigma + 0.15 * step + 0.02 * uncertainty
            local_sigma = float(np.clip(local_sigma, base_sigma, 8.0))
            
            if local_sigma <= base_sigma + 0.1:
                continue 
                
            ksize = int(local_sigma * 3) | 1
            
            gamma = 0.8
            term_a = (2*step - (1+gamma)*(1-gamma**step)/(1-gamma)) / (2*(1-gamma))
            cx = c.centroid_x + c.v_x * step + c.a_x * term_a
            cy = c.centroid_y + c.v_y * step + c.a_y * term_a
            
            area = max(1.0, c.predicted_area_kalman)
            radius = max(5.0, np.sqrt(area / np.pi)) * 1.5
            
            y_min = max(0, int(cy - 3 * radius))
            y_max = min(grid_h, int(cy + 3 * radius + 1))
            x_min = max(0, int(cx - 3 * radius))
            x_max = min(grid_w, int(cx + 3 * radius + 1))
            
            if y_min >= y_max or x_min >= x_max:
                continue
                
            y_slice = slice(y_min, y_max)
            x_slice = slice(x_min, x_max)
            
            patch_orig = shifted_grown[y_slice, x_slice]
            blurred_patch = cv2.GaussianBlur(patch_orig, (ksize, ksize), local_sigma)
            
            kalman_confidence = np.clip(10.0 / (10.0 + c.uncertainty_trace), 0.1, 0.9)
            dist_sq = (x_grid_local[y_slice, x_slice] - cx)**2 + (y_grid_local[y_slice, x_slice] - cy)**2
            weight = np.exp(-dist_sq / (2.0 * radius**2)) * kalman_confidence
            
            patch_base = shifted_grown_blurred[y_slice, x_slice]
            shifted_grown_blurred[y_slice, x_slice] = patch_base * (1.0 - weight) + blurred_patch * weight
            
        return shifted_grown_blurred

    @staticmethod
    def extrapolate(
        rain_rate: np.ndarray,
        flow: np.ndarray | None,
        tracked_cells: list[StormCell],
        horizons: list[tuple[int, str]],
    ) -> tuple[dict[str, sp.csr_matrix], dict[int, np.ndarray]]:
        """Extrapoleaza precipitatiile folosind Advectie Liniara dintr-un singur salt."""
        grid_h, grid_w = rain_rate.shape
        rain_rate = np.nan_to_num(rain_rate, nan=0.0).astype(np.float32)
        y_grid, x_grid = AdvectionEngine._get_grids((grid_h, grid_w))
        
        if flow is None:
            flow_x = np.zeros((grid_h, grid_w), dtype=np.float32)
            flow_y = np.zeros((grid_h, grid_w), dtype=np.float32)
        else:
            flow_x = flow[:, :, 0]
            flow_y = flow[:, :, 1]
            
        sparse_preds = {}
        float_preds = {}
        
        max_step = max(h[0] for h in horizons) if horizons else 0
        horizon_map = {h[0]: h[1] for h in horizons}
        
        valid_cells = [c for c in tracked_cells if c.is_tracked]
        mean_tracking_error = float(np.mean([c.prediction_error_pixels for c in valid_cells])) if valid_cells else 0.5
        base_uncertainty = max(0.2, mean_tracking_error)
        
        map_x = x_grid.copy()
        map_y = y_grid.copy()
        
        for step in range(1, max_step + 1):
            blended_flow_x, blended_flow_y = AdvectionEngine._blend_kinematics(
                step, flow_x, flow_y, valid_cells, grid_h, grid_w, x_grid, y_grid
            )

            flow_at_p_x = cv2.remap(blended_flow_x, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            flow_at_p_y = cv2.remap(blended_flow_y, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            map_x = (map_x - flow_at_p_x).astype(np.float32)
            map_y = (map_y - flow_at_p_y).astype(np.float32)
            
            shifted = cv2.remap(
                rain_rate, map_x, map_y, 
                interpolation=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
            
            growth_mask = AdvectionEngine._create_spatial_growth_mask(
                (grid_h, grid_w), tracked_cells, step
            )
            
            shifted_grown = shifted * growth_mask
            float_preds[step] = shifted_grown
            
            if step in horizon_map:
                name = horizon_map[step]
                
                shifted_grown_blurred = AdvectionEngine._apply_sprog_diffusion(
                    shifted_grown, step, valid_cells, grid_h, grid_w, base_uncertainty
                )
                
                base_mask = (shifted_grown_blurred >= RAIN_THRESHOLD_TRACKING).astype(np.float32)
                sparse_preds[name] = sp.csr_matrix(base_mask)
            
        return sparse_preds, float_preds
