"""Motor de advectie Hibrid (Linear Jump + Spatial Growth + Directional Blur).

Aplica advectia semi-Lagrangiana liniara dintr-un singur salt (V6) pentru a
pastra integritatea formei furtunii la 2 ore, si aplica Cresterea Spatiala 
Localizata (V7) strict pe centroidul anticipat pentru corectia volumetrica.
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from src.core.domain import StormCell
from src.core.algorithms_config import config as algo_config
from src.core.reaction_diffusion import update_energy, lifecycle

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
        simulated_cells: list[StormCell],
        original_cells_dict: dict[str, StormCell]
    ) -> np.ndarray:
        """Creeaza o masca globala de crestere/scadere bazata pe halouri Gaussiene si E_pred."""
        h, w = shape
        growth_mask = np.ones((h, w), dtype=np.float32)
        decay_mask = np.ones((h, w), dtype=np.float32)
        
        y_grid, x_grid = AdvectionEngine._get_grids((h, w))
        
        for c in simulated_cells:
            orig_c = original_cells_dict.get(c.cell_id)
            if not orig_c:
                continue
                
            orig_E = max(1e-6, orig_c.E)
            cumulative_factor = getattr(c, "cumulative_R", c.E / orig_E)
            cumulative_factor = np.clip(cumulative_factor, 0.2, 3.0)
            
            cy = c.predicted_centroid_y
            cx = c.predicted_centroid_x
            
            # Estimam aria plecand de la volumul (E) curent. Presupunem intensitate medie constanta.
            # NOTA: c.E este normalizat (/ 1000.0) in storm_tracker, deci inmultim la loc.
            mean_intensity = max(1e-6, orig_c.mean_intensity)
            
            # Calculam aria bazata STRICT pe cresterea pur-fizica, nu pe artefacte de difuzie
            pred_area = (orig_E * cumulative_factor * 1000.0) / mean_intensity
            
            # Folosim radius marit pentru decay, altfel marginile furtunii supravietuiesc la infinit
            if cumulative_factor < 1.0:
                orig_area = (orig_E * 1000.0) / mean_intensity
                radius = max(5.0, np.sqrt(orig_area / np.pi))
            else:
                radius = max(5.0, np.sqrt(pred_area / np.pi))
            
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
            
            if cumulative_factor >= 1.0:
                growth_mask[y_slice, x_slice] = np.maximum(growth_mask[y_slice, x_slice], local_multiplier)
            else:
                decay_mask[y_slice, x_slice] = np.minimum(decay_mask[y_slice, x_slice], local_multiplier)
            
        final_mask = growth_mask * decay_mask
        return np.clip(final_mask, 0.0, 5.0)

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
        
        # State pentru Reaction-Diffusion (Phase 4)
        simulated_cells = [c.clone() for c in valid_cells]
        for c in simulated_cells:
            c.cumulative_R = 1.0
        original_cells_dict = {c.cell_id: c for c in valid_cells}
        
        for step in range(1, max_step + 1):
            # 1. Update Kinematics (Advectie centroid)
            for c in simulated_cells:
                gamma = 0.8
                term_a = (2*step - (1+gamma)*(1-gamma**step)/(1-gamma)) / (2*(1-gamma))
                c.predicted_centroid_x = c.centroid_x + c.v_x * step + c.a_x * term_a
                c.predicted_centroid_y = c.centroid_y + c.v_y * step + c.a_y * term_a
                
            # 2. Update Reaction-Diffusion (Energetics)
            if len(simulated_cells) > 0:
                coords = np.array([[c.predicted_centroid_x, c.predicted_centroid_y] for c in simulated_cells])
                if len(coords) > 1:
                    dist_matrix = cdist(coords, coords)
                else:
                    dist_matrix = np.zeros((1, 1))
                
                updates = []
                for i, c in enumerate(simulated_cells):
                    if len(coords) > 1:
                        neighbor_indices = np.where((dist_matrix[i] < 50.0) & (dist_matrix[i] > 0))[0]
                        neighbors_E = np.array([simulated_cells[j].E for j in neighbor_indices])
                    else:
                        neighbors_E = np.array([])
                        
                    E_new, dE_new, R_applied = update_energy(c.E, neighbors_E, c.dE)
                    updates.append((E_new, dE_new, R_applied))
                    
                for i, c in enumerate(simulated_cells):
                    E_new, dE_new, R_applied = updates[i]
                    c.E = max(E_new, 1e-6)
                    c.dE = dE_new
                    c.cumulative_R *= R_applied
                    c.lifecycle_phase = lifecycle(c.E, c.dE)
            
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
                (grid_h, grid_w), simulated_cells, original_cells_dict
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
