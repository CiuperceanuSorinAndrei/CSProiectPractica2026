"""Motor de advectie Hibrid (Linear Jump + Spatial Growth + Directional Blur).

Aplica advectia semi-Lagrangiana liniara dintr-un singur salt (V6) pentru a
pastra integritatea formei furtunii la 2 ore, si aplica Cresterea Spatiala 
Localizata (V7) strict pe centroidul anticipat pentru corectia volumetrica.
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.sparse as sp

from config import RAIN_THRESHOLD_MIN


class AdvectionEngine:
    
    @staticmethod
    def _create_spatial_growth_mask(
        shape: tuple[int, int],
        tracked_cells: list[dict],
        steps: int
    ) -> np.ndarray:
        """Creeaza o masca globala de crestere/scadere bazata pe halouri Gaussiene."""
        h, w = shape
        growth_mask = np.ones((h, w), dtype=np.float32)
        
        valid_cells = [c for c in tracked_cells if c.get("is_tracked", False)]
        if not valid_cells:
            return growth_mask
            
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        
        for c in valid_cells:
            d_area = c.get("d_area_kalman", 0.0)
            area = max(1.0, c.get("predicted_area_kalman", 1.0))
            growth_rate = d_area / area
            
            # Limitare a ratei pentru a nu exploda fizic
            step_factor = float(np.clip(1.0 + growth_rate, 0.95, 1.05))
            cumulative_factor = step_factor ** steps
            
            # Deoarece folosim salt direct (Linear Jump), pozitia prezisa este fix:
            cy = c.get("centroid_y", h/2) + c.get("v_y", 0.0) * steps
            cx = c.get("centroid_x", w/2) + c.get("v_x", 0.0) * steps
            
            # Raza haloului este proportionala cu aria curenta
            radius = max(5.0, np.sqrt(area) * 2.0)
            
            dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
            halo = np.exp(-dist_sq / (2.0 * radius**2))
            
            local_multiplier = 1.0 + (cumulative_factor - 1.0) * halo
            growth_mask *= local_multiplier
            
        return growth_mask

    @staticmethod
    def extrapolate(
        rain_rate: np.ndarray,
        flow: np.ndarray | None,
        tracked_cells: list[dict],
        horizons: list[tuple[int, str]],
    ) -> tuple[dict[str, sp.csr_matrix], dict[int, np.ndarray]]:
        """Extrapoleaza precipitatiile folosind Advectie Liniara dintr-un singur salt.
        
        Returneaza:
            sparse_preds: predictii de maska (ploaie da/nu) strict pentru orizonturile cerute.
            float_preds: matricile complete float de rain_rate pentru TOATE step-urile
                         de la 1 la max_step, pentru integrarea volumetrica corecta.
        """
        grid_h, grid_w = rain_rate.shape
        y_grid, x_grid = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
        
        if flow is None:
            flow_x = np.zeros((grid_h, grid_w), dtype=np.float32)
            flow_y = np.zeros((grid_h, grid_w), dtype=np.float32)
        else:
            flow_x = flow[:, :, 0]
            flow_y = flow[:, :, 1]
            
        sparse_preds = {}
        float_preds = {}
        
        valid_cells = [c for c in tracked_cells if c.get("is_tracked", False)]
        mean_v_x = float(np.nan_to_num(np.mean([c.get("v_x", 0.0) for c in valid_cells]))) if valid_cells else 0.0
        mean_v_y = float(np.nan_to_num(np.mean([c.get("v_y", 0.0) for c in valid_cells]))) if valid_cells else 0.0
        
        max_step = max(h[0] for h in horizons) if horizons else 0
        horizon_map = {h[0]: h[1] for h in horizons}
        
        # Calculam ploaia anticipata pentru TOATE sferturile de ora pentru integrarea volumului
        for step in range(1, max_step + 1):
            map_x = x_grid - flow_x * step
            map_y = y_grid - flow_y * step
            
            shifted = cv2.remap(
                rain_rate, map_x, map_y, 
                interpolation=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
            
            growth_mask = AdvectionEngine._create_spatial_growth_mask(
                (grid_h, grid_w), tracked_cells, step
            )
            
            # Masa reala (Volum pur, conservat, fara dilatare artificiala)
            shifted_grown = shifted * growth_mask
            float_preds[step] = shifted_grown
            
            # Masca de incertitudine cinematica (Dilatata pentru POD)
            if step in horizon_map:
                name = horizon_map[step]
                
                # Masca initiala
                base_mask = (shifted_grown >= RAIN_THRESHOLD_MIN).astype(np.float32)
                
                # Aplicam dilatarea STRICT pe masca
                if valid_cells and step > 1:
                    if abs(mean_v_x) > 0.1 or abs(mean_v_y) > 0.1:
                        angle = np.degrees(np.arctan2(mean_v_y, mean_v_x))
                        length = min(int(step * 1.5), 15)
                        if length % 2 == 0:
                            length += 1
                        
                        kernel = np.zeros((length, length), dtype=np.float32)
                        kernel[length // 2, :] = 1.0
                        M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
                        kernel = cv2.warpAffine(kernel, M, (length, length))
                        
                        morph_kernel = (kernel > 0.1).astype(np.uint8)
                        base_mask = cv2.dilate(base_mask, morph_kernel, iterations=1)
                
                sparse_preds[name] = sp.csr_matrix(base_mask)
            
        return sparse_preds, float_preds
