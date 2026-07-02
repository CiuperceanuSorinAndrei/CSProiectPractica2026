from __future__ import annotations

import numpy as np

from src.core.domain import StormCell

class KinematicsEngine:
    """Modul responsabil exclusiv cu rezolvarea cinematicii celulelor de furtuna
    (cuplarea Eulerian-Lagrangiana cu vantul optic, filtrarea EMA si integrarea traiectoriei).
    """

    @staticmethod
    def _sample_flow_at(flow: np.ndarray | None, x: float, y: float) -> tuple[float, float]:
        if flow is None:
            return 0.0, 0.0
        h, w = flow.shape[:2]
        x_idx = int(round(np.clip(x, 0, w - 1)))
        y_idx = int(round(np.clip(y, 0, h - 1)))
        
        y_min = max(0, y_idx - 1)
        y_max = min(h, y_idx + 2)
        x_min = max(0, x_idx - 1)
        x_max = min(w, x_idx + 2)
        
        patch = flow[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            return 0.0, 0.0
            
        u = float(np.nanmedian(patch[:, :, 0]))
        v = float(np.nanmedian(patch[:, :, 1]))
        return u, v

    @staticmethod
    def update_positions(simulated_cells: list[StormCell], flow: np.ndarray | None, step: int):
        """Integreaza viteza in pozitie, aplicand forțarea vantului Eulerian."""
        for c in simulated_cells:
            if flow is not None:
                # Eulerian Forcing
                flow_coupling = 0.25 # Intensitatea cu care vântul afectează mișcarea
                
                # Flow-ul este citit de la locatia la care A AJUNS celula
                f_x, f_y = KinematicsEngine._sample_flow_at(flow, c.predicted_centroid_x, c.predicted_centroid_y)
                
                # Low-pass filter (EMA)
                c.flow_vec_smooth_x = 0.8 * c.flow_vec_smooth_x + 0.2 * f_x
                c.flow_vec_smooth_y = 0.8 * c.flow_vec_smooth_y + 0.2 * f_y
                
                # Accelerație generată de vânt
                a_flow_x = (c.flow_vec_smooth_x - c.v_x) * flow_coupling
                a_flow_y = (c.flow_vec_smooth_y - c.v_y) * flow_coupling
                
                # Viteza se adapteaza strict pe baza curgerii (flow), eliminand acceleratia
                # Kalman din bucla de integrare pentru a preveni "runaway velocity" (FAR 0.57).
                # State-ul Kalman si-a integrat deja acceleratia in predictia initiala.
                c.v_x += a_flow_x
                c.v_y += a_flow_y
            
            # Integratorul (viteză -> poziție)
            c.predicted_centroid_x += c.v_x
            c.predicted_centroid_y += c.v_y

    @staticmethod
    def blend_kinematics(
        flow_x: np.ndarray, 
        flow_y: np.ndarray, 
        simulated_cells: list[StormCell], 
        grid_h: int, 
        grid_w: int, 
        x_grid: np.ndarray, 
        y_grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend intre Optical Flow global si predictiile Euler-Lagrange locale."""
        blended_flow_x = flow_x.copy()
        blended_flow_y = flow_y.copy()
        
        for c in simulated_cells:
            cx = c.predicted_centroid_x
            cy = c.predicted_centroid_y
            vx = c.v_x
            vy = c.v_y
            
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
