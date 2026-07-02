import cv2
import numpy as np
from src.core.domain import StormCell
from src.core.nowcast.kinematics import KinematicsEngine

class SpatialMaskBuilder:
    def __init__(self) -> None:
        self._cached_grids: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _get_grids(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if shape not in self._cached_grids:
            self._cached_grids[shape] = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        return self._cached_grids[shape]

    def create_spatial_growth_mask(
        self,
        shape: tuple[int, int],
        simulated_cells: list[StormCell],
        original_cells_dict: dict[str, StormCell],
        flow: np.ndarray | None = None
    ) -> np.ndarray:
        """Creeaza o masca globala de crestere/scadere bazata pe halouri Gaussiene si E_pred."""
        h, w = shape
        growth_mask = np.ones((h, w), dtype=np.float32)
        decay_mask = np.ones((h, w), dtype=np.float32)
        
        y_grid, x_grid = self._get_grids((h, w))
        
        for c in simulated_cells:
            orig_c = original_cells_dict.get(c.cell_id)
            if not orig_c:
                continue
                
            orig_E = max(1e-6, orig_c.E)
            cumulative_factor = getattr(c, "cumulative_R", c.E / orig_E)
            
            cy = c.predicted_centroid_y
            cx = c.predicted_centroid_x
            
            # Flow Gating (Termodinamica cuplata)
            if flow is not None:
                v_mag = np.hypot(c.v_x, c.v_y)
                if v_mag > 0.1:
                    f_x, f_y = KinematicsEngine._sample_flow_at(flow, cx, cy)
                    f_mag = np.hypot(f_x, f_y)
                    if f_mag > 0.1:
                        cos_sim = (c.v_x * f_x + c.v_y * f_y) / (v_mag * f_mag)
                        # Permitem maxim 1.0 pentru aliniere perfecta (plafonat pentru a preveni explozia 59.2%)
                        alignment_weight = min(1.0, max(0.5, 0.7 + 0.5 * cos_sim))
                        cumulative_factor *= alignment_weight
            
            cumulative_factor = np.clip(cumulative_factor, 0.2, 3.0)
            
            mean_intensity = max(1e-6, orig_c.mean_intensity)
            pred_area = (orig_E * cumulative_factor * 1000.0) / mean_intensity
            
            theta = orig_c.orientation
            sigma_x = max(2.0, orig_c.minor_axis_length / 2.0)
            sigma_y = max(2.0, orig_c.major_axis_length / 2.0)
            
            if cumulative_factor >= 1.0:
                area_factor = min(np.sqrt(cumulative_factor), 1.5)  # ponytail: cap spatial expansion to prevent quadratic volume explosion
                sigma_x *= area_factor
                sigma_y *= area_factor
            else:
                area_factor = 1.0
            
            radius = max(5.0, max(sigma_x, sigma_y))
            
            y_min = max(0, int(cy - 3 * radius))
            y_max = min(h, int(cy + 3 * radius + 1))
            x_min = max(0, int(cx - 3 * radius))
            x_max = min(w, int(cx + 3 * radius + 1))
            
            if y_min >= y_max or x_min >= x_max:
                continue
                
            y_slice = slice(y_min, y_max)
            x_slice = slice(x_min, x_max)
            
            dx = x_grid[y_slice, x_slice] - cx
            dy = y_grid[y_slice, x_slice] - cy
            
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            x_rot = dx * cos_t - dy * sin_t
            y_rot = dx * sin_t + dy * cos_t
            
            halo = np.exp(- (x_rot**2 / (2.0 * sigma_x**2) + y_rot**2 / (2.0 * sigma_y**2)))
            
            local_multiplier = 1.0 + (cumulative_factor - 1.0) * halo
            
            if cumulative_factor >= 1.0:
                # Conservarea masei: Daca marim norul (area_factor > 1), scadem intensitatea precipitațiilor
                local_multiplier /= (area_factor ** 2)
                growth_mask[y_slice, x_slice] = np.maximum(growth_mask[y_slice, x_slice], local_multiplier)
            else:
                decay_mask[y_slice, x_slice] = np.minimum(decay_mask[y_slice, x_slice], local_multiplier)
            
        final_mask = growth_mask * decay_mask
        # ponytail: cap final multiplier to 2.0 to prevent runaway mass injection at long horizons
        return np.clip(final_mask, 0.0, 2.0)

    def apply_sprog_diffusion(
        self,
        shifted_grown: np.ndarray, 
        step: int, 
        valid_cells: list[StormCell], 
        grid_h: int, 
        grid_w: int, 
        base_uncertainty: float
    ) -> np.ndarray:
        """Aplica S-PROG Autoregressive Scale-Dependent Decay pentru disipare si difuzie."""
        if step <= 2 or not valid_cells:
            return shifted_grown.copy()
            
        # Laplacian Pyramid Decomposition (2 scales)
        # Baza: Ploaie stratiforma la scara larga
        # Detaliu: Convectie intensa la scara mica
        k_size = 15  # Filtru fix pentru separarea scarii stratiforme
        base_field = cv2.GaussianBlur(shifted_grown, (k_size, k_size), 0)
        detail_field = shifted_grown - base_field
        
        # S-PROG Temporal Decay (AR-2 Critically Damped)
        # Ofera inertie furtunilor la orizonturi scurte (15m-1h) si cadere accelerata la 2h
        tau_detail = 6.0
        tau_base = 24.0
        
        decayed_detail = detail_field * (1.0 + step / tau_detail) * np.exp(-step / tau_detail)
        decayed_base = base_field * (1.0 + step / tau_base) * np.exp(-step / tau_base)
        
        # Reconstructia si eliminarea valorilor negative
        shifted_grown_sprog = np.clip(decayed_base + decayed_detail, 0.0, None)
        
        # Pastram si un mic blur de advectie (blend_map) pentru netezire fina pe margini
        extra_px = int(0.1 * step + 0.03 * base_uncertainty)
        extra_px = min(extra_px, 3)
        
        if extra_px > 0:
            local_ksize = 2 * extra_px + 1
            sigma = extra_px / 2.0
            blurred = cv2.GaussianBlur(shifted_grown_sprog, (local_ksize, local_ksize), sigma)
            
            blend_map = np.zeros((grid_h, grid_w), dtype=np.float32)
            y_grid_local, x_grid_local = self._get_grids((grid_h, grid_w))
            
            for c in valid_cells:
                cx = c.predicted_centroid_x
                cy = c.predicted_centroid_y
                kalman_confidence = np.clip(10.0 / (10.0 + c.uncertainty_trace), 0.1, 0.9)
                extra_blend = 1.0 - kalman_confidence
                
                area = max(1.0, c.predicted_area_kalman)
                radius = max(5.0, np.sqrt(area / np.pi)) * 2.0
                
                y_min = max(0, int(cy - 3 * radius))
                y_max = min(grid_h, int(cy + 3 * radius + 1))
                x_min = max(0, int(cx - 3 * radius))
                x_max = min(grid_w, int(cx + 3 * radius + 1))
                
                if y_min >= y_max or x_min >= x_max:
                    continue
                    
                y_slice = slice(y_min, y_max)
                x_slice = slice(x_min, x_max)
                
                dist_sq = (x_grid_local[y_slice, x_slice] - cx)**2 + (y_grid_local[y_slice, x_slice] - cy)**2
                halo = np.exp(-dist_sq / (2.0 * radius**2))
                
                blend_map[y_slice, x_slice] = np.maximum(blend_map[y_slice, x_slice], extra_blend * halo)
                
            shifted_grown_sprog = (1.0 - blend_map) * shifted_grown_sprog + blend_map * blurred
            
        return shifted_grown_sprog
