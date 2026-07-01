"""Motor de advectie Hibrid (Linear Jump + Spatial Growth + Directional Blur)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.core.domain import StormCell
from src.core.nowcast.kinematics import KinematicsEngine
from src.core.nowcast.kinematic_advector import KinematicAdvector
from src.core.nowcast.thermodynamic_simulator import ThermodynamicSimulator
from src.core.nowcast.spatial_mask_builder import SpatialMaskBuilder

from config import RAIN_THRESHOLD_TRACKING

class AdvectionEngine:
    def __init__(
        self,
        kinematic_advector: KinematicAdvector,
        thermodynamic_simulator: ThermodynamicSimulator,
        spatial_mask_builder: SpatialMaskBuilder,
    ) -> None:
        self.kinematic_advector = kinematic_advector
        self.thermodynamic_simulator = thermodynamic_simulator
        self.spatial_mask_builder = spatial_mask_builder

    def extrapolate(
        self,
        rain_rate: np.ndarray,
        flow: np.ndarray | None,
        tracked_cells: list[StormCell],
        horizons: list[tuple[int, str]],
    ) -> tuple[dict[str, sp.csr_matrix], dict[int, np.ndarray], dict[int, list[StormCell]]]:
        """Extrapoleaza precipitatiile folosind Advectie Liniara dintr-un singur salt."""
        grid_h, grid_w = rain_rate.shape
        rain_rate = np.nan_to_num(rain_rate, nan=0.0).astype(np.float32)
        y_grid, x_grid = self.spatial_mask_builder._get_grids((grid_h, grid_w))
        
        if flow is None:
            flow_x = np.zeros((grid_h, grid_w), dtype=np.float32)
            flow_y = np.zeros((grid_h, grid_w), dtype=np.float32)
        else:
            flow_x = flow[:, :, 0]
            flow_y = flow[:, :, 1]
            
        sparse_preds = {}
        float_preds = {}
        predicted_cells_dict = {}
        
        max_step = max(h[0] for h in horizons) if horizons else 0
        horizon_map = {h[0]: h[1] for h in horizons}
        
        valid_cells = [c for c in tracked_cells if c.is_tracked]
        
        # ponytail: prevent np.mean([]) NaN warning and crash if valid_cells contains NaNs
        if valid_cells:
            mean_tracking_error = float(np.nanmean([c.prediction_error_pixels for c in valid_cells]))
            if np.isnan(mean_tracking_error):
                mean_tracking_error = 0.5
        else:
            mean_tracking_error = 0.5
        base_uncertainty = max(0.2, mean_tracking_error)
        
        map_x = x_grid.copy()
        map_y = y_grid.copy()
        
        # State pentru Reaction-Diffusion (Phase 4)
        simulated_cells = [c.clone() for c in valid_cells]
        
        # Phase 3: Pre-allocate NumPy arrays for centroid coordinates
        coords = np.zeros((len(simulated_cells), 2), dtype=np.float64)
        
        for c in simulated_cells:
            c.initialize_simulation_state()
            
        original_cells_dict = {c.cell_id: c for c in valid_cells}
        
        for step in range(1, max_step + 1):
            # 1. Update Kinematics (Euler-Lagrange with Flow Forcing)
            KinematicsEngine.update_positions(simulated_cells, flow, step)
                
            # 2. Update Reaction-Diffusion (Energetics)
            self.thermodynamic_simulator.simulate_step(simulated_cells, coords)
            
            blended_flow_x, blended_flow_y = KinematicsEngine.blend_kinematics(
                flow_x, flow_y, simulated_cells, grid_h, grid_w, x_grid, y_grid
            )

            # Phase 1: Fix Semi-Lagrangian Integration
            shifted, map_x, map_y = self.kinematic_advector.advect(
                rain_rate, map_x, map_y, x_grid, y_grid, blended_flow_x, blended_flow_y
            )
            
            growth_mask = self.spatial_mask_builder.create_spatial_growth_mask(
                (grid_h, grid_w), simulated_cells, original_cells_dict, flow
            )
            
            shifted_grown = shifted * growth_mask
            float_preds[step] = shifted_grown
            
            # Phase 6: Păstrăm snapshot-ul obiectelor prezise pentru FAR Inspector
            # Omit celulele care s-au disipat (altfel inspectorul le vede ca BAD_ADVECTION)
            predicted_cells_dict[step] = [c.clone() for c in simulated_cells if c.lifecycle_phase != "DISSIPATION"]
            
            if step in horizon_map:
                name = horizon_map[step]
                
                # Restore SPROG Diffusion
                shifted_grown_blurred = self.spatial_mask_builder.apply_sprog_diffusion(
                    shifted_grown, step, valid_cells, grid_h, grid_w, base_uncertainty
                )
                
                base_mask = (shifted_grown_blurred >= RAIN_THRESHOLD_TRACKING).astype(np.float32)
                sparse_preds[name] = sp.csr_matrix(base_mask)
            
        return sparse_preds, float_preds, predicted_cells_dict
