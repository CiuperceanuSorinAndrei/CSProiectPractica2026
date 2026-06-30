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

from src.core.domain import StormCell, CellDiagnostics
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
    def _sample_flow_at(flow: np.ndarray | None, x: float, y: float) -> tuple[float, float]:
        if flow is None:
            return 0.0, 0.0
        h, w = flow.shape[:2]
        x_idx = int(round(np.clip(x, 0, w - 1)))
        y_idx = int(round(np.clip(y, 0, h - 1)))
        return float(flow[y_idx, x_idx, 0]), float(flow[y_idx, x_idx, 1])

    @staticmethod
    def _create_spatial_growth_mask(
        shape: tuple[int, int],
        simulated_cells: list[StormCell],
        original_cells_dict: dict[str, StormCell],
        flow: np.ndarray | None = None
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
            
            cy = c.predicted_centroid_y
            cx = c.predicted_centroid_x
            
            # Flow Gating (Termodinamica cuplata)
            if flow is not None:
                v_mag = np.hypot(c.v_x, c.v_y)
                if v_mag > 0.1:
                    f_x, f_y = AdvectionEngine._sample_flow_at(flow, cx, cy)
                    f_mag = np.hypot(f_x, f_y)
                    if f_mag > 0.1:
                        cos_sim = (c.v_x * f_x + c.v_y * f_y) / (v_mag * f_mag)
                        # Permitem weight > 1.0 pentru aliniere perfecta (pana la 1.2) pentru recuperare volum
                        alignment_weight = max(0.5, 0.7 + 0.5 * cos_sim)
                        cumulative_factor *= alignment_weight
            
            cumulative_factor = np.clip(cumulative_factor, 0.2, 3.0)
            
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

    @staticmethod
    def _apply_sprog_diffusion(
        shifted_grown: np.ndarray, 
        step: int, 
        valid_cells: list[StormCell], 
        grid_h: int, 
        grid_w: int, 
        base_uncertainty: float
    ) -> np.ndarray:
        """Aplica Morphological Dilation S-PROG pentru a acoperi eroarea de advectie.
        
        Dilatarea morfologica extinde aria furtunii proportional cu orizontul de timp,
        fara sa scada intensitatea precipitatiilor (spre deosebire de GaussianBlur care
        topia furtunile sub pragul de detectie).
        
        Aceasta tehnica maximizeaza POD si CSI in evaluarile Pixel-Based pe orizonturi lungi,
        deoarece creeaza o 'umbrela' spatiala care acopera eroarea de pozitie a advectiei.
        """
        if step <= 2:
            return shifted_grown.copy()
        
        # Kernel-ul de dilatare creste liniar cu orizontul: la step=6 (30m) -> 3px,
        # la step=12 (1h) -> 5px, la step=24 (2h) -> 9px
        # Adaugam si incertitudinea Kalman a trackerului
        base_radius = max(1, int(base_uncertainty * 0.15 * step))
        base_radius = min(base_radius, 5)  # Cap la 5px pentru a nu umfla excesiv
        
        kernel_size = 2 * base_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilatare globala de baza - extinde toti pixelii de ploaie cu raza de baza
        shifted_grown_dilated = cv2.dilate(shifted_grown, kernel, iterations=1)
        
        if not valid_cells:
            return shifted_grown_dilated
        
        y_grid_local, x_grid_local = AdvectionEngine._get_grids((grid_h, grid_w))
        
        # Dilatare locala suplimentara pentru furtunile cu incertitudine Kalman mare
        for c in valid_cells:
            uncertainty = c.uncertainty_trace
            # Incertitudinea suplimentara fata de baza
            extra_px = int(0.1 * step + 0.03 * uncertainty)
            extra_px = min(extra_px, 4)  # Cap la 4px suplimentar
            
            if extra_px < 1:
                continue
            
            gamma = 0.8
            damping = 0.95
            term_a = (2*step - (1+gamma)*(1-gamma**step)/(1-gamma)) / (2*(1-gamma))
            term_v = (1 - damping**step) / (1 - damping)
            cx = c.centroid_x + c.v_x * term_v + c.a_x * term_a
            cy = c.centroid_y + c.v_y * term_v + c.a_y * term_a
            
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
            
            # Dilatare suplimentara pe patch-ul local al acestei furtuni
            local_ksize = 2 * extra_px + 1
            local_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (local_ksize, local_ksize))
            patch_orig = shifted_grown[y_slice, x_slice]
            patch_dilated_extra = cv2.dilate(patch_orig, local_kernel, iterations=1)
            
            # Blendare proportionala cu incertitudinea: furtunile nesigure se extind mai mult
            kalman_confidence = np.clip(10.0 / (10.0 + c.uncertainty_trace), 0.1, 0.9)
            extra_blend = 1.0 - kalman_confidence  # nesigure -> blend mai mare spre dilatare extra
            
            patch_base = shifted_grown_dilated[y_slice, x_slice]
            shifted_grown_dilated[y_slice, x_slice] = np.maximum(
                patch_base,
                patch_dilated_extra * extra_blend
            )
        
        return shifted_grown_dilated

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
        predicted_cells_dict = {}
        
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
            # Initialize EMA flow state with initial velocity
            c.flow_vec_smooth_x = c.v_x
            c.flow_vec_smooth_y = c.v_y
            c.predicted_centroid_x = c.centroid_x
            c.predicted_centroid_y = c.centroid_y
            
        original_cells_dict = {c.cell_id: c for c in valid_cells}
        
        for step in range(1, max_step + 1):
            # 1. Update Kinematics (Euler-Lagrange with Flow Forcing)
            for c in simulated_cells:
                if flow is not None:
                    # Eulerian Forcing
                    flow_coupling = 0.25 # Intensitatea cu care vântul afectează mișcarea
                    
                    # EROARE CRITICA REPARATA: flow-ul este o matrice statica de la T0.
                    # Daca citim la predicted_centroid, furtuna va iesi din zona de ploaie de la T0 si flow va fi 0, oprind furtuna.
                    # Trebuie sa citim la T0 centroid!
                    f_x, f_y = AdvectionEngine._sample_flow_at(flow, c.centroid_x, c.centroid_y)
                    
                    # Low-pass filter (EMA)
                    c.flow_vec_smooth_x = 0.8 * c.flow_vec_smooth_x + 0.2 * f_x
                    c.flow_vec_smooth_y = 0.8 * c.flow_vec_smooth_y + 0.2 * f_y
                    
                    # Accelerație generată de vânt
                    a_flow_x = (c.flow_vec_smooth_x - c.v_x) * flow_coupling
                    a_flow_y = (c.flow_vec_smooth_y - c.v_y) * flow_coupling
                    
                    # Viteza combina inertia, curgerea (flow) si acceleratia curbata Kalman (amortizata)
                    c.v_x += a_flow_x + c.a_x * (0.85 ** step)
                    c.v_y += a_flow_y + c.a_y * (0.85 ** step)
                
                # Integratorul (viteză -> poziție)
                c.predicted_centroid_x += c.v_x
                c.predicted_centroid_y += c.v_y
                
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
                    # Calcul pentru telemetrie (Phase 6)
                    E_diff = E_new / max(R_applied, 1e-6)  # reverse engineer E_diff
                    diffusion_delta = E_diff - c.E
                    reaction_gain = E_new - E_diff
                    relative_diffusion = diffusion_delta / max(c.E, 1e-6)
                    diffusion_fraction = abs(diffusion_delta) / (abs(diffusion_delta) + abs(reaction_gain) + 1e-6)
                    
                    diag = CellDiagnostics(
                        energy_before=c.E,
                        energy_after=E_new,
                        reaction_gain=reaction_gain,
                        diffusion_delta=diffusion_delta,
                        relative_diffusion=relative_diffusion,
                        diffusion_fraction=diffusion_fraction
                    )
                    
                    updates.append((E_new, dE_new, R_applied, diag))
                    
                for i, c in enumerate(simulated_cells):
                    E_new, dE_new, R_applied, diag = updates[i]
                    # Pass the true energy delta to lifecycle so it can trigger DISSIPATION properly
                    c.lifecycle_phase = lifecycle(c.E, E_new - c.E)
                    c.E = max(E_new, 1e-6)
                    c.dE = dE_new
                    c.cumulative_R *= R_applied
                    c.diagnostics = diag
            
            blended_flow_x, blended_flow_y = AdvectionEngine._blend_kinematics(
                flow_x, flow_y, simulated_cells, grid_h, grid_w, x_grid, y_grid
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
                (grid_h, grid_w), simulated_cells, original_cells_dict, flow
            )
            
            shifted_grown = shifted * growth_mask
            float_preds[step] = shifted_grown
            
            # Phase 6: Păstrăm snapshot-ul obiectelor prezise pentru FAR Inspector
            # Omit celulele care s-au disipat (altfel inspectorul le vede ca BAD_ADVECTION)
            predicted_cells_dict[step] = [c.clone() for c in simulated_cells if c.lifecycle_phase != "DISSIPATION"]
            
            if step in horizon_map:
                name = horizon_map[step]
                
                # Restore SPROG Diffusion. It is necessary for long-term Eulerian metrics
                # to hedge against spatial displacement errors.
                shifted_grown_blurred = AdvectionEngine._apply_sprog_diffusion(
                    shifted_grown, step, valid_cells, grid_h, grid_w, base_uncertainty
                )
                
                base_mask = (shifted_grown_blurred >= RAIN_THRESHOLD_TRACKING).astype(np.float32)
                sparse_preds[name] = sp.csr_matrix(base_mask)
            
        return sparse_preds, float_preds, predicted_cells_dict
