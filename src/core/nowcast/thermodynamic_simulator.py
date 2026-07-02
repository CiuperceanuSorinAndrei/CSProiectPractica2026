import numpy as np
from src.core.domain import StormCell, CellDiagnostics
from src.core.nowcast.reaction_diffusion import update_energy, lifecycle

class ThermodynamicSimulator:
    def simulate_step(self, simulated_cells: list[StormCell], coords: np.ndarray) -> None:
        if not simulated_cells:
            return
            
        for idx_c, c in enumerate(simulated_cells):
            coords[idx_c, 0] = c.predicted_centroid_x
            coords[idx_c, 1] = c.predicted_centroid_y
            
        if len(coords) > 1:
            from scipy.spatial import cKDTree
            tree = cKDTree(coords)
            all_neighbors = tree.query_ball_point(coords, r=50.0)
        else:
            all_neighbors = [[]]
        
        updates = []
        for i, c in enumerate(simulated_cells):
            if len(coords) > 1:
                # Excludem celula curenta
                neighbor_indices = [idx for idx in all_neighbors[i] if idx != i]
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
            phase = lifecycle(c.E, E_new - c.E)
            c.update_thermodynamics(E_new, dE_new, R_applied, diag, phase)
