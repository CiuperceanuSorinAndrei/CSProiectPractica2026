import numpy as np

def spatial_diffusion(E: float, neighbors_E: np.ndarray, gamma: float = 0.15) -> float:
    # Bounded diffusion.
    neighbor_mean = neighbors_E.mean() if len(neighbors_E) > 0 else E
    # Mass-conserving linear combination.
    return (1 - gamma) * E + gamma * neighbor_mean



def reaction(E: float, dE: float, alpha_g: float = 0.2, alpha_d: float = 0.2, beta: float = 1.0) -> float:
    # Fractional relative energy change.
    E_safe = max(float(E), 0.0)
    dE_frac = dE / (E_safe + 1e-6)
    
    # Stable inertia.
    base_inertia = 0.9 + 0.1 * (E_safe / (E_safe + 1.0))
    
    if dE_frac >= 0:
        R = base_inertia + alpha_g * dE_frac
        return min(R, 1.05)
        
    # Decay regime
    R = base_inertia * np.exp(-alpha_d * abs(dE_frac))
    return max(R, 0.95)

def update_energy(E: float, neighbors_E: np.ndarray, dE_old: float,
                  gamma: float = 0.15,
                  alpha_g: float = 0.2,
                  alpha_d: float = 0.2,
                  beta: float = 1.0) -> tuple[float, float, float]:
    E_diff = spatial_diffusion(E, neighbors_E, gamma)
    
    diff_term = E_diff - E
    
    # High momentum retention.
    dE_input = 0.95 * dE_old + 0.05 * diff_term
    
    R = reaction(E_diff, dE_input, alpha_g, alpha_d, beta)
    
    E_new = max(E_diff * R, 0.0)
    
    # Return decaying momentum to prevent infinite feedback loops.
    return E_new, dE_input, R

def lifecycle(E: float, dE: float, collapse_threshold: float = 0.2) -> str:
    # Relative collapse rate.
    E_safe = max(float(E), 0.0)
    collapse = max(0.0, -dE) / (E_safe + 1e-6)

    # Dissipate rapidly collapsing or tiny dying cells.
    if collapse > collapse_threshold or (E < 0.8 and dE < 0):
        return "DISSIPATION"
    return "ACTIVE"
