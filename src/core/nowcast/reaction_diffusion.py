import numpy as np

def spatial_diffusion(E: float, neighbors_E: np.ndarray, gamma: float = 0.15) -> float:
    """Bounded diffusion (mass conserving discrete Laplacian)"""
    neighbor_mean = neighbors_E.mean() if len(neighbors_E) > 0 else E
    # Strict mass conservation: diffusion is a complete linear combination
    # without asymmetric eliminations.
    return (1 - gamma) * E + gamma * neighbor_mean

def sigmoid(x: float) -> float:
    x = np.clip(x, -30, 30)
    return float(1 / (1 + np.exp(-x)))

def reaction(E: float, dE: float, alpha_g: float = 1.2, alpha_d: float = 1.2, beta: float = 1.0) -> float:
    # Scale Invariance: dE must be fractional relative to E.
    # Energy cannot be negative; clamp it.
    E_safe = max(float(E), 0.0)
    dE_frac = dE / (E_safe + 1e-6)
    
    # Stable inertia, prevent collapse when E < 1.0
    base_inertia = 0.9 + 0.1 * (E_safe / (E_safe + 1.0))
    
    if dE_frac >= 0:
        R = base_inertia + alpha_g * dE_frac
        return min(R, 1.05)
        
    # Decay regime
    R = base_inertia * np.exp(-alpha_d * abs(dE_frac))
    return max(R, 0.95)

def update_energy(E: float, neighbors_E: np.ndarray, dE_old: float,
                  gamma: float = 0.15,
                  alpha_g: float = 1.5,
                  alpha_d: float = 1.8,
                  beta: float = 1.0) -> tuple[float, float, float]:
    E_diff = spatial_diffusion(E, neighbors_E, gamma)
    
    diff_term = E_diff - E
    
    # High momentum retention (reduced amnesia). Storms remember
    # their growth/dissipation trend for a longer horizon (0.95 instead of 0.7).
    dE_input = 0.95 * dE_old + 0.05 * diff_term
    
    R = reaction(E_diff, dE_input, alpha_g, alpha_d, beta)
    
    E_new = max(E_diff * R, 0.0)
    
    # IMPORTANT: We return dE_input as the new momentum (which decays naturally),
    # not E_new - E, otherwise we create an infinite positive feedback loop when R > 1.0.
    # We also return R (the pure reaction factor) to use for the volumetric mask.
    return E_new, dE_input, R

def lifecycle(E: float, dE: float, collapse_threshold: float = 0.2) -> str:
    # Scale Invariance: relative collapse rate
    E_safe = max(float(E), 0.0)
    collapse = max(0.0, -dE) / (E_safe + 1e-6)

    # If the cell is collapsing faster than 20% of its energy per step, it's dying.
    # Also, if its energy is extremely small and dropping, kill it immediately to prevent lingering.
    if collapse > collapse_threshold or (E < 0.8 and dE < 0):
        return "DISSIPATION"
    return "ACTIVE"
