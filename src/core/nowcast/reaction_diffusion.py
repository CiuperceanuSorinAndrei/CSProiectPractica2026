import numpy as np

def spatial_diffusion(E: float, neighbors_E: np.ndarray, gamma: float = 0.15) -> float:
    """Bounded diffusion (mass conserving discrete Laplacian)"""
    neighbor_mean = neighbors_E.mean() if len(neighbors_E) > 0 else E
    # Conservare stricta a masei: difuzia este o combinatie liniara completa,
    # fara eliminari asimetrice.
    return (1 - gamma) * E + gamma * neighbor_mean

def sigmoid(x: float) -> float:
    x = np.clip(x, -30, 30)
    return float(1 / (1 + np.exp(-x)))

def reaction(E: float, dE: float, alpha_g: float = 1.5, alpha_d: float = 1.8, beta: float = 1.0) -> float:
    # Scale Invariance: dE must be fractional relative to E
    dE_frac = dE / (abs(E) + 1e-6)
    
    # Inertie stabila, nu prabusim cand E < 1.0
    base_inertia = 0.9 + 0.1 * (abs(E) / (abs(E) + 1.0))
    
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
    
    # Momentum cu retentie crescuta (amnezie redusa). Furtunile isi amintesc 
    # trendul de crestere/disipare pentru un orizont mai lung (0.95 in loc de 0.7).
    dE_input = 0.95 * dE_old + 0.05 * diff_term
    
    R = reaction(E_diff, dE_input, alpha_g, alpha_d, beta)
    
    E_new = E_diff * R
    
    # IMPORTANT: Returnam dE_input ca noul momentum (care decade natural cu 0.7), 
    # nu E_new - E, altfel cream un infinite positive feedback loop cand R > 1.0!
    # Returnam de asemenea si R (factorul pur de reactie) pentru a-l folosi la masca volumetrica.
    return E_new, dE_input, R

def lifecycle(E: float, dE: float, collapse_threshold: float = 0.2) -> str:
    # Scale Invariance: relative collapse rate
    collapse = max(0.0, -dE) / (abs(E) + 1e-6)

    # If the cell is collapsing faster than 20% of its energy per step, it's dying.
    # Also, if its energy is extremely small and dropping, kill it immediately to prevent lingering.
    if collapse > collapse_threshold or (E < 0.8 and dE < 0):
        return "DISSIPATION"
    return "ACTIVE"
