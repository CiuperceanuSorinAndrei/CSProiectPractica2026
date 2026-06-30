import numpy as np

def spatial_diffusion(E: float, neighbors_E: np.ndarray, gamma: float = 0.15) -> float:
    """Bounded diffusion (convex combination)"""
    neighbor_mean = neighbors_E.mean() if len(neighbors_E) > 0 else E
    return (1 - gamma) * E + gamma * neighbor_mean

def sigmoid(x: float) -> float:
    x = np.clip(x, -30, 30)
    return float(1 / (1 + np.exp(-x)))

def reaction(E: float, dE: float, alpha_g: float = 1.5, alpha_d: float = 1.8, beta: float = 1.0) -> float:
    # Scale Invariance: dE must be fractional relative to E
    dE_frac = dE / (abs(E) + 1e-6)
    
    # Inerția de bază pentru un sistem stabil (dE_frac = 0). Ex: beta=1.0 -> 0.98
    base_inertia = 1.0 - 0.02 * beta
    
    if dE_frac >= 0:
        # Growth regime: creștere proporțională cu derivata (scale invariant)
        return base_inertia + alpha_g * dE_frac
        
    # Decay regime: colaps exponențial
    return base_inertia * np.exp(-alpha_d * abs(dE_frac))

def update_energy(E: float, neighbors_E: np.ndarray, dE_old: float,
                  gamma: float = 0.15,
                  alpha_g: float = 1.5,
                  alpha_d: float = 1.8,
                  beta: float = 1.0) -> tuple[float, float, float]:
    E_diff = spatial_diffusion(E, neighbors_E, gamma)
    
    # 0.4 inertial dE_old. Momentum-ul termodinamic are nevoie de inertie scazuta
    # deoarece Actioneaza ca un multiplicator geometric. La 0.7, cresterea devine 
    # exponential exploziva. La 0.4, o furtuna in crestere se satureaza gratios.
    dE_input = 0.4 * dE_old
    
    R = reaction(E_diff, dE_input, alpha_g, alpha_d, beta)
    
    E_new = E_diff * R
    
    # IMPORTANT: Returnam dE_input ca noul momentum (care decade natural cu 0.7), 
    # nu E_new - E, altfel cream un infinite positive feedback loop cand R > 1.0!
    # Returnam de asemenea si R (factorul pur de reactie) pentru a-l folosi la masca volumetrica.
    return E_new, dE_input, R

def lifecycle(E: float, dE: float, collapse_threshold: float = 0.3, noise_ratio: float = 1.2) -> str:
    collapse = max(0.0, -dE) / (abs(E) + 1e-6)
    instability = abs(dE) / (abs(E) + 1e-6)

    # Colaps dominant vs zgomot 
    if collapse > collapse_threshold and collapse > instability * noise_ratio:
        return "DISSIPATION"
    return "ACTIVE"
