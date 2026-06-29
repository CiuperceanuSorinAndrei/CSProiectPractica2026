"""Configurări și Magic Numbers specifice algoritmilor matematici și cinematici."""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class AlgorithmsConfig:
    # --- Kalman Filter Parameters ---
    KALMAN_GAMMA_SINGER: float = 0.8
    KALMAN_VAR_POS_MANEUVER: float = 0.05
    KALMAN_VAR_AREA_MANEUVER: float = 0.01
    KALMAN_R_POS: float = 5.0
    KALMAN_R_AREA: float = 0.2
    
    # --- Advection & Growth Parameters ---
    ADV_TAU_GROWTH: float = 3.0    # Număr de pași pentru creștere la asimptotă
    ADV_TAU_DECAY: float = 4.0     # Număr de pași pentru descreștere la asimptotă
    ADV_MAX_GROWTH_LIMIT: float = 1.2
    ADV_CLIMATOLOGICAL_DECAY_RATE: float = 0.15
    
    # --- Thermodynamic Lifecycle (Phase 2.5/3) ---
    ENABLE_THERMODYNAMIC_DECAY: bool = True
    DECAY_MODEL: str = "piecewise"
    BIRTH_MAX_MULTIPLIER: float = 1.3
    
    DECAY_CURVES: dict[str, list[float]] = field(default_factory=lambda: {
        "BIRTH":       [1.00, 1.00, 1.00, 0.99, 0.98, 0.96, 0.94, 0.92, 0.90],
        "MATURITY":    [1.00, 1.00, 1.00, 0.97, 0.93, 0.88, 0.82, 0.76, 0.70],
        "DISSIPATION": [1.00, 1.00, 0.95, 0.88, 0.78, 0.68, 0.58, 0.50, 0.45]
    })
    
    # --- S-PROG Diffusion ---
    SPROG_BASE_UNCERTAINTY_WEIGHT: float = 0.1
    SPROG_MIN_SIGMA: float = 0.2
    SPROG_MAX_SIGMA: float = 8.0

    # --- Matcher Weights ---
    MATCHER_MAX_COST: float = 500.0
    MATCHER_DIST_WEIGHT: float = 1.0
    MATCHER_AREA_WEIGHT: float = 0.5
    MATCHER_VOL_WEIGHT: float = 0.5
    MATCHER_IOU_WEIGHT: float = 1.5

config = AlgorithmsConfig()
