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
    
    # --- Reaction-Diffusion (Phase 4) Parameters ---
    RD_GAMMA: float = 0.15
    RD_ALPHA_G: float = 2.0
    RD_ALPHA_D: float = 2.5
    RD_BETA: float = 1.0
    
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

    # --- Diagnostics & Inspector ---
    INSPECTOR_BASE_RADIUS: float = 15.0
    INSPECTOR_MAX_RADIUS_MULT: float = 2.0

config = AlgorithmsConfig()
