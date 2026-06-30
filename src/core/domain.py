from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np

@dataclass
class StormCell:
    """Modeleaza o celula de furtuna (Domain Object) inlocuind 'Primitive Obsession' (dict)."""
    # Identificare
    id: int = 0
    cell_id: str = ""
    is_tracked: bool = False
    
    # Lifecycle (Termodinamica formei)
    age_frames: int = 1
    lifecycle_phase: str = "BIRTH"
    
    # Pozitionare si Morfologie curenta
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    geo_lon: float = 0.0
    geo_lat: float = 0.0
    area_pixels: int = 0
    volume: float = 0.0
    max_intensity: float = 0.0
    mean_intensity: float = 0.0
    coords: np.ndarray | list = field(default_factory=list)
    _cached_mask: np.ndarray | None = field(default=None, repr=False)
    
    # Cinematica (Kalman 8D)
    v_x: float = 0.0
    v_y: float = 0.0
    a_x: float = 0.0
    a_y: float = 0.0
    
    # Predictii Kalman
    predicted_centroid_x: float = 0.0
    predicted_centroid_y: float = 0.0
    predicted_area_kalman: float = 1.0
    d_area_kalman: float = 0.0
    dd_area_kalman: float = 0.0
    uncertainty_trace: float = 0.0
    
    # Phase 4 Energetics (Reaction-Diffusion)
    E: float = 0.0
    dE: float = 0.0
    
    # Trenduri si Predictii Avansate
    volume_trend: float = 1.0
    predicted_area_pixels: int = 0
    predicted_mask: np.ndarray | None = field(default=None, repr=False)
    
    # Metrici de performanta (Errors)
    prediction_error_pixels: float = 0.0
    size_error_pixels: int = 0
    size_error_percent: float = 0.0
    
    # Istoric
    centroid_history: list[tuple[float, float]] = field(default_factory=list)
    area_history: list[int] = field(default_factory=list)
    cell_history: list[dict[str, Any]] = field(default_factory=list)
    
    def as_dict(self) -> dict[str, Any]:
        """Convertire sigura in dict pentru Dash/JSON (DTO Adapter)."""
        return asdict(self)

    def clone(self) -> StormCell:
        """Creeaza o copie sigura si rapida, partajand matricele numpy fara deepcopy."""
        import dataclasses
        kwargs = {}
        for f in dataclasses.fields(self):
            name = f.name
            if name in ('coords', '_cached_mask', 'predicted_mask'):
                # Shallow copy (reference) pentru performanta si RAM
                kwargs[name] = getattr(self, name)
            elif name in ('centroid_history', 'area_history', 'cell_history'):
                # Explicit list copy for histories to prevent mutation side-effects
                val = getattr(self, name)
                kwargs[name] = list(val) if val is not None else []
            else:
                kwargs[name] = getattr(self, name)
        return StormCell(**kwargs)

