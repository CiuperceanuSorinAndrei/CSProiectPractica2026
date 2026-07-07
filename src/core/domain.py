from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np


@dataclass
class StormCell:
    """Domain object for a storm cell."""
    id: int = 0
    cell_id: str = ""
    is_tracked: bool = False
    
    age_frames: int = 1
    orphan_age: int = 0
    lifecycle_phase: str = "BIRTH"
    
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    geo_lon: float = 0.0
    geo_lat: float = 0.0
    area_pixels: int = 0
    volume: float = 0.0
    max_intensity: float = 0.0
    mean_intensity: float = 0.0
    orientation: float = 0.0
    major_axis_length: float = 0.0
    minor_axis_length: float = 0.0
    coords: np.ndarray | list = field(default_factory=list)
    _cached_mask: np.ndarray | None = field(default=None, repr=False)
    
    v_x: float = 0.0
    v_y: float = 0.0
    
    predicted_centroid_x: float = 0.0
    predicted_centroid_y: float = 0.0
    uncertainty_trace: float = 0.0
    
    flow_vec_smooth_x: float = 0.0
    flow_vec_smooth_y: float = 0.0
    
    volume_trend: float = 1.0
    predicted_area_pixels: int = 0
    predicted_coords: np.ndarray | list | None = field(default=None, repr=False)
    
    prediction_error_pixels: float = 0.0
    size_error_pixels: int = 0
    size_error_percent: float = 0.0
    
    centroid_history: list[tuple[float, float]] = field(default_factory=list)
    area_history: list[int] = field(default_factory=list)
    cell_history: list[dict[str, Any]] = field(default_factory=list)
    
    def as_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    def initialize_simulation_state(self) -> None:
        """Initialize state for nowcast advection."""
        self.flow_vec_smooth_x = self.v_x
        self.flow_vec_smooth_y = self.v_y
        self.predicted_centroid_x = self.centroid_x
        self.predicted_centroid_y = self.centroid_y

    def clone(self) -> StormCell:
        """Create a fast shallow copy, making new lists for histories to prevent mutation."""
        import dataclasses
        kwargs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        for field in ('centroid_history', 'area_history', 'cell_history'):
            if kwargs[field] is not None:
                kwargs[field] = list(kwargs[field])
        return StormCell(**kwargs)

