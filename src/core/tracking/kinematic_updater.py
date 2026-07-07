from __future__ import annotations

from src.core.tracking.storm_filter import StormFilter
from src.core.domain import StormCell

class KinematicUpdater:
    """Manages the Kalman filter bank and kinematic updates."""
    def __init__(self):
        self._kalman_bank: dict[str, StormFilter] = {}

    def reset(self) -> None:
        self._kalman_bank.clear()

    def predict_all(self) -> None:
        for kf in self._kalman_bank.values():
            kf.predict()

    def update_cell(self, cell_id: str, c_cell: StormCell, tracked_cell: StormCell, c_area: float) -> None:
        kf = self._kalman_bank[cell_id]
        kf.update(c_cell.centroid_x, c_cell.centroid_y)

        tracked_cell.v_x = kf.v_x
        tracked_cell.v_y = kf.v_y
        tracked_cell.uncertainty_trace = kf.positional_uncertainty

        # Centroid prediction based on Constant Velocity
        predicted_centroid_x = kf.x + kf.v_x
        predicted_centroid_y = kf.y + kf.v_y
        tracked_cell.predicted_centroid_x = float(predicted_centroid_x)
        tracked_cell.predicted_centroid_y = float(predicted_centroid_y)

    def register_new_cell(self, cell_id: str, c_cell: StormCell, c_area: float, inherited_vx: float, inherited_vy: float) -> None:
        self._kalman_bank[cell_id] = StormFilter(
            initial_y=c_cell.centroid_y, initial_x=c_cell.centroid_x,
            initial_vy=inherited_vy, initial_vx=inherited_vx,
            initial_area=float(c_area), initial_d_area=0.0
        )

    def get_prior_prediction(self, cell_id: str) -> tuple[float, float]:
        kf = self._kalman_bank[cell_id]
        return kf.x, kf.y

    def get_filter(self, cell_id: str) -> StormFilter | None:
        return self._kalman_bank.get(cell_id)

    def is_tracked(self, cell_id: str) -> bool:
        return cell_id in self._kalman_bank

    def cleanup_inactive(self, active_ids: set[str]) -> None:
        for old_id in list(self._kalman_bank.keys()):
            if old_id not in active_ids:
                del self._kalman_bank[old_id]
