from __future__ import annotations
import numpy as np
from src.core.domain import StormCell

class CellLifecycleManager:
    """Gestioneaza istoricul si tendintele de crestere ale celulelor convective."""
    
    @staticmethod
    def transfer_history(c_cell: StormCell, tracked_cell: StormCell, best_match: StormCell | None, c_area: float) -> None:
        if best_match:
            tracked_cell.age_frames = getattr(best_match, 'age_frames', 1) + 1
            tracked_cell.centroid_history = list(best_match.centroid_history or [])
            tracked_cell.area_history = list(best_match.area_history or [])
            tracked_cell.cell_history = list(best_match.cell_history or [])
            
            tracked_cell.centroid_history.append((float(c_cell.centroid_y), float(c_cell.centroid_x)))
            tracked_cell.centroid_history = tracked_cell.centroid_history[-6:]
            tracked_cell.area_history.append(int(c_area))
            tracked_cell.area_history = tracked_cell.area_history[-6:]
            tracked_cell.cell_history.append({
                "centroid_y": float(c_cell.centroid_y),
                "centroid_x": float(c_cell.centroid_x),
                "area_pixels": int(c_area),
            })
            tracked_cell.cell_history = tracked_cell.cell_history[-6:]
        else:
            tracked_cell.age_frames = 1
            tracked_cell.centroid_history = list(c_cell.centroid_history or [])
            tracked_cell.area_history = list(c_cell.area_history or [])
            tracked_cell.cell_history = list(c_cell.cell_history or [])
            if not tracked_cell.centroid_history:
                tracked_cell.centroid_history = [(float(c_cell.centroid_y), float(c_cell.centroid_x))]
            if not tracked_cell.area_history:
                tracked_cell.area_history = [int(c_area)]
            if not tracked_cell.cell_history:
                tracked_cell.cell_history = [{
                    "centroid_y": float(c_cell.centroid_y),
                    "centroid_x": float(c_cell.centroid_x),
                    "area_pixels": int(c_area),
                }]

    # evaluate_lifecycle removed (using dynamic E/dE reaction-diffusion now)
    def compute_area_trend(tracked_cell: StormCell) -> float:
        if len(tracked_cell.area_history) >= 2:
            area_deltas = [
                max(tracked_cell.area_history[idx], 1) / max(tracked_cell.area_history[idx - 1], 1)
                for idx in range(1, len(tracked_cell.area_history))
            ]
            raw_area_trend = float(np.prod(area_deltas[-3:]) ** (1.0 / len(area_deltas[-3:])))
        else:
            raw_area_trend = 1.0

        if len(tracked_cell.cell_history) >= 3:
            recent_areas = [item["area_pixels"] for item in tracked_cell.cell_history[-3:]]
            recent_area_trend = float(recent_areas[-1] / max(recent_areas[0], 1))
        else:
            recent_area_trend = raw_area_trend

        # Area trend este strict izolat de volum (morfologie pura)
        return float(recent_area_trend)
