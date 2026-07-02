from __future__ import annotations

import numpy as np

from src.core.domain import StormCell

class KinematicsEngine:
    """Modul responsabil exclusiv cu rezolvarea cinematicii celulelor de furtuna
    (integrarea traiectoriei Lagrangiene rigide bazate pe centroid).
    """

    @staticmethod
    def update_positions(simulated_cells: list[StormCell], step: int):
        """Integreaza viteza in pozitie pentru advectie rigida."""
        for c in simulated_cells:
            # Integratorul rigid (viteză -> poziție) bazat strict pe Kalman 4D
            c.predicted_centroid_x += c.v_x
            c.predicted_centroid_y += c.v_y

