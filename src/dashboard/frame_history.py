import numpy as np


class FrameHistory:
    """Acumuleaza volumul total si seriile de metrici globale (modul istoric)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.last_frame_idx = -1
        self.total_volume_m3 = 0.0
        self.total_predicted_volume_m3 = 0.0
        # Dictionare care pastreaza array-ul de scoruri per orizont
        self.csi: dict[str, list[float]] = {"15m": [], "1h": [], "2h": []}
        self.far: dict[str, list[float]] = {"15m": [], "1h": [], "2h": []}
        self.pod: dict[str, list[float]] = {"15m": [], "1h": [], "2h": []}

    def accumulate(self, result) -> None:
        self.total_volume_m3 += result.roi_volume_m3
        self.total_predicted_volume_m3 += result.predicted_roi_volume_m3
        if result.global_csi:
            for horizon in ["15m", "1h", "2h"]:
                if horizon in result.global_csi:
                    self.csi[horizon].append(result.global_csi[horizon])
                    self.far[horizon].append(result.global_far[horizon])
                    self.pod[horizon].append(result.global_pod[horizon])

    def averages(self, horizon: str = "15m") -> tuple[float | None, float | None, float | None]:
        def avg(xs):
            return float(np.mean(xs)) if xs else None
        return avg(self.csi[horizon]), avg(self.far[horizon]), avg(self.pod[horizon])
