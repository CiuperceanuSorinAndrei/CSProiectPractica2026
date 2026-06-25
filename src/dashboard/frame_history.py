import numpy as np


class FrameHistory:
    """Acumuleaza volumul total si seriile de metrici globale (modul istoric)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.last_frame_idx = -1
        self.total_volume_m3 = 0.0
        self.csi: list[float] = []
        self.far: list[float] = []
        self.pod: list[float] = []

    def accumulate(self, result) -> None:
        self.total_volume_m3 += result.roi_volume_m3
        if result.global_csi is not None:
            self.csi.append(result.global_csi)
            self.far.append(result.global_far)
            self.pod.append(result.global_pod)

    def averages(self) -> tuple[float, float, float]:
        def avg(xs):
            return float(np.mean(xs)) if xs else 0.0
        return avg(self.csi), avg(self.far), avg(self.pod)
