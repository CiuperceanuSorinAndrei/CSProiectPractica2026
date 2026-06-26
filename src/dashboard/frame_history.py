import numpy as np

class FrameHistory:
    """Acumuleaza volumul total si seriile de metrici globale (modul istoric)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.last_frame_idx = -1
        self.total_volume_m3 = 0.0
        self.frames_processed = 0
        
        # Stocam istoricul valorilor instantanee
        self.true_volumes = []
        self.pred_volumes = {"30m": [], "1h": [], "2h": []}
        
        # Volum acumulat estimat pe mai multe orizonturi (legacy, pentru afisare rapida)
        self.predicted_volume_accumulation = {"30m": 0.0, "1h": 0.0, "2h": 0.0}
        
        # Metric history (CSI, FAR, POD, FSS)
        self.metrics_history = {
            "csi": [],
            "far": [],
            "pod": [],
            "fss": []
        }

    def accumulate(self, result) -> None:
        self.total_volume_m3 += result.roi_volume_m3
        self.frames_processed += 1
        
        self.true_volumes.append(result.roi_volume_m3)
        
        if hasattr(result, "instant_predicted_volumes") and result.instant_predicted_volumes:
            for horizon in ["30m", "1h", "2h"]:
                val = result.instant_predicted_volumes.get(horizon, 0.0)
                self.predicted_volume_accumulation[horizon] += val
                self.pred_volumes[horizon].append(val)
                
        # Adaugam metricile pentru a putea face mediile
        if result.global_csi:
            self.metrics_history["csi"].append(result.global_csi.copy())
            self.metrics_history["far"].append(result.global_far.copy())
            self.metrics_history["pod"].append(result.global_pod.copy())
            self.metrics_history["fss"].append(result.global_fss.copy())
