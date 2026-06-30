import numpy as np

from src.diagnostics.false_alarm_inspector import FalseAlarmInspector

class FrameHistory:
    """Acumuleaza volumul total si seriile de metrici globale (modul istoric)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.last_frame_idx = -1
        self.total_volume_m3 = 0.0
        self.frames_processed = 0
        self.last_result = None
        
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
        
        # Phase 6: FAR Inspector data
        # list of (frame_idx, horizon_name, predicted_cells, observed_cells)
        self.far_episode_data = []
        self.far_episode_data_raw = []
        self.far_inspector = FalseAlarmInspector(area_conservation_tolerance=0.20)

    def accumulate(self, result) -> None:
        self.total_volume_m3 += result.roi_volume_m3
        self.frames_processed += 1
        self.last_result = result
        
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

        # Phase 6: Adaugam observatiile pentru tracker history
        if result.raw_tracked_cells:
            self.far_inspector.collect_observations(self.frames_processed, result.raw_tracked_cells)
            
        # Adaugam perechile de predicții (din trecut) și observații curente
        # Horizons are 2 (30m), 4 (1h), 8 (2h)
        horizons_map = {2: "30m", 4: "1h", 8: "2h"}
        
        for steps_back, h_name in horizons_map.items():
            # If we have enough history, the prediction from `frames_processed - steps_back` is validating NOW
            if len(self.far_episode_data_raw) >= steps_back:
                past_result = self.far_episode_data_raw[-steps_back]
                if past_result and past_result.raw_predicted_cells:
                    # advection_engine indexes predicted_cells_dict by integer step
                    past_preds = past_result.raw_predicted_cells.get(steps_back, [])
                    curr_obs = result.raw_tracked_cells or []
                    self.far_episode_data.append((self.frames_processed, h_name, past_preds, curr_obs))
                    
        self.far_episode_data_raw.append(result)

    def generate_far_report(self):
        """Called at the end of the historical simulation to run Hungarian matching and print the report."""
        if self.far_episode_data:
            self.far_inspector.evaluate_episode(self.far_episode_data)
            self.far_inspector.generate_report()
