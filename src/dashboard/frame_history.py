from src.core.constants import HORIZON_NAMES, HORIZON_STEPS


class FrameHistory:
    """Accumulates total volume and global metric series (historic mode)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.last_frame_idx = -1
        self.total_map_mm = 0.0
        self.frames_processed = 0
        self.last_result = None
        
        # Store the history of instantaneous values
        self.true_volumes = []
        
        # Estimated accumulated volume across multiple horizons
        self.predicted_volume_accumulation = {horizon: 0.0 for horizon in HORIZON_NAMES}
        
        # Store predicted and instantaneous cumulative values
        self.pred_volumes_acc = {horizon: [] for horizon in HORIZON_NAMES}
        self.pred_volumes = {horizon: [] for horizon in HORIZON_NAMES}
        
        # Event Reliability (Catchment level)
        self.thresholds = [1.0, 5.0]
        self.reliability_counts = {}
        for t in self.thresholds:
            self.reliability_counts[t] = {
                horizon: {"hits": 0, "fa": 0, "miss": 0, "cr": 0, "abs_err_sum": 0.0}
                for horizon in HORIZON_NAMES
            }

    def accumulate(self, result) -> None:
        self.total_map_mm += result.roi_map_mm
        self.frames_processed += 1
        self.last_result = result
        
        self.true_volumes.append(result.roi_map_mm)
        
        # Save cumulative predictions 
        for horizon in HORIZON_NAMES:
            if hasattr(result, "predicted_volumes_horizons") and result.predicted_volumes_horizons:
                val = result.predicted_volumes_horizons.get(horizon, 0.0)
            else:
                val = 0.0
            self.pred_volumes_acc[horizon].append(val)
                
        # Save instantaneous predictions
        if hasattr(result, "instant_predicted_volumes") and result.instant_predicted_volumes:
            for horizon in HORIZON_NAMES:
                val = result.instant_predicted_volumes.get(horizon, 0.0)
                self.predicted_volume_accumulation[horizon] += val
                self.pred_volumes[horizon].append(val)
        else:
            for horizon in HORIZON_NAMES:
                self.pred_volumes[horizon].append(0.0)
                
        # Calculate Catchment Event Reliability on the fly using Cumulative Windows
        # IMPORTANT: The values must be IDENTICAL to the target_step from frame_processor.py horizons
        for horizon, steps in HORIZON_STEPS.items():
            if len(self.true_volumes) > steps:
                # Cumulative reality (e.g., sum of rain from the last 1h)
                # Extract the last 'steps' elements and calculate the sum
                actual_acc_val = sum(self.true_volumes[-steps:])
                
                # The prediction made 'steps' frames ago regarding the CUMULATIVE amount over those 'steps' frames
                pred_acc_val = self.pred_volumes_acc[horizon][-1 - steps]
                
                for t in self.thresholds:
                    pred_event = pred_acc_val >= t
                    actual_event = actual_acc_val >= t
                    
                    counts = self.reliability_counts[t][horizon]
                    if pred_event and actual_event:
                        counts["hits"] += 1
                        # Symmetric percentage quantitative error over the interval (sMAPE)
                        denominator = pred_acc_val + actual_acc_val
                        if denominator > 0:
                            counts["abs_err_sum"] += 2.0 * abs(pred_acc_val - actual_acc_val) / denominator * 100.0
                    elif pred_event and not actual_event:
                        counts["fa"] += 1
                    elif not pred_event and actual_event:
                        counts["miss"] += 1
                    else:
                        counts["cr"] += 1

    def volume_sums(self, horizon: str) -> tuple[float, float]:
        """Returns real/predicted MAP aligned to the horizon lead time."""
        steps = HORIZON_STEPS[horizon]
        if len(self.true_volumes) > steps and len(self.pred_volumes_acc[horizon]) > steps:
            actual_sum = 0.0
            pred_sum = 0.0
            for current_idx in range(steps, len(self.true_volumes)):
                actual_sum += sum(self.true_volumes[current_idx - steps + 1:current_idx + 1])
                pred_sum += self.pred_volumes_acc[horizon][current_idx - steps]
            return actual_sum, pred_sum
        return (
            self.total_map_mm,
            self.predicted_volume_accumulation.get(horizon, 0.0),
        )

    def get_reliability_metrics(self) -> dict[float, dict[str, dict[str, float]]]:
        """Returns Catchment level POD, FAR and CMAE for each threshold and horizon."""
        metrics = {}
        for t in self.thresholds:
            metrics[t] = {}
            for horizon, counts in self.reliability_counts[t].items():
                hits = counts["hits"]
                fa = counts["fa"]
                miss = counts["miss"]
                abs_err = counts["abs_err_sum"]
                
                pod = hits / (hits + miss) if (hits + miss) > 0 else 0.0
                far = fa / (hits + fa) if (hits + fa) > 0 else 0.0
                cmae = abs_err / hits if hits > 0 else 0.0
                
                metrics[t][horizon] = {"pod": pod * 100.0, "far": far * 100.0, "cmae": cmae}
        return metrics

    def get_latest_verification(self) -> dict[str, dict[str, float]]:
        """Returns the actual vs predicted accumulation for the most recently completed horizons."""
        verification = {}
        for horizon, steps in HORIZON_STEPS.items():
            if len(self.true_volumes) >= steps:
                actual_acc_val = sum(self.true_volumes[-steps:])
                pred_acc_val = self.pred_volumes_acc[horizon][-steps] if len(self.pred_volumes_acc[horizon]) >= steps else 0.0
                error_pct = ((pred_acc_val - actual_acc_val) / actual_acc_val * 100.0) if actual_acc_val > 0.1 else 0.0
                verification[horizon] = {
                    "actual_mm": actual_acc_val,
                    "predicted_mm": pred_acc_val,
                    "error_pct": error_pct
                }
            else:
                verification[horizon] = None
        return verification
