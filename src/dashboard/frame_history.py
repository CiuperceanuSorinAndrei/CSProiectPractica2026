from src.core.constants import HORIZON_NAMES, HORIZON_STEPS


class FrameHistory:
    """Acumuleaza volumul total si seriile de metrici globale (modul istoric)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.last_frame_idx = -1
        self.total_map_mm = 0.0
        self.frames_processed = 0
        self.last_result = None
        
        # Stocam istoricul valorilor instantanee
        self.true_volumes = []
        
        # Volum acumulat estimat pe mai multe orizonturi (legacy, pentru afisare rapida)
        self.predicted_volume_accumulation = {horizon: 0.0 for horizon in HORIZON_NAMES}
        
        # Stocăm valorile cumulate prezise și instantanee
        self.pred_volumes_acc = {horizon: [] for horizon in HORIZON_NAMES}
        self.pred_volumes = {horizon: [] for horizon in HORIZON_NAMES}
        
        # Event Reliability (Catchment level: Thresholds 0.1 L/m2 si 1.0 L/m2 CUMULAT)
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
        
        # Salvăm predicțiile cumulate (ce cantitate de apă se așteaptă să cadă PÂNĂ LA acel orizont)
        for horizon in HORIZON_NAMES:
            if hasattr(result, "predicted_volumes_horizons") and result.predicted_volumes_horizons:
                val = result.predicted_volumes_horizons.get(horizon, 0.0)
            else:
                val = 0.0
            self.pred_volumes_acc[horizon].append(val)
                
        # Salvăm predicțiile instantanee
        if hasattr(result, "instant_predicted_volumes") and result.instant_predicted_volumes:
            for horizon in HORIZON_NAMES:
                val = result.instant_predicted_volumes.get(horizon, 0.0)
                self.predicted_volume_accumulation[horizon] += val
                self.pred_volumes[horizon].append(val)
        else:
            for horizon in HORIZON_NAMES:
                self.pred_volumes[horizon].append(0.0)
                
        # Calculăm Catchment Event Reliability on the fly folosind Ferestre Cumulate
        # IMPORTANT: Valorile trebuie să fie IDENTICE cu target_step din frame_processor.py horizons
        for horizon, steps in HORIZON_STEPS.items():
            if len(self.true_volumes) > steps:
                # Realitatea cumulată (ex: suma ploilor din ultimul 1h)
                # Extragem ultimele 'steps' elemente și facem suma
                actual_acc_val = sum(self.true_volumes[-steps:])
                
                # Predicția făcută acum 'steps' cadre în urmă referitoare la cantitatea CUMULATĂ pe parcursul celor 'steps' cadre
                pred_acc_val = self.pred_volumes_acc[horizon][-1 - steps]
                
                for t in self.thresholds:
                    pred_event = pred_acc_val >= t
                    actual_event = actual_acc_val >= t
                    
                    counts = self.reliability_counts[t][horizon]
                    if pred_event and actual_event:
                        counts["hits"] += 1
                        # Eroarea cantitativă procentuală simetrică pe interval (sMAPE)
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
        """Returneaza MAP real/prezis aliniat la lead time-ul orizontului."""
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
        """Returneaza POD, FAR si CMAE la nivel de bazin pentru fiecare prag si orizont."""
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
