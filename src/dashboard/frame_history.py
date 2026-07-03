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
        self.predicted_volume_accumulation = {"15m": 0.0, "1h": 0.0, "2h": 0.0}
        
        # Stocăm valorile cumulate prezise și instantanee
        self.pred_volumes_acc = {"15m": [], "1h": [], "2h": []}
        self.pred_volumes = {"15m": [], "1h": [], "2h": []}
        
        # Event Reliability (Catchment level: Thresholds 0.1 L/m2 si 1.0 L/m2 CUMULAT)
        self.thresholds = [1.0, 5.0]
        self.reliability_counts = {}
        for t in self.thresholds:
            self.reliability_counts[t] = {
                "15m": {"hits": 0, "fa": 0, "miss": 0, "cr": 0, "abs_err_sum": 0.0},
                "1h": {"hits": 0, "fa": 0, "miss": 0, "cr": 0, "abs_err_sum": 0.0},
                "2h": {"hits": 0, "fa": 0, "miss": 0, "cr": 0, "abs_err_sum": 0.0}
            }

    def accumulate(self, result) -> None:
        self.total_map_mm += result.roi_map_mm
        self.frames_processed += 1
        self.last_result = result
        
        self.true_volumes.append(result.roi_map_mm)
        
        # Salvăm predicțiile cumulate (ce cantitate de apă se așteaptă să cadă PÂNĂ LA acel orizont)
        if hasattr(result, "predicted_volumes_horizons") and result.predicted_volumes_horizons:
            for horizon in ["15m", "1h", "2h"]:
                val = result.predicted_volumes_horizons.get(horizon, 0.0)
                self.pred_volumes_acc[horizon].append(val)
                
        # Salvăm predicțiile instantanee
        if hasattr(result, "instant_predicted_volumes") and result.instant_predicted_volumes:
            for horizon in ["15m", "1h", "2h"]:
                val = result.instant_predicted_volumes.get(horizon, 0.0)
                self.predicted_volume_accumulation[horizon] += val
                self.pred_volumes[horizon].append(val)
                
        # Calculăm Catchment Event Reliability on the fly folosind Ferestre Cumulate
        # IMPORTANT: Valorile trebuie să fie IDENTICE cu target_step din frame_processor.py horizons
        horizon_steps = {"15m": 2, "1h": 5, "2h": 9}
        
        for horizon, steps in horizon_steps.items():
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
