"""Motor de advectie Rigid (Lagrangian Persistence)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


from src.core.domain import StormCell
from src.core.nowcast.kinematics import KinematicsEngine
from src.core.nowcast.kinematic_advector import KinematicAdvector

from config import RAIN_THRESHOLD_TRACKING, RAIN_THRESHOLD_MIN

class AdvectionEngine:
    def __init__(
        self,
        kinematic_advector: KinematicAdvector,
    ) -> None:
        self.kinematic_advector = kinematic_advector
        self._ema_trend: float = 1.0  # Exponential Moving Average al trendului global
        
        # State PID Feedback Control
        self.dynamic_bias_correction: float = 1.0
        self._pid_bias: float = 1.0
        self._error_history: list[dict] = []
        
        # Horizon Calibrators
        self._bias_15m = 1.0
        self._bias_1h = 1.0
        self._bias_2h = 1.0

    def update_feedback(self, actual_map: float, preds: dict) -> None:
        """Encapsulated PID feedback update to keep AdvectionEngine state internal."""
        self._error_history.append({
            "actual": actual_map,
            "preds": preds
        })
        
        # Keep last 10 frames for T-9 calculation
        if len(self._error_history) > 10:
            self._error_history.pop(0)
            
        # Calculate 15m bias (index -2)
        if len(self._error_history) >= 2:
            past = self._error_history[-2]
            pred = past["preds"].get("15m", 0.0)
            if pred > 0.1 and actual_map > 0.1:
                # Log-space error for PID
                error = abs(np.log(actual_map + 0.01) - np.log(pred + 0.01))
                alpha = float(np.tanh(error))
                raw_ratio = (actual_map + 0.1) / (pred + 0.1)
                self._pid_bias = (1.0 - alpha) * self._pid_bias + alpha * raw_ratio
                self._bias_15m = 0.9 * self._bias_15m + 0.1 * np.clip(raw_ratio, 0.2, 3.0)
            else:
                self._pid_bias = 0.9 * self._pid_bias + 0.1 * 1.0
                self._bias_15m = 0.9 * self._bias_15m + 0.1 * 1.0
        # Calculate 1h bias (index -5)
        if len(self._error_history) >= 5:
            past = self._error_history[-5]
            pred = past["preds"].get("1h", 0.0)
            if pred > 0.1 and actual_map > 0.1:
                ratio = (actual_map + 0.1) / (pred + 0.1)
                self._bias_1h = 0.8 * self._bias_1h + 0.2 * np.clip(ratio, 0.2, 3.0)
            else:
                self._bias_1h = 0.9 * self._bias_1h + 0.1 * 1.0

        # Calculate 2h bias (index -9)
        if len(self._error_history) >= 9:
            past = self._error_history[-9]
            pred = past["preds"].get("2h", 0.0)
            if pred > 0.1 and actual_map > 0.1:
                ratio = (actual_map + 0.1) / (pred + 0.1)
                self._bias_2h = 0.8 * self._bias_2h + 0.2 * np.clip(ratio, 0.2, 3.0)
            else:
                self._bias_2h = 0.9 * self._bias_2h + 0.1 * 1.0
        self.dynamic_bias_correction = self._pid_bias

    def extrapolate(
        self,
        rain_rate: np.ndarray,
        tracked_cells: list[StormCell],
        horizons: list[tuple[int, str]],
        roi_mask: np.ndarray = None,
    ) -> tuple[dict[str, sp.csr_matrix], dict[int, np.ndarray], dict[int, list[StormCell]]]:
        """Extrapoleaza precipitatiile (Hydrological Catchment Nowcasting)."""
        rain_rate = np.nan_to_num(rain_rate, nan=0.0).astype(np.float32)
        
        sparse_preds = {}
        float_preds = {}
        predicted_cells_dict = {}
        
        max_step = max(h[0] for h in horizons) if horizons else 0
        horizon_map = {h[0]: h[1] for h in horizons}
        
        valid_cells = [c for c in tracked_cells if c.is_tracked]
        simulated_cells = [c.clone() for c in valid_cells]
        
        # 1. Kinematic Threat Filter (TTA)
        global_vx, global_vy = 0.0, 0.0
        mean_E, mean_dE = 1.0, 0.0
        if simulated_cells:
            # Target-Locked Advection: Centrul de greutate se calculeaza strict pe bazin (roi_mask)!
            if roi_mask is not None:
                y_indices, x_indices = np.where(roi_mask > 0)
                if len(y_indices) > 0:
                    cy = float(np.mean(y_indices))
                    cx = float(np.mean(x_indices))
                else:
                    cy, cx = rain_rate.shape[0] / 2.0, rain_rate.shape[1] / 2.0
            else:
                cy, cx = rain_rate.shape[0] / 2.0, rain_rate.shape[1] / 2.0
                
            threat_cells = []
            
            for c in simulated_cells:
                dx = cx - getattr(c, 'centroid_x', cx)
                dy = cy - getattr(c, 'centroid_y', cy)
                dist = (dx**2 + dy**2)**0.5
                
                # Furtuna e deja foarte aproape (amenintare iminenta)
                if dist <= 30.0:
                    threat_cells.append(c)
                    continue
                    
                # Verifica daca vectorul vitezei bate spre centrul ROI
                dot = c.v_x * dx + c.v_y * dy
                if dot > 0:
                    v_proj = dot / dist
                    if v_proj > 0:
                        tta = dist / v_proj  # Time-to-Arrival (cadre)
                        # Daca ne loveste in fereastra de predictie (plus o marja de eroare)
                        if tta <= max_step + 4:
                            threat_cells.append(c)
            
            # Daca niciuna nu ne ameninta direct in orizontul nostru,
            # selectam cele mai apropiate 3 celule pentru a capta fluxul sinoptic local.
            if not threat_cells:
                sorted_cells = sorted(simulated_cells, key=lambda c: (cx - getattr(c, 'centroid_x', cx))**2 + (cy - getattr(c, 'centroid_y', cy))**2)
                threat_cells = sorted_cells[:3]
                
            global_vx = float(np.median([c.v_x for c in threat_cells]))
            global_vy = float(np.median([c.v_y for c in threat_cells]))
        
        # 2. Extragem starea termodinamica curenta folosind Volumele REALE (masa reala de apa)
        # Folosim DOAR threat_cells (Norii relevanti care vin spre noi). Restul tarii nu ne mai influenteaza!
        mean_E = np.mean([max(getattr(c, 'volume', 1.0), 1e-6) for c in threat_cells]) if 'threat_cells' in locals() and threat_cells else 1.0
        
        # dE real = Volumul curent - Volumul precedent (pe baza la volume_trend)
        mean_dE = np.mean([
            getattr(c, 'volume', 0.0) - (getattr(c, 'volume', 0.0) / max(getattr(c, 'volume_trend', 1.0), 1e-5)) 
            for c in threat_cells
        ]) if 'threat_cells' in locals() and threat_cells else 0.0
        
        current_E = float(mean_E)
        current_dE = float(mean_dE)
        
        from src.core.nowcast.reaction_diffusion import update_energy
        
        # PID Feedback Loop (short-term shock absorber)
        short_term_correction = 1.0 + (self.dynamic_bias_correction - 1.0) * 0.5
        thermo_multiplier = short_term_correction
        
        for step in range(1, max_step + 1):
            
            # Advectie uniforma prin shiftare scipy
            shift_y = step * global_vy
            shift_x = step * global_vx
            
            shifted_raw = self.kinematic_advector.advect(
                rain_rate, shift_y, shift_x
            )
            
            # Masa aflata efectiv pe ecran (include scaderea naturala la margini)
            mass_in_domain = float(np.sum(shifted_raw))
            
            # Lăsăm pixelii nealterați. Pragul va fi aplicat corect doar de Evaluator, 
            # DUPĂ ce PID-ul și Termodinamica au scalat complet furtuna. 
            # Aceasta previne Efectul de Clichet Asimetric (pierderea sistematică de masă la margini).
            shifted = shifted_raw
            
            # 1. Conservarea Masei a fost stearsa (Nu dorim compensarea ploilor usoare sterse).
            # Ploaia stearsa la prag ramane stearsa, evitand inflatia artificiala a nucleelor severe.
            
            # 2. Simulăm pasul termodinamic organic (care are limite fizice absolute in reaction_diffusion.py)
            current_E, current_dE, R_step = update_energy(current_E, np.array([]), current_dE)
            thermo_multiplier *= R_step
            
            # Calibrare globală specifică orizontului (aplicata simplu, o singura data)
            global_bias = 1.0
            if step <= 2: global_bias = self._bias_15m
            elif step <= 5: global_bias = self._bias_1h
            else: global_bias = self._bias_2h
            
            # Aplicăm creșterea termodinamică și calibrarea orizontului
            shifted = shifted * thermo_multiplier * global_bias
            
            float_preds[step] = shifted
            
            # Pentru compatibilitatea codului (nu le mai folosim în raport)
            predicted_cells_dict[step] = []
            
            if step in horizon_map:
                name = horizon_map[step]
                base_mask = (shifted >= RAIN_THRESHOLD_TRACKING).astype(np.float32)
                sparse_preds[name] = sp.csr_matrix(base_mask)
            
        return sparse_preds, float_preds, predicted_cells_dict
