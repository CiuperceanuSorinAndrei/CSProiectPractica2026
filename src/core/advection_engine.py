"""Motor de advectie Hibrid (Linear Jump + Spatial Growth + Directional Blur).

Aplica advectia semi-Lagrangiana liniara dintr-un singur salt (V6) pentru a
pastra integritatea formei furtunii la 2 ore, si aplica Cresterea Spatiala 
Localizata (V7) strict pe centroidul anticipat pentru corectia volumetrica.
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.sparse as sp

from config import RAIN_THRESHOLD_MIN, RAIN_THRESHOLD_TRACKING


class AdvectionEngine:
    
    @staticmethod
    def _create_spatial_growth_mask(
        shape: tuple[int, int],
        tracked_cells: list[dict],
        steps: int
    ) -> np.ndarray:
        """Creeaza o masca globala de crestere/scadere bazata pe halouri Gaussiene."""
        h, w = shape
        growth_mask = np.ones((h, w), dtype=np.float32)
        decay_mask = np.ones((h, w), dtype=np.float32)
        
        valid_cells = [c for c in tracked_cells if c.get("is_tracked", False)]
        if not valid_cells:
            return growth_mask
            
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        
        for c in valid_cells:
            area = max(1.0, c.get("predicted_area_kalman", 1.0))
            d_area = c.get("d_area_kalman", 0.0)
            d_area = c.get("d_area_kalman", 0.0)
            
            # V18: Model Asimptotic de Crestere (Fara parabole explozive)
            if d_area > 0:
                tau_growth = 3.0 # Furtuna creste timp de ~45 min
                pred_area = area + d_area * tau_growth * (1.0 - np.exp(-steps / tau_growth))
            else:
                tau_decay = 4.0 # Furtuna moare lent
                pred_area = area + d_area * tau_decay * (1.0 - np.exp(-steps / tau_decay))
                
            pred_area = max(1.0, pred_area) 
            
            # V23 Climatological Decay: Fortam scaderea volumului la orizonturi mari (Regression to the mean).
            # Fara NWP, furtunile convective prezise la >1h tind sa supraestimeze realitatea (deoarece ele se disipa).
            # steps=2 (30m) -> max_growth = 1.2
            # steps=4 (1h)  -> max_growth = 0.9
            # steps=8 (2h)  -> max_growth = 0.3
            max_growth = max(0.3, 1.2 - 0.15 * max(0, steps - 2))
            cumulative_factor = min(pred_area / area, max_growth)
            
            # 3. Acceleratia Spatiala (Traiectorii Curbate)
            a_x = c.get("a_x", 0.0)
            a_y = c.get("a_y", 0.0)
            cy = c.get("centroid_y", h/2) + c.get("v_y", 0.0) * steps + 0.5 * a_y * (steps ** 2)
            cx = c.get("centroid_x", w/2) + c.get("v_x", 0.0) * steps + 0.5 * a_x * (steps ** 2)
            
            # V19: Raza haloului asimetrica (corecteaza explozia de volum!)
            # Daca furtuna creste, folosim aria extinsa. Daca moare, TRBUIE sa acoperim intreaga furtuna existenta!
            if cumulative_factor >= 1.0:
                radius = max(5.0, np.sqrt(pred_area / np.pi))
            else:
                radius = max(5.0, np.sqrt(area / np.pi))
            
            dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
            halo = np.exp(-dist_sq / (2.0 * radius**2))
            
            local_multiplier = 1.0 + (cumulative_factor - 1.0) * halo
            
            # Evitam inmultirea multiplicatorilor (explozie) si luam doar impactul maxim
            if cumulative_factor >= 1.0:
                growth_mask = np.maximum(growth_mask, local_multiplier)
            else:
                decay_mask = np.minimum(decay_mask, local_multiplier)
            
        final_mask = growth_mask * decay_mask
        return np.clip(final_mask, 0.5, 2.0)

    @staticmethod
    def extrapolate(
        rain_rate: np.ndarray,
        flow: np.ndarray | None,
        tracked_cells: list[dict],
        horizons: list[tuple[int, str]],
    ) -> tuple[dict[str, sp.csr_matrix], dict[int, np.ndarray]]:
        """Extrapoleaza precipitatiile folosind Advectie Liniara dintr-un singur salt.
        
        Returneaza:
            sparse_preds: predictii de maska (ploaie da/nu) strict pentru orizonturile cerute.
            float_preds: matricile complete float de rain_rate pentru TOATE step-urile
                         de la 1 la max_step, pentru integrarea volumetrica corecta.
        """
        grid_h, grid_w = rain_rate.shape
        y_grid, x_grid = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
        
        if flow is None:
            flow_x = np.zeros((grid_h, grid_w), dtype=np.float32)
            flow_y = np.zeros((grid_h, grid_w), dtype=np.float32)
        else:
            flow_x = flow[:, :, 0]
            flow_y = flow[:, :, 1]
            
        sparse_preds = {}
        float_preds = {}
        
        max_step = max(h[0] for h in horizons) if horizons else 0
        horizon_map = {h[0]: h[1] for h in horizons}
        
        valid_cells = [c for c in tracked_cells if c.get("is_tracked", False)]
        if valid_cells:
            mean_tracking_error = float(np.mean([c.get("prediction_error_pixels", 0.5) for c in valid_cells]))
        else:
            mean_tracking_error = 0.5
            
        base_uncertainty = max(0.2, mean_tracking_error)
        sum_orig = float(np.sum(rain_rate))
        
        # Coordonatele curente de plecare (backward trajectory)
        map_x = x_grid.copy()
        map_y = y_grid.copy()
        
        # 1. Advectie Semi-Lagrangiana cu Traiectorii (Zero-Diffusion)
        for step in range(1, max_step + 1):
            # Evaluam viteza vantului la pozitia precedenta de plecare
            flow_at_p_x = cv2.remap(flow_x, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            flow_at_p_y = cv2.remap(flow_y, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Facem inca un pas inapoi in timp pentru a gasi originea
            map_x = map_x - flow_at_p_x
            map_y = map_y - flow_at_p_y
            
            # Mutam ploaia O SINGURA DATA de la momentul t=0! Fara blur adaugat recursiv.
            shifted = cv2.remap(
                rain_rate, map_x, map_y, 
                interpolation=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
            
            # 2. Conservarea Masei pe Ambele Sensuri
            # V21: ELIMINAT. "shifted *= (sum_orig / sum_shifted)" forta ca totalul de ploaie sa ramana 
            # egal cu cel de la T0. Daca ploaia iesea in mod real de pe radar, 
            # acest "fixer" inmultea agresiv norii ramasi pentru a compensa apa pierduta (Mass Drift de frontiera).
            # Lasam volumul sa scada natural cand furtunile se destrama sau ies din acoperire!
            
            growth_mask = AdvectionEngine._create_spatial_growth_mask(
                (grid_h, grid_w), tracked_cells, step
            )
            
            # Masa reala (Volum pur, conservat, fara dilatare artificiala)
            shifted_grown = shifted * growth_mask
            float_preds[step] = shifted_grown
            
            # Masca de incertitudine cinematica (Dilatata pentru POD)
            if step in horizon_map:
                name = horizon_map[step]
                
                # S-PROG Gaussian Diffusion: Pe masura ce creste orizontul,
                # predictibilitatea detaliilor scade. Aplicam un blur pe campul de precipitatii
                # care conserva complet volumul, dar "topeste" varfurile intense.
                if step > 2:
                    # V19: Hyper-Diffusion Fix. Limitare fizica a erorii [0.2, 0.6]
                    sigma_factor = np.clip(base_uncertainty * 0.1, 0.2, 0.6)
                    sigma = float(sigma_factor * step)
                    ksize = int(sigma * 3) | 1  # trebuie sa fie impar
                    # Aplicam blur doar daca dimensiunea e >= 3
                    if ksize >= 3:
                        shifted_grown_blurred = cv2.GaussianBlur(shifted_grown, (ksize, ksize), sigma)
                    else:
                        shifted_grown_blurred = shifted_grown
                else:
                    shifted_grown_blurred = shifted_grown
                
                # Masca initiala (folosind pragul de Core Tracking)
                # Datorita blur-ului S-PROG, varfurile s-au latit dar intensitatea a scazut.
                # Threshold-ul de 0.5 va taia marginile scazute, scazand natural FAR-ul!
                base_mask = (shifted_grown_blurred >= RAIN_THRESHOLD_TRACKING).astype(np.float32)
                
                sparse_preds[name] = sp.csr_matrix(base_mask)
            
        return sparse_preds, float_preds
