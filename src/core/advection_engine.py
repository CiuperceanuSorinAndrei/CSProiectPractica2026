"""Motor de advectie Semi-Lagrangiana.

Foloseste fluxul optic dens (Dense Optical Flow) pentru a extrapola cinematic 
campul de precipitatii, incluzand un filtru directionat (motion blur) bazat 
pe vectorul de miscare, pastrand dinamica fluidelor.
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage as ndimage
import scipy.sparse as sp

from config import RAIN_THRESHOLD_MIN


class AdvectionEngine:
    
    @staticmethod
    def extrapolate(
        rain_rate: np.ndarray,
        flow: np.ndarray | None,
        tracked_cells: list[dict],
        horizons: list[tuple[int, str]],
    ) -> tuple[dict[str, sp.csr_matrix], dict[str, np.ndarray]]:
        """Extrapoleaza precipitatiile si genereaza predictii.
        
        Returneaza:
            - sparse_preds: dictionar cu masti booleene rare (pentru economie RAM)
            - float_preds: dictionar cu matricile de precipitatii extrapolate (pentru calcul volum)
        """
        grid_h, grid_w = rain_rate.shape
        y_grid, x_grid = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
        
        # 1. Calculam rata globala de crestere/scadere din Kalman (Dinamica)
        valid_cells = [c for c in tracked_cells if c.get("is_tracked", False)]
        if valid_cells:
            growth_rates = [
                c.get("d_area_kalman", 0) / max(1.0, c.get("predicted_area_kalman", 1))
                for c in valid_cells
            ]
            mean_growth_rate = float(np.mean(growth_rates))
            # Vector mediu de miscare pentru Motion Blur directional
            mean_v_x = float(np.mean([c.get("v_x", 0.0) for c in valid_cells]))
            mean_v_y = float(np.mean([c.get("v_y", 0.0) for c in valid_cells]))
        else:
            mean_growth_rate = 0.0
            mean_v_x, mean_v_y = 0.0, 0.0
            
        step_decay_factor = float(np.clip(1.0 + mean_growth_rate, 0.95, 1.02))
        
        # Daca nu avem flow (ex: primul cadru), returnam ploaia statica
        if flow is None:
            flow_x = np.zeros((grid_h, grid_w), dtype=np.float32)
            flow_y = np.zeros((grid_h, grid_w), dtype=np.float32)
        else:
            flow_x = flow[:, :, 0]
            flow_y = flow[:, :, 1]
            
        # Filtram flow-ul pentru a evita artefacte de zgomot la advectie
        flow_x = ndimage.gaussian_filter(flow_x, sigma=1.0)
        flow_y = ndimage.gaussian_filter(flow_y, sigma=1.0)
            
        sparse_preds = {}
        float_preds = {}
        
        for steps, name in horizons:
            # Maparea inversa pentru Semi-Lagrangian advection
            map_x = x_grid - flow_x * steps
            map_y = y_grid - flow_y * steps
            
            # Advectia propriu-zisa
            shifted = cv2.remap(
                rain_rate, map_x, map_y, 
                interpolation=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
            
            # Decadere termodinamica bazata pe predictia Kalman de arie
            shifted *= (step_decay_factor ** steps)
            
            # Estompare Directionala (Motion Blur)
            if steps > 1 and (abs(mean_v_x) > 0.1 or abs(mean_v_y) > 0.1):
                # Generam un kernel directional bazat pe vectorul mediu
                # Calculam unghiul
                angle = np.degrees(np.arctan2(mean_v_y, mean_v_x))
                length = min(int(steps * 1.5), 15)  # Lungimea kernelului creste cu orizontul
                if length % 2 == 0:
                    length += 1
                
                # Cream un kernel linie si il rotim
                kernel = np.zeros((length, length), dtype=np.float32)
                kernel[length // 2, :] = 1.0
                
                # Rotim kernelul folosind cv2
                M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
                kernel = cv2.warpAffine(kernel, M, (length, length))
                kernel /= np.sum(kernel)
                
                # Aplicam convolutia pentru a dispersa ploaia pe directia de miscare
                shifted = cv2.filter2D(shifted, -1, kernel)
            elif steps > 1:
                # Fallback la blur simplu daca nu este miscare clara
                shifted = ndimage.gaussian_filter(shifted, sigma=min(steps * 0.1, 1.5))
                
            float_preds[name] = shifted
            # Generam masca booleana (si o aplicam pe ROI)
            pred_mask = (shifted >= RAIN_THRESHOLD_MIN).astype(np.float32)
            sparse_preds[name] = sp.csr_matrix(pred_mask)
            
        return sparse_preds, float_preds
