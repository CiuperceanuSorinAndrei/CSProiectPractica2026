"""Estimator de flux optic pentru fluide (precipitatii).

Inlocuieste Farneback cu DIS (Dense Inverse Search) Optical Flow.
DIS este mult mai rapid si capteaza dislocari mari (jet-streams).
"""
from __future__ import annotations

import cv2
import numpy as np


class FlowEstimator:
    """Calculeaza campul global de miscare al furtunilor (Dense Optical Flow)."""

    def __init__(self):
        # Folosim preset-ul MEDIUM care ofera un balans perfect intre precizie si viteza
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def compute(self, previous_rain: np.ndarray | None, current_rain: np.ndarray) -> np.ndarray | None:
        """Calculeaza Dense Optical Flow intre cadru precedent si curent.
        
        Args:
            previous_rain: Matricea 2D a ploii la T-1.
            current_rain: Matricea 2D a ploii la T0.
            
        Returns:
            flow_full: O matrice 3D (H, W, 2) cu vectorii de miscare (dx, dy).
        """
        if previous_rain is None or previous_rain.shape != current_rain.shape:
            return None
            
        # Normalizare pentru a scoate in evidenta contrastul (formele norilor)
        # V19: Trecere la Logarithmic dBZ in loc de Linear Clip pentru a nu orbi algoritmul la furtunile > 25.5 mm/h
        def rain_to_uint8(rain: np.ndarray) -> np.ndarray:
            r_safe = np.clip(rain, 0.01, None)
            dbz = 23.0 + 16.0 * np.log10(r_safe)
            # Clipam fizic intre 0 si 60 dBZ si mapam spre 0-255 (uint8)
            dbz_norm = np.clip(dbz, 0.0, 60.0) * (255.0 / 60.0)
            return dbz_norm.astype(np.uint8)
            
        h, w = previous_rain.shape
        
        # Facem downscale IN DOMENIU LINIAR pentru a prinde miscarile largi
        # Fara sa nivelam varfurile fizice de precipitatii (rezolva problema orbirii)
        prev_small_lin = cv2.resize(previous_rain, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        curr_small_lin = cv2.resize(current_rain, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        
        prev_small = rain_to_uint8(prev_small_lin)
        curr_small = rain_to_uint8(curr_small_lin)
        
        # DIS Optical Flow (foarte stabil)
        flow_small = self._dis.calc(prev_small, curr_small, None)
        
        # Aplicam blur pe varianta small (de 4x mai rapid) si evitam ca aerul senin
        # (vector=0) sa incetineasca furtuna din cauza blur-ului.
        flow_blur_x = cv2.GaussianBlur(flow_small[:, :, 0], (15, 15), 0)
        flow_blur_y = cv2.GaussianBlur(flow_small[:, :, 1], (15, 15), 0)
        
        mask = (curr_small_lin > 0.1).astype(np.float32)
        flow_small[:, :, 0] = flow_blur_x * mask + flow_small[:, :, 0] * (1.0 - mask)
        flow_small[:, :, 1] = flow_blur_y * mask + flow_small[:, :, 1] * (1.0 - mask)
        
        # Upscale inapoi la rezolutia originala
        flow_full = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Inmultim cu 2 deoarece am facut downscale cu factor de 0.5 (x2 la pixel movement)
        return flow_full * 2.0
