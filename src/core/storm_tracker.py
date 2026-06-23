"""Modul de cinematica: urmarire centroizi (Kalman) + predictie morfologica (OpenCV Flow).

StormTracker pastreaza starea de tracking (bancul de filtre Kalman, celulele si
matricea de ploaie din cadrul anterior) intre apeluri, deci o singura instanta
proceseaza intregul flux de cadre.
"""
from __future__ import annotations

import uuid
from typing import Any

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter


class StormTracker:
    _max_dist_pixels: int = None
    _kalman_bank: dict = None
    _previous_cells: list = None
    _previous_rain_matrix: Any = None

    def __init__(self, max_dist_pixels: int = 18):
        self._max_dist_pixels = max_dist_pixels
        self.reset()

    def reset(self) -> None:
        """Goleste complet starea de tracking."""
        self._kalman_bank = {}
        self._previous_cells = []
        self._previous_rain_matrix = None

    def track(self, current_cells: list[dict[str, Any]], rain_matrix: np.ndarray) -> list[dict[str, Any]]:
        """Urmareste centroizii folosind Kalman Filter si estimeaza evolutia morfologica
        a formei folosind algoritmul OpenCV Dense Optical Flow (Farneback).

        Args:
            current_cells: Lista celulelor detectate in cadrul curent
            rain_matrix: Matricea de precipitatii curenta

        Returns:
            Lista celulelor urmarite cu predictii de centroid si masca
        """
        # Daca s-a schimbat rezolutia grilei (zoom in/out pe harta), coordonatele
        # Kalman nu mai sunt valide pe noul grid, deci resetam starea.
        if self._previous_rain_matrix is not None and self._previous_rain_matrix.shape != rain_matrix.shape:
            self.reset()

        tracked_cells: list[dict[str, Any]] = []
        active_ids: set[str] = set()

        # 1. Calculam campul de advectie (Optical Flow) daca avem cadrul anterior
        flow = self._compute_optical_flow(self._previous_rain_matrix, rain_matrix)

        # 2. Predict pentru toate filtrele Kalman existente
        for kf in self._kalman_bank.values():
            kf.predict()

        # 3. Procesam fiecare celula curenta
        for c_cell in current_cells:
            tracked_cell = c_cell.copy()
            tracked_cell["is_tracked"] = False
            tracked_cell["centroid_history"] = list(c_cell.get("centroid_history", []))
            tracked_cell["area_history"] = list(c_cell.get("area_history", []))
            tracked_cell["cell_history"] = list(c_cell.get("cell_history", []))
            tracked_cell["predicted_centroid_x"] = float(c_cell.get("centroid_x", 0.0))
            tracked_cell["predicted_centroid_y"] = float(c_cell.get("centroid_y", 0.0))

            c_area = c_cell.get("area_pixels", c_cell.get("area", 1.0))
            c_max = c_cell.get("max_intensity", 1.0)
            c_volume = c_area * (c_max * 0.4)
            tracked_cell["volume"] = c_volume
            tracked_cell["area_pixels"] = int(c_area)

            # --- Matching cu celulele anterioare ---
            best_match = None
            min_hybrid_cost = float("inf")

            for p_cell in self._previous_cells:
                p_id = p_cell.get("cell_id")
                if p_id not in self._kalman_bank:
                    continue

                pred_x = self._kalman_bank[p_id].x[0, 0]
                pred_y = self._kalman_bank[p_id].x[1, 0]
                radius_limit = max(
                    self._max_dist_pixels,
                    self._adaptive_match_radius(p_cell.get("area_pixels", p_cell.get("area", 1.0))),
                )

                dist = self._calculate_distance(c_cell["centroid_y"], c_cell["centroid_x"], pred_y, pred_x)
                if dist > radius_limit:
                    continue

                p_area = p_cell.get("area_pixels", p_cell.get("area", 1.0))
                area_ratio = max(c_area, p_area) / (min(c_area, p_area) + 1e-5)

                p_volume = p_cell.get("volume", p_area * (p_cell.get("max_intensity", 1.0) * 0.4))
                volume_ratio = max(c_volume, p_volume) / (min(c_volume, p_volume) + 1e-5)

                prev_mask = self.build_cell_mask(p_cell, rain_matrix)
                curr_mask = self.build_cell_mask(c_cell, rain_matrix)
                iou_penalty = 1.0 - self._mask_iou(curr_mask, prev_mask)

                hybrid_cost = dist + (area_ratio * 1.2) + (volume_ratio * 1.8) + (iou_penalty * 3.0)

                if hybrid_cost < min_hybrid_cost:
                    min_hybrid_cost = hybrid_cost
                    best_match = p_cell

            # --- Actualizare Kalman + istoric ---
            if best_match:
                cell_id = best_match["cell_id"]
                tracked_cell["cell_id"] = cell_id
                tracked_cell["is_tracked"] = True
                tracked_cell["centroid_history"] = list(best_match.get("centroid_history", []))
                tracked_cell["area_history"] = list(best_match.get("area_history", []))
                tracked_cell["cell_history"] = list(best_match.get("cell_history", []))

                p_volume = best_match.get("volume", 1.0)
                tracked_cell["volume_trend"] = c_volume / (p_volume + 1e-5)
                prev_area = float(best_match.get("area_pixels", c_area))
                area_trend = c_area / (prev_area + 1e-5)
                tracked_cell["area_trend"] = area_trend

                kf = self._kalman_bank[cell_id]
                pred_x_prior, pred_y_prior = kf.x[0, 0], kf.x[1, 0]
                tracked_cell["prediction_error_pixels"] = self._calculate_distance(
                    c_cell["centroid_y"], c_cell["centroid_x"], pred_y_prior, pred_x_prior,
                )

                kf.update(np.array([[c_cell["centroid_x"]], [c_cell["centroid_y"]]]))
                tracked_cell["centroid_history"].append(
                    (float(c_cell["centroid_y"]), float(c_cell["centroid_x"]))
                )
                tracked_cell["centroid_history"] = tracked_cell["centroid_history"][-6:]
                tracked_cell["area_history"].append(int(c_area))
                tracked_cell["area_history"] = tracked_cell["area_history"][-6:]
                tracked_cell["cell_history"].append({
                    "centroid_y": float(c_cell["centroid_y"]),
                    "centroid_x": float(c_cell["centroid_x"]),
                    "area_pixels": int(c_area),
                })
                tracked_cell["cell_history"] = tracked_cell["cell_history"][-6:]
                active_ids.add(cell_id)
            else:
                cell_id = str(uuid.uuid4())[:8]
                tracked_cell["cell_id"] = cell_id
                tracked_cell["prediction_error_pixels"] = 0.0
                tracked_cell["volume_trend"] = 1.0
                tracked_cell["centroid_history"] = [
                    (float(c_cell["centroid_y"]), float(c_cell["centroid_x"]))
                ]
                tracked_cell["area_history"] = [int(c_area)]
                tracked_cell["cell_history"] = [{
                    "centroid_y": float(c_cell["centroid_y"]),
                    "centroid_x": float(c_cell["centroid_x"]),
                    "area_pixels": int(c_area),
                }]
                tracked_cell["area_trend"] = 1.0

                self._kalman_bank[cell_id] = self._instantiate_kalman_filter(
                    c_cell["centroid_y"], c_cell["centroid_x"],
                )
                active_ids.add(cell_id)

            # --- Viteza din Kalman + observatii ---
            current_kf = self._kalman_bank[cell_id]
            tracked_cell["v_x"] = current_kf.x[2, 0]
            tracked_cell["v_y"] = current_kf.x[3, 0]

            if len(tracked_cell["centroid_history"]) >= 2:
                history = tracked_cell["centroid_history"]
                deltas_x = []
                deltas_y = []
                for idx in range(1, len(history)):
                    prev_y, prev_x = history[idx - 1]
                    curr_y, curr_x = history[idx]
                    deltas_x.append(curr_x - prev_x)
                    deltas_y.append(curr_y - prev_y)

                obs_v_x = float(np.mean(deltas_x[-3:]))
                obs_v_y = float(np.mean(deltas_y[-3:]))
                tracked_cell["v_x"] = 0.4 * tracked_cell["v_x"] + 0.6 * obs_v_x
                tracked_cell["v_y"] = 0.4 * tracked_cell["v_y"] + 0.6 * obs_v_y
                current_kf.x[2, 0] = tracked_cell["v_x"]
                current_kf.x[3, 0] = tracked_cell["v_y"]

            predicted_centroid_x = tracked_cell["centroid_x"] + tracked_cell["v_x"]
            predicted_centroid_y = tracked_cell["centroid_y"] + tracked_cell["v_y"]
            tracked_cell["predicted_centroid_x"] = float(predicted_centroid_x)
            tracked_cell["predicted_centroid_y"] = float(predicted_centroid_y)

            # --- Predictie masca morfologica ---
            full_current_mask = self.build_cell_mask(c_cell, rain_matrix)

            if len(tracked_cell["area_history"]) >= 2:
                area_deltas = []
                for idx in range(1, len(tracked_cell["area_history"])):
                    prev_a = max(tracked_cell["area_history"][idx - 1], 1)
                    curr_a = max(tracked_cell["area_history"][idx], 1)
                    area_deltas.append(curr_a / prev_a)
                raw_area_trend = float(np.mean(area_deltas[-3:]))
            else:
                raw_area_trend = float(tracked_cell.get("area_trend", 1.0))

            if len(tracked_cell["cell_history"]) >= 3:
                recent_areas = [item["area_pixels"] for item in tracked_cell["cell_history"][-3:]]
                recent_area_trend = float(np.mean(recent_areas) / max(recent_areas[0], 1))
            else:
                recent_area_trend = raw_area_trend

            predicted_area_trend = float(np.clip(
                0.8 * recent_area_trend + 0.2 * tracked_cell.get("volume_trend", 1.0),
                0.90, 1.14,
            ))
            scale_factor = float(np.clip(np.sqrt(predicted_area_trend), 0.97, 1.05))
            predicted_area_pixels = int(round(max(c_area, 1.0) * predicted_area_trend))

            # Predictie forma prin deplasare masca (Optical Flow + Kalman blend)
            if flow is not None:
                cell_pixels = full_current_mask == 1
                mean_flow_y = float(np.mean(flow[:, :, 1][cell_pixels])) if np.any(cell_pixels) else 0.0
                mean_flow_x = float(np.mean(flow[:, :, 0][cell_pixels])) if np.any(cell_pixels) else 0.0
                shift_x = 0.75 * tracked_cell["v_x"] + 0.25 * mean_flow_x
                shift_y = 0.75 * tracked_cell["v_y"] + 0.25 * mean_flow_y
            else:
                shift_x = tracked_cell["v_x"]
                shift_y = tracked_cell["v_y"]

            predicted_mask = self._predict_mask_shape(
                full_current_mask,
                tracked_cell["centroid_y"],
                tracked_cell["centroid_x"],
                shift_y, shift_x,
                scale_factor,
            )

            tracked_cell["predicted_mask"] = predicted_mask
            tracked_cell["predicted_area_pixels"] = predicted_area_pixels
            tracked_cell["size_error_pixels"] = abs(predicted_area_pixels - int(c_area))
            tracked_cell["size_error_percent"] = (
                100.0 * tracked_cell["size_error_pixels"] / max(int(c_area), 1)
            )
            tracked_cell["area_trend"] = predicted_area_trend
            tracked_cells.append(tracked_cell)

        # Cleanup: stergem filtrele Kalman pentru celulele care nu mai sunt active
        for old_id in list(self._kalman_bank.keys()):
            if old_id not in active_ids:
                del self._kalman_bank[old_id]

        # Salvam starea pentru cadrul urmator
        self._previous_cells = tracked_cells
        self._previous_rain_matrix = rain_matrix.copy()

        return tracked_cells

    # -----------------------------------------------------------------------
    # Utilitare publice (folosite si de orchestrator pentru predictia volumului)
    # -----------------------------------------------------------------------
    @staticmethod
    def build_cell_mask(cell: dict, rain_matrix: np.ndarray) -> np.ndarray:
        """Construieste masca binara a unei celule pe dimensiunea matricei de ploaie."""
        mask = np.zeros_like(rain_matrix, dtype=np.uint8)
        coords = cell.get("coords")

        if coords is not None and len(coords) > 0:
            coords = np.asarray(coords)
            mask[coords[:, 0], coords[:, 1]] = 1
            return mask

        # Fallback: reconstruim masca din centroid + arie
        cy = int(cell.get("centroid_y", 0))
        cx = int(cell.get("centroid_x", 0))
        area = float(cell.get("area_pixels", cell.get("area", 1.0)))
        radius = max(1, int(np.sqrt(max(area, 1.0) / np.pi)))
        y0 = max(0, cy - radius)
        y1 = min(rain_matrix.shape[0], cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(rain_matrix.shape[1], cx + radius + 1)

        local = rain_matrix[y0:y1, x0:x1] >= 0.5
        mask[y0:y1, x0:x1][local] = 1
        return mask

    @staticmethod
    def translate_mask(mask: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
        """Translateaza o masca binara cu (shift_y, shift_x) pixeli."""
        shifted = np.zeros_like(mask, dtype=np.uint8)
        src_y, src_x = np.where(mask > 0)
        dst_y = np.rint(src_y + shift_y).astype(int)
        dst_x = np.rint(src_x + shift_x).astype(int)

        valid = (
            (dst_y >= 0) & (dst_y < mask.shape[0])
            & (dst_x >= 0) & (dst_x < mask.shape[1])
        )
        shifted[dst_y[valid], dst_x[valid]] = 1
        return shifted

    # -----------------------------------------------------------------------
    # Utilitare interne
    # -----------------------------------------------------------------------
    @staticmethod
    def _instantiate_kalman_filter(initial_y: float, initial_x: float) -> KalmanFilter:
        """Creeaza un Kalman Filter 4D (x, y, vx, vy) pentru urmarirea centroidului."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([[initial_x], [initial_y], [0.0], [0.0]])
        kf.F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        kf.P *= 1.2
        kf.Q = np.array([
            [0.005, 0.0, 0.0, 0.0],
            [0.0, 0.005, 0.0, 0.0],
            [0.0, 0.0, 0.02, 0.0],
            [0.0, 0.0, 0.0, 0.02],
        ])
        kf.R = np.array([
            [10.0, 0.0],
            [0.0, 10.0],
        ])
        return kf

    @staticmethod
    def _compute_optical_flow(previous_rain: np.ndarray | None, current_rain: np.ndarray) -> np.ndarray | None:
        """Calculeaza campul de advectie dense (Farneback) intre doua cadre consecutive."""
        if previous_rain is None or previous_rain.shape != current_rain.shape:
            return None

        # OpenCV are nevoie de imagini pe 8 biti normalizate
        prev_img = np.clip(previous_rain * 10, 0, 255).astype(np.uint8)
        curr_img = np.clip(current_rain * 10, 0, 255).astype(np.uint8)

        return cv2.calcOpticalFlowFarneback(
            prev_img, curr_img, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

    @staticmethod
    def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Intersection over Union intre doua masti binare."""
        if mask_a.shape != mask_b.shape:
            return 0.0

        intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
        union = np.logical_or(mask_a > 0, mask_b > 0).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def _predict_mask_shape(
        mask: np.ndarray | None,
        center_y: float, center_x: float,
        shift_y: float, shift_x: float,
        scale_factor: float,
    ) -> np.ndarray | None:
        """Prezice forma viitoare a mastii prin scalare + translatie afina."""
        if mask is None or np.sum(mask) == 0:
            return mask

        h, w = mask.shape
        matrix = np.array([
            [scale_factor, 0.0, (1.0 - scale_factor) * center_x + shift_x],
            [0.0, scale_factor, (1.0 - scale_factor) * center_y + shift_y],
        ], dtype=np.float32)

        warped = cv2.warpAffine(
            mask.astype(np.uint8), matrix, (w, h),
            flags=cv2.INTER_NEAREST, borderValue=0,
        )
        return (warped > 0).astype(np.uint8)

    @staticmethod
    def _calculate_distance(y1: float, x1: float, y2: float, x2: float) -> float:
        """Distanta euclidiana intre doua puncte (y, x)."""
        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    @staticmethod
    def _adaptive_match_radius(area_pixels: float) -> float:
        """Raza adaptiva de matching bazata pe aria celulei."""
        return float(np.clip(10.0 + np.sqrt(max(area_pixels, 1.0)) * 0.9, 14.0, 32.0))
