"""Modul de cinematica: urmarire centroizi folosind filtru Kalman 8D si KD-Tree Matcher."""
from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from src.core.storm_filter import StormFilter
from src.core.matcher import Matcher
from src.core.flow_estimator import FlowEstimator


class StormTracker:
    USE_LOGISTIC_GROWTH = True
    USE_ADAPTIVE_KALMAN = True

    def __init__(self, max_dist_pixels: int = 15):
        self._max_dist_pixels = max_dist_pixels
        self._kalman_bank: dict[str, StormFilter] = {}
        self._previous_cells: list[dict] = []
        self._previous_rain_matrix: np.ndarray | None = None
        self._flow_estimator = FlowEstimator()

    @staticmethod
    def build_cell_mask(cell: dict, rain_matrix: np.ndarray) -> np.ndarray:
        if "_cached_mask" in cell:
            return cell["_cached_mask"]
        mask = np.zeros_like(rain_matrix, dtype=np.uint8)
        coords = cell.get("coords", np.array([]))
        if len(coords) > 0:
            mask[coords[:, 0], coords[:, 1]] = 1
        cell["_cached_mask"] = mask
        return mask

    def reset(self) -> None:
        self._kalman_bank = {}
        self._previous_cells = []
        self._previous_rain_matrix = None

    def track(self, current_cells: list[dict[str, Any]], rain_matrix: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray | None]:
        if self._previous_rain_matrix is not None and self._previous_rain_matrix.shape != rain_matrix.shape:
            self.reset()

        tracked_cells: list[dict[str, Any]] = []
        active_ids: set[str] = set()

        # 1. Optical flow global (DIS)
        flow = self._flow_estimator.compute(self._previous_rain_matrix, rain_matrix)

        # 2. Kalman predict (Constant Acceleration Model)
        for kf in self._kalman_bank.values():
            kf.predict()

        # Pregatim celulele curente
        for c_cell in current_cells:
            c_area = c_cell.get("area_pixels", c_cell.get("area", 1.0))
            c_max = c_cell.get("max_intensity", 1.0)
            c_cell["volume"] = c_area * (c_max * 0.4)
            c_cell["area_pixels"] = int(c_area)
            self.build_cell_mask(c_cell, rain_matrix)

        for p_cell in self._previous_cells:
            self.build_cell_mask(p_cell, rain_matrix)

        # 3. Construim matricea de asocieri folosind KD-Tree
        matches = Matcher.match_cells(
            current_cells, self._previous_cells, self._kalman_bank, self._max_dist_pixels
        )

        # 4. Procesam fiecare celula cu asocierea gasita
        for i, c_cell in enumerate(current_cells):
            tracked_cell = c_cell.copy()
            tracked_cell["is_tracked"] = False
            tracked_cell["centroid_history"] = list(c_cell.get("centroid_history", []))
            tracked_cell["area_history"] = list(c_cell.get("area_history", []))
            tracked_cell["cell_history"] = list(c_cell.get("cell_history", []))
            c_area = tracked_cell["area_pixels"]
            c_volume = tracked_cell["volume"]

            if i in matches:
                # Match existent
                best_match = self._previous_cells[matches[i]]
                cell_id = best_match["cell_id"]
                tracked_cell["cell_id"] = cell_id
                tracked_cell["is_tracked"] = True
                tracked_cell["centroid_history"] = list(best_match.get("centroid_history", []))
                tracked_cell["area_history"] = list(best_match.get("area_history", []))
                tracked_cell["cell_history"] = list(best_match.get("cell_history", []))

                p_volume = best_match.get("volume", 1.0)
                tracked_cell["volume_trend"] = c_volume / (p_volume + 1e-5)
                
                kf = self._kalman_bank[cell_id]
                pred_x_prior, pred_y_prior = kf.x, kf.y
                tracked_cell["prediction_error_pixels"] = np.sqrt(
                    (c_cell["centroid_x"] - pred_x_prior) ** 2 + (c_cell["centroid_y"] - pred_y_prior) ** 2
                )

                # Update Kalman cu noua locatie si arie
                kf.update(c_cell["centroid_x"], c_cell["centroid_y"], float(c_area))

                tracked_cell["centroid_history"].append((float(c_cell["centroid_y"]), float(c_cell["centroid_x"])))
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
                # Celula noua sau SPLIT
                cell_id = str(uuid.uuid4())[:8]
                tracked_cell["cell_id"] = cell_id
                tracked_cell["prediction_error_pixels"] = 0.0
                tracked_cell["volume_trend"] = 1.0
                tracked_cell["centroid_history"] = [(float(c_cell["centroid_y"]), float(c_cell["centroid_x"]))]
                tracked_cell["area_history"] = [int(c_area)]
                tracked_cell["cell_history"] = [{
                    "centroid_y": float(c_cell["centroid_y"]),
                    "centroid_x": float(c_cell["centroid_x"]),
                    "area_pixels": int(c_area),
                }]

                # Pass 2: Detectare SPLIT din orfani
                best_parent_dist = 1000.0
                inherited_vy, inherited_vx = 0.0, 0.0
                
                for p_cell in self._previous_cells:
                    p_id = p_cell.get("cell_id")
                    if p_id not in self._kalman_bank:
                        continue
                        
                    kf_parent = self._kalman_bank[p_id]
                    dist = np.sqrt((c_cell["centroid_x"] - kf_parent.x) ** 2 + (c_cell["centroid_y"] - kf_parent.y) ** 2)
                    
                    if dist <= (self._max_dist_pixels * 2.5) and dist < best_parent_dist:
                        best_parent_dist = dist
                        inherited_vx = kf_parent.v_x
                        inherited_vy = kf_parent.v_y

                self._kalman_bank[cell_id] = StormFilter(
                    initial_y=c_cell["centroid_y"], initial_x=c_cell["centroid_x"],
                    initial_vy=inherited_vy, initial_vx=inherited_vx,
                    initial_area=float(c_area), initial_d_area=0.0
                )
                active_ids.add(cell_id)

            # Viteza si dinamica din Kalman 8D
            kf_current = self._kalman_bank[cell_id]
            tracked_cell["v_x"] = kf_current.v_x
            tracked_cell["v_y"] = kf_current.v_y
            tracked_cell["a_x"] = kf_current.a_x
            tracked_cell["a_y"] = kf_current.a_y
            tracked_cell["predicted_area_kalman"] = max(1.0, kf_current.area)
            tracked_cell["d_area_kalman"] = kf_current.d_area
            tracked_cell["dd_area_kalman"] = kf_current.dd_area

            predicted_centroid_x = tracked_cell["centroid_x"] + tracked_cell["v_x"]
            predicted_centroid_y = tracked_cell["centroid_y"] + tracked_cell["v_y"]
            tracked_cell["predicted_centroid_x"] = float(predicted_centroid_x)
            tracked_cell["predicted_centroid_y"] = float(predicted_centroid_y)

            # --- Predictie masca morfologica ---
            coords = c_cell.get("coords")
            if flow is not None and coords is not None and len(coords) > 0:
                mean_flow_y = float(np.mean(flow[coords[:, 0], coords[:, 1], 1]))
                mean_flow_x = float(np.mean(flow[coords[:, 0], coords[:, 1], 0]))
                shift_x = 0.75 * tracked_cell["v_x"] + 0.25 * mean_flow_x
                shift_y = 0.75 * tracked_cell["v_y"] + 0.25 * mean_flow_y
            else:
                shift_x = tracked_cell["v_x"]
                shift_y = tracked_cell["v_y"]

            cached_mask = c_cell.get("_cached_mask")
            if cached_mask is not None:
                tracked_cell["predicted_mask"] = self.translate_mask(cached_mask, shift_y, shift_x)

            # Area trend logic
            trend = self._compute_area_trend(tracked_cell)
            tracked_cell["volume_trend"] = trend
            predicted_area_pixels = int(round(max(c_area, 1.0) * trend))
            tracked_cell["predicted_area_pixels"] = predicted_area_pixels
            tracked_cell["size_error_pixels"] = abs(predicted_area_pixels - int(c_area))
            tracked_cell["size_error_percent"] = 100.0 * tracked_cell["size_error_pixels"] / max(int(c_area), 1)
            
            tracked_cells.append(tracked_cell)

        # Cleanup: stergem filtrele Kalman pentru celulele moarte
        for old_id in list(self._kalman_bank.keys()):
            if old_id not in active_ids:
                del self._kalman_bank[old_id]

        self._previous_cells = tracked_cells
        self._previous_rain_matrix = rain_matrix.copy()

        return tracked_cells, flow

    @staticmethod
    def _compute_area_trend(tracked_cell: dict) -> float:
        if len(tracked_cell["area_history"]) >= 2:
            area_deltas = [
                max(tracked_cell["area_history"][idx], 1) / max(tracked_cell["area_history"][idx - 1], 1)
                for idx in range(1, len(tracked_cell["area_history"]))
            ]
            raw_area_trend = float(np.mean(area_deltas[-3:]))
        else:
            raw_area_trend = float(tracked_cell.get("area_trend", 1.0))

        if len(tracked_cell["cell_history"]) >= 3:
            recent_areas = [item["area_pixels"] for item in tracked_cell["cell_history"][-3:]]
            recent_area_trend = float(np.mean(recent_areas) / max(recent_areas[0], 1))
        else:
            recent_area_trend = raw_area_trend

        return float(np.clip(
            0.8 * recent_area_trend + 0.2 * tracked_cell.get("volume_trend", 1.0),
            0.90, 1.14,
        ))

    @staticmethod
    def translate_mask(mask: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
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
