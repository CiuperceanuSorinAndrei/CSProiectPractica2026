"""Modul de cinematica: urmarire centroizi folosind filtru Kalman 8D si KD-Tree Matcher."""
from __future__ import annotations

import uuid
import copy
from typing import Any

import numpy as np

from src.core.matcher import Matcher
from src.core.flow_estimator import FlowEstimator
from src.core.domain import StormCell
from src.core.kinematic_updater import KinematicUpdater
from src.core.cell_lifecycle import CellLifecycleManager
from src.core.reaction_diffusion import lifecycle


class StormTracker:
    USE_LOGISTIC_GROWTH = True
    USE_ADAPTIVE_KALMAN = True

    def __init__(self, max_dist_pixels: int = 15):
        self._max_dist_pixels = max_dist_pixels
        self._kinematic_updater = KinematicUpdater()
        self._previous_cells: list[StormCell] = []
        self._previous_rain_matrix: np.ndarray | None = None
        self._flow_estimator = FlowEstimator()

    @staticmethod
    def build_cell_mask(cell: StormCell, rain_matrix: np.ndarray) -> np.ndarray:
        """Extrage masca 2D a celulei. Foloseste cache pentru performanta."""
        if cell._cached_mask is not None:
            return cell._cached_mask
        mask = np.zeros_like(rain_matrix, dtype=np.uint8)
        coords = np.asarray(cell.coords)
        if len(coords) > 0:
            mask[coords[:, 0], coords[:, 1]] = 1
        cell._cached_mask = mask
        return mask

    def reset(self) -> None:
        self._kinematic_updater.reset()
        self._previous_cells = []
        self._previous_rain_matrix = None

    def track(self, current_cells: list[StormCell], rain_matrix: np.ndarray) -> tuple[list[StormCell], np.ndarray | None]:
        if self._previous_rain_matrix is not None and self._previous_rain_matrix.shape != rain_matrix.shape:
            self.reset()

        tracked_cells: list[StormCell] = []
        active_ids: set[str] = set()

        # 1. Optical flow global (DIS)
        flow = self._flow_estimator.compute(self._previous_rain_matrix, rain_matrix)

        # 2. Kalman predict (Constant Acceleration Model)
        self._kinematic_updater.predict_all()

        # Pregatim celulele curente
        for c_cell in current_cells:
            c_area = c_cell.area_pixels if c_cell.area_pixels else 1.0
            
            coords = c_cell.coords
            if coords is not None and len(coords) > 0:
                c_cell.volume = float(np.sum(rain_matrix[coords[:, 0], coords[:, 1]]))
                c_cell.mean_intensity = c_cell.volume / c_area
            else:
                c_cell.volume = 1.0
                c_cell.mean_intensity = 0.0
                
            c_cell.area_pixels = int(c_area)
            c_cell.E = c_cell.volume / 1000.0
            c_cell.dE = 0.0
            
            self.build_cell_mask(c_cell, rain_matrix)

        for p_cell in self._previous_cells:
            self.build_cell_mask(p_cell, rain_matrix)

        # 3. Construim matricea de asocieri folosind KD-Tree
        matches = Matcher.match_cells(
            current_cells, self._previous_cells, self._kinematic_updater._kalman_bank, self._max_dist_pixels
        )

        # 4. Procesam fiecare celula cu asocierea gasita
        for i, c_cell in enumerate(current_cells):
            tracked_cell = c_cell.clone()
            tracked_cell.is_tracked = False
            c_area = tracked_cell.area_pixels
            c_volume = tracked_cell.volume

            if i in matches:
                # Match existent
                best_match = self._previous_cells[matches[i]]
                cell_id = best_match.cell_id
                tracked_cell.cell_id = cell_id
                tracked_cell.is_tracked = True

                p_volume = best_match.volume or 1.0
                tracked_cell.volume_trend = c_volume / (p_volume + 1e-5)
                
                tracked_cell.E = c_cell.E
                tracked_cell.dE = c_cell.E - best_match.E
                
                pred_x_prior, pred_y_prior = self._kinematic_updater.get_prior_prediction(cell_id)
                tracked_cell.prediction_error_pixels = np.sqrt(
                    (c_cell.centroid_x - pred_x_prior) ** 2 + (c_cell.centroid_y - pred_y_prior) ** 2
                )

                self._kinematic_updater.update_cell(cell_id, c_cell, tracked_cell, float(c_area))
                CellLifecycleManager.transfer_history(c_cell, tracked_cell, best_match, float(c_area))
                active_ids.add(cell_id)
            else:
                # Celula noua sau SPLIT
                cell_id = str(uuid.uuid4())[:8]
                tracked_cell.cell_id = cell_id
                tracked_cell.prediction_error_pixels = 0.0
                tracked_cell.volume_trend = 1.0
                tracked_cell.E = c_cell.E
                tracked_cell.dE = 0.0
                
                CellLifecycleManager.transfer_history(c_cell, tracked_cell, None, float(c_area))

                # Pass 2: Detectare SPLIT din orfani
                best_parent_dist = 1000.0
                inherited_vy, inherited_vx = 0.0, 0.0
                
                for p_cell in self._previous_cells:
                    p_id = p_cell.cell_id
                    if not p_id or not self._kinematic_updater.is_tracked(p_id):
                        continue
                        
                    kf_parent = self._kinematic_updater.get_filter(p_id)
                    px = p_cell.predicted_centroid_x if p_cell.predicted_centroid_x else kf_parent.x
                    py = p_cell.predicted_centroid_y if p_cell.predicted_centroid_y else kf_parent.y
                    dist = np.sqrt((c_cell.centroid_x - px) ** 2 + (c_cell.centroid_y - py) ** 2)
                    
                    if dist <= (self._max_dist_pixels * 2.5) and dist < best_parent_dist:
                        best_parent_dist = dist
                        inherited_vx = kf_parent.v_x
                        inherited_vy = kf_parent.v_y

                self._kinematic_updater.register_new_cell(cell_id, c_cell, float(c_area), inherited_vx, inherited_vy)
                
                kf_current = self._kinematic_updater.get_filter(cell_id)
                tracked_cell.v_x = kf_current.v_x
                tracked_cell.v_y = kf_current.v_y
                tracked_cell.a_x = kf_current.a_x
                tracked_cell.a_y = kf_current.a_y
                tracked_cell.predicted_area_kalman = max(1.0, kf_current.area)
                tracked_cell.d_area_kalman = kf_current.d_area
                tracked_cell.dd_area_kalman = kf_current.dd_area
                tracked_cell.uncertainty_trace = kf_current.positional_uncertainty

                predicted_centroid_x = tracked_cell.centroid_x + tracked_cell.v_x + 0.5 * tracked_cell.a_x
                predicted_centroid_y = tracked_cell.centroid_y + tracked_cell.v_y + 0.5 * tracked_cell.a_y
                tracked_cell.predicted_centroid_x = float(predicted_centroid_x)
                tracked_cell.predicted_centroid_y = float(predicted_centroid_y)
                
                active_ids.add(cell_id)

            # --- Predictie masca morfologica ---
            coords = c_cell.coords
            if flow is not None and coords is not None and len(coords) > 0:
                mean_flow_y = float(np.mean(flow[coords[:, 0], coords[:, 1], 1]))
                mean_flow_x = float(np.mean(flow[coords[:, 0], coords[:, 1], 0]))
                
                # Adaptive Kinematic Weighting
                confidence = np.clip(10.0 / (10.0 + tracked_cell.uncertainty_trace), 0.1, 0.9)
                shift_x = confidence * tracked_cell.v_x + (1.0 - confidence) * mean_flow_x
                shift_y = confidence * tracked_cell.v_y + (1.0 - confidence) * mean_flow_y
            else:
                shift_x = tracked_cell.v_x
                shift_y = tracked_cell.v_y

            cached_mask = c_cell._cached_mask
            if cached_mask is not None:
                tracked_cell.predicted_mask = self.translate_mask(cached_mask, shift_y, shift_x)

            # Area trend logic
            trend = CellLifecycleManager.compute_area_trend(tracked_cell)
            tracked_cell.volume_trend = trend
            
            # Phase 4 Lifecycle logic
            tracked_cell.lifecycle_phase = lifecycle(tracked_cell.E, tracked_cell.dE)
            
            predicted_area_pixels = int(round(max(c_area, 1.0) * trend))
            tracked_cell.predicted_area_pixels = predicted_area_pixels
            tracked_cell.size_error_pixels = abs(predicted_area_pixels - int(c_area))
            tracked_cell.size_error_percent = 100.0 * tracked_cell.size_error_pixels / max(int(c_area), 1)
            
            tracked_cells.append(tracked_cell)

        # Cleanup: stergem filtrele Kalman pentru celulele moarte
        self._kinematic_updater.cleanup_inactive(active_ids)

        self._previous_cells = tracked_cells
        self._previous_rain_matrix = rain_matrix.copy()

        return tracked_cells, flow

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
