"""Modul de cinematica: urmarire centroizi folosind filtru Kalman 8D si KD-Tree Matcher."""
from __future__ import annotations

import uuid

import numpy as np

from src.core.tracking.matcher import Matcher

from src.core.domain import StormCell
from src.core.tracking.kinematic_updater import KinematicUpdater
from src.core.tracking.cell_lifecycle import CellLifecycleManager
from src.core.nowcast.reaction_diffusion import lifecycle


class StormTracker:
    USE_LOGISTIC_GROWTH = True
    USE_ADAPTIVE_KALMAN = True

    def __init__(self, max_dist_pixels: int = 15):
        self._max_dist_pixels = max_dist_pixels
        self._kinematic_updater = KinematicUpdater()
        self._previous_cells: list[StormCell] = []
        self._previous_rain_matrix: np.ndarray | None = None




    def reset(self) -> None:
        self._kinematic_updater.reset()
        self._previous_cells = []
        self._previous_rain_matrix = None

    def track(self, current_cells: list[StormCell], rain_matrix: np.ndarray) -> tuple[list[StormCell], np.ndarray | None]:
        if self._previous_rain_matrix is not None and self._previous_rain_matrix.shape != rain_matrix.shape:
            self.reset()

        tracked_cells: list[StormCell] = []
        active_ids: set[str] = set()

        # 2. Kalman predict (Constant Acceleration Model)
        self._kinematic_updater.predict_all()

        # 3. Pregatim celulele (volum, intensitate, energie)
        self._prepare_current_cells(current_cells, rain_matrix)

        # 4. Construim matricea de asocieri folosind KD-Tree
        matches = Matcher.match_cells(
            current_cells, self._previous_cells, self._kinematic_updater._kalman_bank, self._max_dist_pixels
        )

        # 5. Procesam fiecare celula cu asocierea gasita
        for i, c_cell in enumerate(current_cells):
            tracked_cell = c_cell.clone()
            tracked_cell.is_tracked = False

            if i in matches:
                cell_id = self._apply_matched_cell(c_cell, tracked_cell, self._previous_cells[matches[i]])
            else:
                cell_id = self._apply_new_or_split_cell(c_cell, tracked_cell)
            active_ids.add(cell_id)

            self._predict_cell_mask(c_cell, tracked_cell)
            self._finalize_cell_trend(tracked_cell)

            tracked_cells.append(tracked_cell)

        # Cleanup: stergem filtrele Kalman pentru celulele moarte
        self._kinematic_updater.cleanup_inactive(active_ids)

        self._previous_cells = tracked_cells
        self._previous_rain_matrix = rain_matrix.copy()

        return tracked_cells

    def _prepare_current_cells(self, current_cells: list[StormCell], rain_matrix: np.ndarray) -> None:
        """Calculeaza volum, intensitate medie, energie (E) si masca pentru fiecare celula curenta."""
        for c_cell in current_cells:
            c_area = c_cell.area_pixels if c_cell.area_pixels else 1.0

            coords = c_cell.coords
            if coords is not None and len(coords) > 0:
                c_cell.volume = float(np.nansum(rain_matrix[coords[:, 0], coords[:, 1]]))
                c_cell.mean_intensity = c_cell.volume / c_area
            else:
                c_cell.volume = 1.0
                c_cell.mean_intensity = 0.0

            c_cell.area_pixels = int(c_area)
            c_cell.E = c_cell.volume / 1000.0
            c_cell.dE = 0.0



    def _apply_matched_cell(self, c_cell: StormCell, tracked_cell: StormCell, best_match: StormCell) -> str:
        """Asociere existenta: preia ID-ul parintelui, actualizeaza Kalman si transfera istoricul."""
        c_area = tracked_cell.area_pixels
        c_volume = tracked_cell.volume

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
        return cell_id

    def _apply_new_or_split_cell(self, c_cell: StormCell, tracked_cell: StormCell) -> str:
        """Celula noua sau SPLIT: genereaza ID, mosteneste viteza unui parinte si initializeaza Kalman."""
        c_area = tracked_cell.area_pixels

        cell_id = str(uuid.uuid4())[:8]
        tracked_cell.cell_id = cell_id
        tracked_cell.prediction_error_pixels = 0.0
        tracked_cell.volume_trend = 1.0
        tracked_cell.E = c_cell.E
        tracked_cell.dE = 0.0

        CellLifecycleManager.transfer_history(c_cell, tracked_cell, None, float(c_area))

        inherited_vx, inherited_vy = self._inherit_parent_velocity(c_cell)
        self._kinematic_updater.register_new_cell(cell_id, c_cell, float(c_area), inherited_vx, inherited_vy)

        kf_current = self._kinematic_updater.get_filter(cell_id)
        tracked_cell.v_x = kf_current.v_x
        tracked_cell.v_y = kf_current.v_y
        tracked_cell.uncertainty_trace = kf_current.positional_uncertainty

        tracked_cell.predicted_centroid_x = float(tracked_cell.centroid_x + tracked_cell.v_x)
        tracked_cell.predicted_centroid_y = float(tracked_cell.centroid_y + tracked_cell.v_y)
        return cell_id

    def _inherit_parent_velocity(self, c_cell: StormCell) -> tuple[float, float]:
        """Pass 2: cel mai apropiat parinte urmarit doneaza viteza initiala (detectare SPLIT din orfani)."""
        best_parent_dist = 1000.0
        inherited_vx, inherited_vy = 0.0, 0.0

        for p_cell in self._previous_cells:
            p_id = p_cell.cell_id
            if not p_id or not self._kinematic_updater.is_tracked(p_id):
                continue

            kf_parent = self._kinematic_updater.get_filter(p_id)
            px = p_cell.predicted_centroid_x if p_cell.predicted_centroid_x else kf_parent.x
            py = p_cell.predicted_centroid_y if p_cell.predicted_centroid_y else kf_parent.y
            dist = np.sqrt((c_cell.centroid_x - px) ** 2 + (c_cell.centroid_y - py) ** 2)

            # Limita Mahalanobis 3-Sigma cu capat fizic
            actual_limit = np.clip(np.sqrt(max(kf_parent.positional_uncertainty, 1.0)) * 3.0, 10.0, 30.0)

            if dist <= actual_limit and dist < best_parent_dist:
                best_parent_dist = dist
                inherited_vx = kf_parent.v_x
                inherited_vy = kf_parent.v_y

        return inherited_vx, inherited_vy

    def _predict_cell_mask(self, c_cell: StormCell, tracked_cell: StormCell) -> None:
        """Prezice masca morfologica viitoare folosind doar Kalman velocity."""
        shift_x = tracked_cell.v_x
        shift_y = tracked_cell.v_y

        coords = c_cell.coords
        if coords is not None and len(coords) > 0:
            tracked_cell.predicted_coords = self.translate_coords(coords, shift_y, shift_x)

    @staticmethod
    def _finalize_cell_trend(tracked_cell: StormCell) -> None:
        """Calculeaza trendul de arie, faza de viata (Phase 4) si eroarea de dimensiune prezisa.
        
        IMPORTANT: volume_trend (calculat din raportul volumelor reale) NU este suprascris.
        area_trend (calculat din istoria ariei in pixeli) este folosit doar pentru predicted_area_pixels.
        """
        c_area = tracked_cell.area_pixels

        # Area trend - folosit DOAR pentru predictia morfologica (marimea celulei)
        area_trend = CellLifecycleManager.compute_area_trend(tracked_cell)

        tracked_cell.lifecycle_phase = lifecycle(tracked_cell.E, tracked_cell.dE)

        predicted_area_pixels = int(round(max(c_area, 1.0) * area_trend))
        tracked_cell.predicted_area_pixels = predicted_area_pixels
        tracked_cell.size_error_pixels = abs(predicted_area_pixels - int(c_area))
        tracked_cell.size_error_percent = 100.0 * tracked_cell.size_error_pixels / max(int(c_area), 1)

    @staticmethod
    def translate_coords(coords: np.ndarray | list, shift_y: float, shift_x: float) -> list:
        arr = np.asarray(coords)
        if len(arr) == 0:
            return []
        dst_y = np.rint(arr[:, 0] + shift_y).astype(int)
        dst_x = np.rint(arr[:, 1] + shift_x).astype(int)
        return list(zip(dst_y, dst_x))
