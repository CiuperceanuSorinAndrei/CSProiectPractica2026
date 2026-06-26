"""Modul pentru corelarea celulelor intre cadre folosind KD-Tree si Hungarian Algorithm."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from src.core.storm_filter import StormFilter


class Matcher:
    """Asociaza celulele curente cu celulele precedente optimizat prin KD-Tree."""

    @staticmethod
    def _coords_iou(coords_a: np.ndarray | list | None, coords_b: np.ndarray | list | None) -> float:
        if coords_a is None or coords_b is None or len(coords_a) == 0 or len(coords_b) == 0:
            return 0.0
        arr_a = np.asarray(coords_a)
        arr_b = np.asarray(coords_b)
        
        min_a, max_a = np.min(arr_a, axis=0), np.max(arr_a, axis=0)
        min_b, max_b = np.min(arr_b, axis=0), np.max(arr_b, axis=0)
        if (min_a[0] > max_b[0] or max_a[0] < min_b[0] or 
            min_a[1] > max_b[1] or max_a[1] < min_b[1]):
            return 0.0

        min_y, min_x = min(min_a[0], min_b[0]), min(min_a[1], min_b[1])
        max_y, max_x = max(max_a[0], max_b[0]), max(max_a[1], max_b[1])
        
        h = max_y - min_y + 1
        w = max_x - min_x + 1
        
        mask_a = np.zeros((h, w), dtype=bool)
        mask_b = np.zeros((h, w), dtype=bool)
        
        mask_a[arr_a[:, 0] - min_y, arr_a[:, 1] - min_x] = True
        mask_b[arr_b[:, 0] - min_y, arr_b[:, 1] - min_x] = True
        
        intersection = float(np.sum(mask_a & mask_b))
        if intersection == 0:
            return 0.0
        union = float(np.sum(mask_a | mask_b))
        return intersection / union

    @staticmethod
    def match_cells(
        current_cells: list[dict],
        previous_cells: list[dict],
        kalman_bank: dict[str, StormFilter],
        max_dist_pixels: int = 15
    ) -> dict[int, int]:
        """Returneaza un dictionar care mapeaza indexul curent la indexul precedent."""
        num_curr = len(current_cells)
        num_prev = len(previous_cells)
        
        if num_curr == 0 or num_prev == 0:
            return {}

        # Extragem pozitiile prezise de Kalman pentru celulele precedente
        prev_coords = []
        valid_prev_indices = []
        for j, p_cell in enumerate(previous_cells):
            p_id = p_cell.get("cell_id")
            if p_id in kalman_bank:
                kf = kalman_bank[p_id]
                prev_coords.append([kf.y, kf.x])
                valid_prev_indices.append(j)

        if not prev_coords:
            return {}

        prev_coords_arr = np.array(prev_coords)
        curr_coords_arr = np.array([[c["centroid_y"], c["centroid_x"]] for c in current_cells])

        # KD-Tree pre-filtering
        tree = cKDTree(prev_coords_arr)
        
        # Cautam toti vecinii pe o raza dubla pentru a include split-uri si erori
        radius_limit = max_dist_pixels * 2.0
        
        cost_matrix = np.full((num_curr, num_prev), 1000.0)
        
        for i, c_cell in enumerate(current_cells):
            c_area = c_cell.get("area_pixels", 1.0)
            c_volume = c_cell.get("volume", c_area)
            c_y, c_x = c_cell["centroid_y"], c_cell["centroid_x"]
            
            # Query the KDTree
            indices = tree.query_ball_point([c_y, c_x], r=radius_limit)
            
            for tree_idx in indices:
                j = valid_prev_indices[tree_idx]
                p_cell = previous_cells[j]
                
                pred_y, pred_x = prev_coords_arr[tree_idx]
                
                dist = np.sqrt((c_x - pred_x) ** 2 + (c_y - pred_y) ** 2)
                
                adaptive_radius = np.clip(10.0 + np.sqrt(max(float(p_cell.get("area_pixels", 1.0)), 1.0)) * 0.9, 14.0, 32.0)
                actual_limit = max(max_dist_pixels, adaptive_radius)
                
                if dist > actual_limit:
                    continue
                    
                dist_norm = dist / actual_limit
                
                p_area = p_cell.get("area_pixels", 1.0)
                area_ratio = min(c_area, p_area) / (max(c_area, p_area) + 1e-5)
                area_penalty = 1.0 - area_ratio

                p_volume = p_cell.get("volume", p_area)
                volume_ratio = min(c_volume, p_volume) / (max(c_volume, p_volume) + 1e-5)
                volume_penalty = 1.0 - volume_ratio

                iou = Matcher._coords_iou(c_cell.get("coords"), p_cell.get("coords"))
                iou_penalty = 1.0 - iou

                hybrid_cost = dist_norm + (area_penalty * 0.5) + (volume_penalty * 0.5) + (iou_penalty * 1.5)
                cost_matrix[i, j] = hybrid_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = {}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 500.0:  # Prag valid de asociere
                matches[r] = c

        return matches
