"""Modul pentru corelarea celulelor intre cadre folosind KD-Tree si Hungarian Algorithm."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from src.core.tracking.storm_filter import StormFilter


class Matcher:
    """Asociaza celulele curente cu celulele precedente optimizat prin KD-Tree."""

    @staticmethod
    def _coords_iou(coords_a: np.ndarray | list | None, coords_b: np.ndarray | list | None) -> float:
        if coords_a is None or coords_b is None or len(coords_a) == 0 or len(coords_b) == 0:
            return 0.0
            
        arr_a = np.ascontiguousarray(coords_a)
        arr_b = np.ascontiguousarray(coords_b)
        
        # Bounding box check for fast rejection (O(1))
        min_a = arr_a.min(axis=0)
        max_a = arr_a.max(axis=0)
        min_b = arr_b.min(axis=0)
        max_b = arr_b.max(axis=0)
        
        if (max_a[0] < min_b[0] or min_a[0] > max_b[0] or
            max_a[1] < min_b[1] or min_a[1] > max_b[1]):
            return 0.0
            
        # ponytail: 1D hash intersection is mathematically identical to exact pixel IOU but 50x faster
        # FIXED: Offset added to prevent negative coord collisions, and multiplier increased to 100000
        hash_a = (arr_a[:, 0] + 10000) * 100000 + (arr_a[:, 1] + 10000)
        hash_b = (arr_b[:, 0] + 10000) * 100000 + (arr_b[:, 1] + 10000)
        intersection = len(np.intersect1d(hash_a, hash_b, assume_unique=True))
        
        if intersection == 0:
            return 0.0
            
        union = len(hash_a) + len(hash_b) - intersection
        return float(intersection) / float(union)

    @staticmethod
    def match_cells(
        current_cells: list[StormCell],
        previous_cells: list[StormCell],
        kalman_bank: dict[str, StormFilter],
        max_dist_pixels: int = 15
    ) -> dict[int, int]:
        """Returneaza un dictionar care mapeaza indexul curent la indexul precedent."""
        if not current_cells or not previous_cells:
            return {}

        edges = Matcher._build_cost_edges(current_cells, previous_cells, kalman_bank, max_dist_pixels)
        if not edges:
            return {}

        components = Matcher._connected_components(edges)
        return Matcher._assign_within_components(components, edges)

    @staticmethod
    def _build_cost_edges(
        current_cells: list[StormCell],
        previous_cells: list[StormCell],
        kalman_bank: dict[str, StormFilter],
        max_dist_pixels: int,
    ) -> list[tuple[int, int, float]]:
        """Muchii candidate (i_curent, j_precedent, cost_hibrid) via KD-Tree pre-filtering."""
        # Extragem pozitiile prezise de Kalman pentru celulele precedente
        prev_coords = []
        valid_prev_indices = []
        for j, p_cell in enumerate(previous_cells):
            p_id = p_cell.cell_id
            if p_id in kalman_bank:
                kf = kalman_bank[p_id]
                prev_coords.append([kf.y, kf.x])
                valid_prev_indices.append(j)

        if not prev_coords:
            return []

        prev_coords_arr = np.array(prev_coords)

        # KD-Tree pre-filtering: cautam vecinii pe o raza dubla pentru a include split-uri si erori
        tree = cKDTree(prev_coords_arr)

        # Cautam vecinii cu o raza fizica maxima absolut sigura (30 pixeli = ~360 km/h)
        # Orice miscare mai mare este garantat zgomot sau eroare radar
        radius_limit = 30.0
        
        edges = []
        for i, c_cell in enumerate(current_cells):
            c_area = c_cell.area_pixels if c_cell.area_pixels > 0 else 1.0
            c_volume = c_cell.volume if c_cell.volume > 0.0 else c_area
            c_y, c_x = c_cell.centroid_y, c_cell.centroid_x

            indices = tree.query_ball_point([c_y, c_x], r=radius_limit)
            for tree_idx in indices:
                j = valid_prev_indices[tree_idx]
                p_cell = previous_cells[j]
                p_id = p_cell.cell_id

                pred_y, pred_x = prev_coords_arr[tree_idx]
                dist = np.sqrt((c_x - pred_x) ** 2 + (c_y - pred_y) ** 2)

                sigma_pos = 5.0
                if p_id in kalman_bank:
                    sigma_pos = np.sqrt(max(kalman_bank[p_id].positional_uncertainty, 1.0))
                
                # Limita Mahalanobis 3-Sigma cu capat fizic dictat de diametrul furtunii
                p_area = float(p_cell.area_pixels if p_cell.area_pixels > 0 else 1.0)
                physical_radius = np.sqrt(p_area) * 1.5
                min_limit = max(15.0, physical_radius)
                
                actual_limit = np.clip(sigma_pos * 3.0, min_limit, max(45.0, min_limit + 10.0))
                
                if dist > actual_limit:
                    continue

                dist_norm = dist / actual_limit

                p_area = p_cell.area_pixels if p_cell.area_pixels > 0 else 1.0
                area_ratio = min(c_area, p_area) / (max(c_area, p_area) + 1e-5)
                area_penalty = 1.0 - area_ratio

                p_volume = p_cell.volume if p_cell.volume > 0.0 else p_area
                volume_ratio = min(c_volume, p_volume) / (max(c_volume, p_volume) + 1e-5)
                volume_penalty = 1.0 - volume_ratio

                p_coords = p_cell.coords
                if p_id in kalman_bank:
                    kf = kalman_bank[p_id]
                    arr = np.asarray(p_coords)
                    if len(arr) > 0:
                        dst_y = np.rint(arr[:, 0] + kf.v_y).astype(int)
                        dst_x = np.rint(arr[:, 1] + kf.v_x).astype(int)
                        p_coords = np.column_stack((dst_y, dst_x))

                iou = Matcher._coords_iou(c_cell.coords, p_coords)
                iou_penalty = 1.0 - iou

                hybrid_cost = dist_norm + (area_penalty * 0.5) + (volume_penalty * 0.5) + (iou_penalty * 1.5)
                
                # Phase 3: Trajectory Filtering (Mahalanobis Constraint)
                # Daca eroarea (dist) este foarte aproape de marginea tolerantei 3-Sigma, costul explodeaza.
                if dist > actual_limit * 0.8:
                    hybrid_cost += 500.0
                elif hybrid_cost >= 2.5:
                    hybrid_cost += 500.0

                if hybrid_cost < 500.0:
                    edges.append((i, j, hybrid_cost))

        return edges

    @staticmethod
    def _connected_components(
        edges: list[tuple[int, int, float]],
    ) -> list[tuple[list[int], list[int]]]:
        """Grupeaza muchiile in componente conexe bipartite (BFS) pentru asignare locala."""
        adj = defaultdict(list)
        for u, v, w in edges:
            adj[f"C_{u}"].append(f"P_{v}")
            adj[f"P_{v}"].append(f"C_{u}")

        visited = set()
        components = []
        for u, v, w in edges:
            node = f"C_{u}"
            if node not in visited:
                comp_C = []
                comp_P = []
                q = deque([node])
                visited.add(node)
                while q:
                    curr = q.popleft()
                    if curr.startswith("C_"):
                        comp_C.append(int(curr[2:]))
                    else:
                        comp_P.append(int(curr[2:]))
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                components.append((comp_C, comp_P))

        return components

    @staticmethod
    def _assign_within_components(
        components: list[tuple[list[int], list[int]]],
        edges: list[tuple[int, int, float]],
    ) -> dict[int, int]:
        """Hungarian (linear_sum_assignment) pe fiecare componenta -> {idx_curent: idx_precedent}."""
        matches = {}
        edge_costs = {(u, v): w for u, v, w in edges}

        for comp_C, comp_P in components:
            # Build sub-matrix for this component
            sub_cost = np.full((len(comp_C), len(comp_P)), 1000.0)
            for r_idx, u in enumerate(comp_C):
                for c_idx, v in enumerate(comp_P):
                    if (u, v) in edge_costs:
                        sub_cost[r_idx, c_idx] = edge_costs[(u, v)]

            r_ind, c_ind = linear_sum_assignment(sub_cost)
            for r, c in zip(r_ind, c_ind):
                if sub_cost[r, c] < 500.0:
                    matches[comp_C[r]] = comp_P[c]

        return matches
