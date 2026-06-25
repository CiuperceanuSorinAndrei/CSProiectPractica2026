"""Modul de cinematica: urmarire centroizi (Kalman) + predictie morfologica (OpenCV Flow)."""
from __future__ import annotations

import uuid
from collections import OrderedDict
from typing import Any

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

_MGRID_CACHE_MAXSIZE = 20
_mgrid_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()


class StormTracker:
    # Setari Optimizate Activate Permanent (conform Benchmark)
    USE_LOGISTIC_GROWTH = True
    USE_ADAPTIVE_KALMAN = True
    USE_PARALLAX_CORRECTION = False

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
        if self._previous_rain_matrix is not None and self._previous_rain_matrix.shape != rain_matrix.shape:
            self.reset()

        tracked_cells: list[dict[str, Any]] = []
        active_ids: set[str] = set()

        # 1. Optical flow global
        flow = self._compute_optical_flow(self._previous_rain_matrix, rain_matrix)

        # 2. Kalman predict (pentru ambele modele)
        for kf_dict in self._kalman_bank.values():
            kf_dict["CV"].predict()
            kf_dict["CA"].predict()

        # Pregatim celulele curente (calcul volum, etc.)
        for c_cell in current_cells:
            if self.USE_PARALLAX_CORRECTION:
                c_cell["centroid_y"] += 2.0
                coords = c_cell.get("coords", np.array([]))
                if len(coords) > 0:
                    grid_h = rain_matrix.shape[0]
                    coords[:, 0] = np.clip(coords[:, 0] + 2, 0, grid_h - 1)

            c_area = c_cell.get("area_pixels", c_cell.get("area", 1.0))
            c_max = c_cell.get("max_intensity", 1.0)
            c_cell["volume"] = c_area * (c_max * 0.4)
            c_cell["area_pixels"] = int(c_area)

        # 3. Construim matricea de cost pentru asocieri (Hungarian Algorithm)
        num_curr = len(current_cells)
        num_prev = len(self._previous_cells)
        
        cost_matrix = np.full((num_curr, num_prev), 1000.0)
        
        for i, c_cell in enumerate(current_cells):
            c_area = c_cell["area_pixels"]
            c_volume = c_cell["volume"]
            
            for j, p_cell in enumerate(self._previous_cells):
                p_id = p_cell.get("cell_id")
                if p_id not in self._kalman_bank:
                    continue
                    
                # Pentru calculul de cost (distanta), folosim predictia modelului "best" curent
                best_model = self._kalman_bank[p_id]["best"]
                pred_x = self._kalman_bank[p_id][best_model].x[0, 0]
                pred_y = self._kalman_bank[p_id][best_model].x[1, 0]
                radius_limit = max(
                    self._max_dist_pixels,
                    self._adaptive_match_radius(p_cell.get("area_pixels", p_cell.get("area", 1.0))),
                )

                dist = self._calculate_distance(c_cell["centroid_y"], c_cell["centroid_x"], pred_y, pred_x)
                if dist > radius_limit:
                    continue
                    
                # Normalizam distanta pentru a o echilibra in ecuatia costului
                dist_norm = dist / radius_limit
                
                p_area = p_cell.get("area_pixels", p_cell.get("area", 1.0))
                area_ratio = min(c_area, p_area) / (max(c_area, p_area) + 1e-5)
                # area_penalty: 0.0 for identical, approaches 1.0 for very different
                area_penalty = 1.0 - area_ratio

                p_volume = p_cell.get("volume", p_area * (p_cell.get("max_intensity", 1.0) * 0.4))
                volume_ratio = min(c_volume, p_volume) / (max(c_volume, p_volume) + 1e-5)
                volume_penalty = 1.0 - volume_ratio

                iou = self._coords_iou(c_cell.get("coords"), p_cell.get("coords"))
                iou_penalty = 1.0 - iou

                hybrid_cost = dist_norm + (area_penalty * 0.5) + (volume_penalty * 0.5) + (iou_penalty * 1.5)
                cost_matrix[i, j] = hybrid_cost

        # Rezolvam asignarile globale optime
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = {}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 500.0:  # Prag valid de asociere
                matches[r] = c

        # 4. Procesam fiecare celula cu asocierea gasita
        for i, c_cell in enumerate(current_cells):
            tracked_cell = c_cell.copy()
            tracked_cell["is_tracked"] = False
            tracked_cell["centroid_history"] = list(c_cell.get("centroid_history", []))
            tracked_cell["area_history"] = list(c_cell.get("area_history", []))
            tracked_cell["cell_history"] = list(c_cell.get("cell_history", []))
            tracked_cell["predicted_centroid_x"] = float(c_cell.get("centroid_x", 0.0))
            tracked_cell["predicted_centroid_y"] = float(c_cell.get("centroid_y", 0.0))
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
                prev_area = float(best_match.get("area_pixels", c_area))
                base_trend = c_area / (prev_area + 1e-5)
                
                if self.USE_LOGISTIC_GROWTH:
                    max_area = 50.0  # Furtunile locale sunt mult mai mici in pixeli (aprox 3-4km/pixel)
                    damping = max(0.0, 1.0 - (c_area / max_area))
                    tracked_cell["area_trend"] = 1.0 + (base_trend - 1.0) * damping
                else:
                    tracked_cell["area_trend"] = base_trend

                kf_dict = self._kalman_bank[cell_id]
                best_model = kf_dict["best"]
                pred_x_prior, pred_y_prior = kf_dict[best_model].x[0, 0], kf_dict[best_model].x[1, 0]
                tracked_cell["prediction_error_pixels"] = self._calculate_distance(
                    c_cell["centroid_y"], c_cell["centroid_x"], pred_y_prior, pred_x_prior,
                )

                obs_z = np.array([[c_cell["centroid_x"]], [c_cell["centroid_y"]]])

                if self.USE_ADAPTIVE_KALMAN and tracked_cell["prediction_error_pixels"] > 1.5:
                    # Incredere maxima in masuratoare pt a prinde virajul
                    kf_dict["CV"].R /= 10.0
                    kf_dict["CA"].R /= 10.0
                    kf_dict["CV"].update(obs_z)
                    kf_dict["CA"].update(obs_z)
                    kf_dict["CV"].R *= 10.0
                    kf_dict["CA"].R *= 10.0
                else:
                    kf_dict["CV"].update(obs_z)
                    kf_dict["CA"].update(obs_z)

                # Switching: Alegem modelul care a avut eroarea (residual y) mai mica
                err_cv = np.linalg.norm(kf_dict["CV"].y)
                err_ca = np.linalg.norm(kf_dict["CA"].y)
                kf_dict["best"] = "CV" if err_cv <= err_ca else "CA"
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
                tracked_cell["area_trend"] = 1.0

                # --- PASS 2: Detectare SPLIT ---
                # Cautam daca "orfanul" a aparut in perimetrul unei celule trecute
                best_parent_dist = 1000.0
                inherited_vy, inherited_vx = 0.0, 0.0
                
                for p_cell in self._previous_cells:
                    p_id = p_cell.get("cell_id")
                    if p_id not in self._kalman_bank:
                        continue
                        
                    best_model = self._kalman_bank[p_id]["best"]
                    pred_x = self._kalman_bank[p_id][best_model].x[0, 0]
                    pred_y = self._kalman_bank[p_id][best_model].x[1, 0]
                    
                    dist = self._calculate_distance(
                        c_cell["centroid_y"], c_cell["centroid_x"], pred_y, pred_x
                    )
                    
                    if dist <= self._max_dist_pixels and dist < best_parent_dist:
                        best_parent_dist = dist
                        inherited_vx = self._kalman_bank[p_id][best_model].x[2, 0]
                        inherited_vy = self._kalman_bank[p_id][best_model].x[3, 0]

                # Initializam cu viteza mostenita (daca e 0.0, 0.0 inseamna nastere normala)
                self._kalman_bank[cell_id] = self._instantiate_kalman_filter(
                    c_cell["centroid_y"], c_cell["centroid_x"],
                    inherited_vy, inherited_vx
                )
                active_ids.add(cell_id)

            # --- Viteza din Kalman + observatii ---
            kf_dict = self._kalman_bank[cell_id]
            best_model = kf_dict["best"]
            current_kf = kf_dict[best_model]
            
            tracked_cell["v_x"] = current_kf.x[2, 0]
            tracked_cell["v_y"] = current_kf.x[3, 0]

            if len(tracked_cell["centroid_history"]) >= 2:
                history = tracked_cell["centroid_history"]
                deltas_x = [history[idx][1] - history[idx - 1][1] for idx in range(1, len(history))]
                deltas_y = [history[idx][0] - history[idx - 1][0] for idx in range(1, len(history))]

                obs_v_x = float(np.mean(deltas_x[-3:]))
                obs_v_y = float(np.mean(deltas_y[-3:]))
                tracked_cell["v_x"] = 0.4 * tracked_cell["v_x"] + 0.6 * obs_v_x
                tracked_cell["v_y"] = 0.4 * tracked_cell["v_y"] + 0.6 * obs_v_y
                # Sincronizam inapoi viteza observata doar in modelul curent ales
                current_kf.x[2, 0] = tracked_cell["v_x"]
                current_kf.x[3, 0] = tracked_cell["v_y"]

            predicted_centroid_x = tracked_cell["centroid_x"] + tracked_cell["v_x"]
            predicted_centroid_y = tracked_cell["centroid_y"] + tracked_cell["v_y"]
            tracked_cell["predicted_centroid_x"] = float(predicted_centroid_x)
            tracked_cell["predicted_centroid_y"] = float(predicted_centroid_y)

            # --- Predictie masca morfologica cu optimizare Bounding Box ---
            coords = c_cell.get("coords")
            if flow is not None and coords is not None and len(coords) > 0:
                # Optimizare memorie extremă: selectam flow-ul strict de pe pixeli
                mean_flow_y = float(np.mean(flow[coords[:, 0], coords[:, 1], 1]))
                mean_flow_x = float(np.mean(flow[coords[:, 0], coords[:, 1], 0]))
                shift_x = 0.75 * tracked_cell["v_x"] + 0.25 * mean_flow_x
                shift_y = 0.75 * tracked_cell["v_y"] + 0.25 * mean_flow_y
            else:
                shift_x = tracked_cell["v_x"]
                shift_y = tracked_cell["v_y"]

            predicted_area_trend = self._compute_area_trend(tracked_cell)
            scale_factor = float(np.clip(np.sqrt(predicted_area_trend), 0.97, 1.05))
            predicted_area_pixels = int(round(max(c_area, 1.0) * predicted_area_trend))

            predicted_mask = self._predict_mask_shape(
                coords, rain_matrix.shape,
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

        # Cleanup: stergem filtrele Kalman pentru celulele moarte
        for old_id in list(self._kalman_bank.keys()):
            if old_id not in active_ids:
                del self._kalman_bank[old_id]

        self._previous_cells = tracked_cells
        self._previous_rain_matrix = rain_matrix.copy()

        return tracked_cells

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
    def _coords_iou(coords_a: np.ndarray | list | None, coords_b: np.ndarray | list | None) -> float:
        """Calculeaza Intersection over Union de mii de ori mai rapid direct pe liste de coordonate."""
        if coords_a is None or coords_b is None or len(coords_a) == 0 or len(coords_b) == 0:
            return 0.0
        set_a = set(map(tuple, coords_a))
        set_b = set(map(tuple, coords_b))
        intersection = len(set_a.intersection(set_b))
        if intersection == 0:
            return 0.0
        union = len(set_a.union(set_b))
        return float(intersection / union)

    @staticmethod
    def _predict_mask_shape(
        coords: np.ndarray | None,
        grid_shape: tuple[int, int],
        center_y: float, center_x: float,
        shift_y: float, shift_x: float,
        scale_factor: float,
    ) -> np.ndarray | None:
        """Prezice forma deformand doar Bounding Box-ul curent, salvand zeci de MB de memorie per celula."""
        if coords is None or len(coords) == 0:
            return None

        h, w = grid_shape
        # Gasim Bounding Box-ul stramt
        min_y, max_y = np.min(coords[:, 0]), np.max(coords[:, 0])
        min_x, max_x = np.min(coords[:, 1]), np.max(coords[:, 1])

        # Adaugam un padding generos pentru a permite miscarea si extinderea (ex. max shift 30 pixeli + 20 px padding)
        pad = int(max(abs(shift_y), abs(shift_x)) + 20)
        y0 = max(0, min_y - pad)
        y1 = min(h, max_y + pad + 1)
        x0 = max(0, min_x - pad)
        x1 = min(w, max_x + pad + 1)

        bb_h, bb_w = y1 - y0, x1 - x0
        if bb_h <= 0 or bb_w <= 0:
            return None

        # Construim o masca DOAR pe Bounding Box
        local_mask = np.zeros((bb_h, bb_w), dtype=np.float32)
        local_mask[coords[:, 0] - y0, coords[:, 1] - x0] = 1.0

        if (bb_h, bb_w) not in _mgrid_cache:
            if len(_mgrid_cache) >= _MGRID_CACHE_MAXSIZE:
                _mgrid_cache.popitem(last=False)
            _mgrid_cache[(bb_h, bb_w)] = np.mgrid[0:bb_h, 0:bb_w].astype(np.float32)
        else:
            _mgrid_cache.move_to_end((bb_h, bb_w))

        y_coords, x_coords = _mgrid_cache[(bb_h, bb_w)]
        
        map_x = (x_coords - shift_x).astype(np.float32)
        map_y = (y_coords - shift_y).astype(np.float32)
        
        warped_local = cv2.remap(
            local_mask, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        if abs(scale_factor - 1.0) > 1e-4:
            # Centrul local in Bounding Box
            local_center_x = center_x - x0
            local_center_y = center_y - y0
            matrix = np.array([
                [scale_factor, 0.0, (1.0 - scale_factor) * local_center_x],
                [0.0, scale_factor, (1.0 - scale_factor) * local_center_y],
            ], dtype=np.float32)
            
            warped_local = cv2.warpAffine(
                warped_local, matrix, (bb_w, bb_h),
                flags=cv2.INTER_LINEAR, borderValue=0,
            )
            
        # Reasamblam masca prezisa pe full grid? 
        # Nu, orchestrator-ul probabil are nevoie de masca intreaga pentru plot.
        # Daca orchestrator-ul o deseneaza, se bazeaza pe dimensiunea rain_matrix!
        full_mask = np.zeros(grid_shape, dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = (warped_local > 0.3).astype(np.uint8)
        return full_mask

    @staticmethod
    def _instantiate_kalman_filter(
        initial_y: float, initial_x: float,
        initial_vy: float = 0.0, initial_vx: float = 0.0
    ) -> dict[str, Any]:
        kf_cv = KalmanFilter(dim_x=4, dim_z=2)
        kf_cv.x = np.array([[initial_x], [initial_y], [initial_vx], [initial_vy]])
        kf_cv.F = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        kf_cv.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        kf_cv.P *= 1.2
        kf_cv.Q = np.array([[0.005, 0.0, 0.0, 0.0], [0.0, 0.005, 0.0, 0.0], [0.0, 0.0, 0.02, 0.0], [0.0, 0.0, 0.0, 0.02]])
        kf_cv.R = np.array([[10.0, 0.0], [0.0, 10.0]])

        kf_ca = KalmanFilter(dim_x=6, dim_z=2)
        kf_ca.x = np.array([[initial_x], [initial_y], [initial_vx], [initial_vy], [0.0], [0.0]])
        kf_ca.F = np.array([
            [1., 0., 1., 0., 0.5, 0. ],
            [0., 1., 0., 1., 0.,  0.5],
            [0., 0., 1., 0., 1.,  0. ],
            [0., 0., 0., 1., 0.,  1. ],
            [0., 0., 0., 0., 1.,  0. ],
            [0., 0., 0., 0., 0.,  1. ]
        ])
        kf_ca.H = np.array([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]])
        kf_ca.P *= 1.2
        kf_ca.Q = np.eye(6) * 0.01
        kf_ca.R = np.array([[10.0, 0.0], [0.0, 10.0]])

        return {"CV": kf_cv, "CA": kf_ca, "best": "CV"}

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

    @staticmethod
    def _compute_optical_flow(previous_rain: np.ndarray | None, current_rain: np.ndarray) -> np.ndarray | None:
        if previous_rain is None or previous_rain.shape != current_rain.shape:
            return None
        prev_img = np.clip(previous_rain * 10, 0, 255).astype(np.uint8)
        curr_img = np.clip(current_rain * 10, 0, 255).astype(np.uint8)
        h, w = prev_img.shape
        prev_small = cv2.resize(prev_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        flow_small = cv2.calcOpticalFlowFarneback(
            prev_small, curr_small, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow_full = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return flow_full * 2.0

    @staticmethod
    def _calculate_distance(y1: float, x1: float, y2: float, x2: float) -> float:
        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    @staticmethod
    def _adaptive_match_radius(area_pixels: float) -> float:
        return float(np.clip(10.0 + np.sqrt(max(area_pixels, 1.0)) * 0.9, 14.0, 32.0))
