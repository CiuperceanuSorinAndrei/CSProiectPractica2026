from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


class StormCellDetector:
    _threshold: float = None
    _min_size: int = None
    _small_cell_threshold: float = None
    _large_cell_threshold: float = None
    _small_cell_max_area: int = None

    def __init__(
        self,
        threshold: float = 0.5,
        min_size: int = 5,
        small_cell_threshold: float | None = None,
        large_cell_threshold: float | None = None,
        small_cell_max_area: int | None = None,
    ):
        self._threshold = threshold
        self._min_size = min_size
        self._small_cell_threshold = small_cell_threshold
        self._large_cell_threshold = large_cell_threshold
        self._small_cell_max_area = small_cell_max_area

    # Detecteaza nucleele de furtuna folosind doua praguri (mare si mic) pentru a
    # prinde atat celulele principale cat si cele mici care nu se suprapun cu ele.
    def extract_cells(self, rain_matrix: np.ndarray) -> list[dict]:
        small_thr = self._threshold if self._small_cell_threshold is None else self._small_cell_threshold
        large_thr = self._threshold if self._large_cell_threshold is None else self._large_cell_threshold

        struct = np.ones((3, 3))
        large_mask = ndi.binary_opening(rain_matrix >= large_thr, structure=struct)
        
        # Etichetare simpla a componentelor conexe
        large_labels = self._label_connected_components(rain_matrix, large_mask)
        cells = self._extract_components_from_labels(rain_matrix, large_labels, self._min_size)

        # Daca pragurile sunt identice, Pass 2 e inutil (risipa de CPU)
        if abs(small_thr - large_thr) < 1e-5 and self._small_cell_max_area is None:
            return cells

        # Construim masca de pixeli deja acoperiti de celulele mari (vectorizat)
        seen_mask = np.zeros(rain_matrix.shape, dtype=bool)
        for cell in cells:
            coords = np.asarray(cell["coords"])
            if len(coords) > 0:
                seen_mask[coords[:, 0], coords[:, 1]] = True

        # Adaugam celulele mici care nu se suprapun
        small_mask = ndi.binary_opening(rain_matrix >= small_thr, structure=struct)
        small_labels = self._label_connected_components(rain_matrix, small_mask)
        small_cells = self._extract_components_from_labels(
            rain_matrix, small_labels, self._min_size, max_area=self._small_cell_max_area,
        )

        next_id = len(cells) + 1
        for cell in small_cells:
            coords = np.asarray(cell["coords"])
            # Daca exista ORICE pixel care e deja in seen_mask, ignoram celula
            if len(coords) > 0 and np.any(seen_mask[coords[:, 0], coords[:, 1]]):
                continue
            cell["id"] = next_id
            next_id += 1
            cells.append(cell)

        return cells

    @staticmethod
    def _label_connected_components(rain_matrix: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """Aplica etichetarea componentelor conexe (Connected Component Labeling)."""
        labels, _ = ndi.label(base_mask)
        return labels

    @staticmethod
    def _extract_components_from_labels(
        rain_matrix: np.ndarray,
        labeled_mask: np.ndarray,
        min_size: int,
        max_area: int | None = None,
    ) -> list[dict]:
        labels = np.unique(labeled_mask)
        labels = labels[labels > 0]
        if len(labels) == 0:
            return []

        mask = labeled_mask > 0
        areas = ndi.sum_labels(mask, labeled_mask, labels).astype(int)
        centroids = ndi.center_of_mass(rain_matrix, labeled_mask, labels)
        max_intensities = ndi.maximum(rain_matrix, labeled_mask, labels)
        mean_intensities = ndi.mean(rain_matrix, labeled_mask, labels)

        cells = []
        for i, label in enumerate(labels):
            cell_pixels = int(areas[i])
            if cell_pixels < min_size:
                continue
            if max_area is not None and cell_pixels > max_area:
                continue

            y_center, x_center = centroids[i]
            coords = np.argwhere(labeled_mask == label)

            cells.append({
                "id": int(label),
                "centroid_y": float(y_center),
                "centroid_x": float(x_center),
                "area_pixels": cell_pixels,
                "max_intensity": float(max_intensities[i]),
                "mean_intensity": float(mean_intensities[i]),
                "coords": coords,
            })

        return cells


# --- Testing ---
if __name__ == "__main__":
    # Matrice sintetica cu doua formatiuni de ploaie
    rain_matrix = np.zeros((20, 20))
    rain_matrix[3:8, 3:8] = 4.5      # celula mare (peste min_size)
    rain_matrix[14:16, 14:16] = 2.0  # celula mica (sub min_size)

    detector = StormCellDetector(threshold=0.5, min_size=5)
    cells = detector.extract_cells(rain_matrix)

    print(f"Celule detectate: {len(cells)}")
    for cell in cells:
        print({k: v for k, v in cell.items() if k != "coords"}, "| coords:", len(cell["coords"]))
