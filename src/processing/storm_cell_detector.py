import numpy as np
import scipy.ndimage as ndi


class StormCellDetector:
    _threshold: float = None
    _min_size: int = None

    def __init__(self, threshold: float = 0.5, min_size: int = 5):
        self.set_params(threshold, min_size)

    def set_params(self, threshold: float = 0.5, min_size: int = 5):
        self._threshold = threshold
        self._min_size = min_size

    # Detecteaza nucleele de furtuna pe baza unui prag de precipitatii
    def extract_cells(self, rain_matrix: np.ndarray) -> list[dict]:
        labeled_mask, num_features = self._build_label_mask(rain_matrix)

        cells = []
        if num_features == 0:
            return cells

        # Calcul centre de greutate pentru fiecare formatiune
        centroids = ndi.center_of_mass(rain_matrix, labeled_mask, range(1, num_features + 1))

        for i in range(1, num_features + 1):
            cell_pixels = np.sum(labeled_mask == i)

            # Ignoram formatiunile prea mici
            if cell_pixels < self._min_size:
                continue

            y_center, x_center = centroids[i - 1]
            max_intensity = np.max(rain_matrix[labeled_mask == i])

            cells.append({
                "id": i,
                "centroid_y": y_center,
                "centroid_x": x_center,
                "area_pixels": cell_pixels,
                "max_intensity": max_intensity
            })

        return cells

    # Binarizeaza, curata zgomotul si eticheteaza formatiunile conectate
    def _build_label_mask(self, rain_matrix: np.ndarray) -> tuple[np.ndarray, int]:
        # Binarizare: pastram doar pixelii care depasesc pragul de ploaie
        binary_mask = rain_matrix >= self._threshold

        # Eliminare zgomot: stergem pixelii izolati
        clean_mask = ndi.binary_opening(binary_mask, structure=np.ones((3, 3)))

        # Etichetare: identificam grupurile de pixeli conectati
        return ndi.label(clean_mask)


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
        print(cell)
