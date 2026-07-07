import numpy as np

from src.core.detection.storm_cell_detector import StormCellDetector


def _one_cell(rain):
    cells = StormCellDetector(threshold=1.0, min_size=1).extract_cells(rain)
    assert len(cells) == 1
    return cells[0]


def test_uniform_cell_centroid_matches_geometric_center():
    rain = np.zeros((7, 7), dtype=float)
    rain[2:5, 2:5] = 2.0

    cell = _one_cell(rain)

    assert cell.centroid_y == 3.0
    assert cell.centroid_x == 3.0


def test_intense_core_shifts_centroid_toward_mass_center():
    rain = np.zeros((7, 7), dtype=float)
    rain[2:5, 2:5] = 1.0
    rain[2, 4] = 20.0

    cell = _one_cell(rain)

    assert cell.centroid_y < 3.0
    assert cell.centroid_x > 3.0


def test_invalid_centroid_weights_are_ignored_or_fall_back():
    coords = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    rain = np.array([
        [0.0, 0.0, 0.0],
        [0.0, np.nan, -1.0],
        [0.0, 0.0, 4.0],
    ])

    assert StormCellDetector._rain_weighted_centroid(rain, coords, (1.5, 1.5)) == (2.0, 2.0)

    rain[2, 2] = 0.0
    assert StormCellDetector._rain_weighted_centroid(rain, coords, (1.5, 1.5)) == (1.5, 1.5)
