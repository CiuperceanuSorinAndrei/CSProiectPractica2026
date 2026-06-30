import numpy as np
from src.io.frame_preprocessor import FrameGeometry, preprocess
from dataclasses import dataclass

def test_preprocess_nan_filtering():
    # Mocam funcția internă doar pentru a verifica conversia NaN a preprocesorului
    class MockGeometry:
        y_slice = slice(0, 5)
        x_slice = slice(0, 5)
        lon_grid = np.zeros((5, 5))
        lat_grid = np.zeros((5, 5))
        roi_mask = np.ones((5, 5), dtype=bool)

    # Dacă _read_rain_window returnează NaN-uri, preprocess ar trebui să le convertească în 0.0
    from src.io import frame_preprocessor
    original_read = frame_preprocessor._read_rain_window
    try:
        def mock_read(file_path, y, x):
            rr = np.ones((5, 5))
            rr[2, 2] = np.nan
            rr[0, 0] = -5.0
            return rr
            
        frame_preprocessor._read_rain_window = mock_read
        
        prep = preprocess("dummy.nc", MockGeometry(), (0, 0, 0, 0))
        assert prep is not None
        assert np.isnan(prep.rain_rate[2, 2]), "NaN values should be preserved as missing data"
        assert prep.rain_rate[0, 0] == 0.0, "Negative values should be zeroed"
        
    finally:
        frame_preprocessor._read_rain_window = original_read
