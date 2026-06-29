import numpy as np
import pytest

from src.core.storm_filter import StormFilter
from src.core.evaluator import Evaluator

def test_storm_filter_log_space_area():
    # Test area is properly exponentiated
    sf = StormFilter(10.0, 10.0, initial_area=50.0)
    assert np.isclose(sf.area, 50.0)
    
    # Test update prevents log(0) or log(negative)
    sf.update(10.0, 10.0, -10.0)
    # the internal max(1.0, observed_area) will feed log(1.0)=0 to the filter.
    # Due to Kalman smoothing, the state won't instantly jump to 1.0, but it will drop significantly from 50.
    assert sf.area < 25.0
    assert sf.area > 0.0
    
def test_storm_filter_joseph_form_symmetry():
    sf = StormFilter(10.0, 10.0)
    # Modify P to be slightly asymmetric to simulate float errors
    sf._kf.P[0, 1] += 1e-10
    sf.update(12.0, 12.0, 15.0)
    
    # Check if P is perfectly symmetric
    P = sf._kf.P
    assert np.allclose(P, P.T)
    # Check positive definite
    try:
        np.linalg.cholesky(P)  # Should not raise LinAlgError
    except np.linalg.LinAlgError:
        pytest.fail("Matrix P is not positive definite!")

def test_evaluator_area_consistency():
    # Ensure calculate_volumes multiplies correctly (no inverted division)
    rain_rate = np.ones((5, 5))
    roi_mask = np.ones((5, 5), dtype=bool)
    pixel_area_km2 = np.ones((5, 5)) * 10.0 # 10 km2 per pixel
    
    # 25 pixels * 10 km2 = 250 km2
    # conversion_factor = 250.0
    # expected roi volume = 250 * 250 = 62500
    
    roi_vol, _, _ = Evaluator.calculate_volumes(
        rain_rate, {}, roi_mask, pixel_area_km2, horizons=[]
    )
    
    assert np.isclose(roi_vol, 62500.0)
