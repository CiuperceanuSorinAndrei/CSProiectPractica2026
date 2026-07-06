import numpy as np
import pytest

from src.core.tracking.storm_filter import StormFilter
from src.core.metrics.evaluator import Evaluator

def test_storm_filter_velocity_update_is_finite():
    sf = StormFilter(10.0, 10.0)
    sf.predict()
    sf.update(12.0, 13.0)

    assert np.isfinite([sf.x, sf.y, sf.v_x, sf.v_y]).all()
    
def test_storm_filter_joseph_form_symmetry():
    sf = StormFilter(10.0, 10.0)
    # Modify P to be slightly asymmetric to simulate float errors
    sf._kf.P[0, 1] += 1e-10
    sf.update(12.0, 12.0)
    
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
    
    # 1.0 mm/h rain rate over 15 mins (0.25h) should result in 0.25 mm MAP.
    roi_vol, _, _ = Evaluator.calculate_volumes(
        rain_rate, {}, roi_mask, pixel_area_km2, horizons=[]
    )
    
    assert np.isclose(roi_vol, 0.25)
