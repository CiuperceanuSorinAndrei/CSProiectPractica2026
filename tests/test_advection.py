import numpy as np
from src.core.nowcast.advection_engine import AdvectionEngine
from src.core.domain import StormCell

def test_advection_extrapolate_nan_handling():
    # Check if NaN in rain_rate is handled before processing
    rain_rate = np.zeros((10, 10))
    rain_rate[5, 5] = np.nan
    flow = np.zeros((10, 10, 2))
    
    sparse_preds, float_preds = AdvectionEngine.extrapolate(
        rain_rate, flow, tracked_cells=[], horizons=[(1, "15m")]
    )
    
    # float_preds[1] shouldn't contain any NaNs
    assert not np.isnan(float_preds[1]).any()
    
def test_local_adaptive_weighting_bounds():
    # Test bounding box creation limits (no out-of-bounds indexing / NaN in growth mask).
    # Cell sits near the (0,0) corner so the 3-sigma halo box is clipped on both axes.
    cell = StormCell(
        cell_id="c1", is_tracked=True,
        centroid_y=10.0, centroid_x=10.0,
        predicted_centroid_y=10.0, predicted_centroid_x=10.0,
        predicted_area_kalman=50.0, E=1.0, mean_intensity=1.0,
    )
    mask = AdvectionEngine._create_spatial_growth_mask(
        (100, 100), [cell], {cell.cell_id: cell}
    )
    # Shape should remain valid and no out of bounds error
    assert mask.shape == (100, 100)
    assert not np.isnan(mask).any()
