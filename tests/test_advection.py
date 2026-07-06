import numpy as np
from src.core.nowcast.advection_engine import AdvectionEngine
from src.core.nowcast.kinematic_advector import KinematicAdvector
from src.core.constants import HORIZON_STEPS
from src.core.domain import StormCell

def _create_engine():
    return AdvectionEngine(KinematicAdvector())

def test_advection_extrapolate_nan_handling():
    # Check if NaN in rain_rate is handled before processing
    rain_rate = np.zeros((10, 10))
    rain_rate[5, 5] = np.nan
    
    engine = _create_engine()
    float_preds = engine.extrapolate(
        rain_rate, tracked_cells=[], horizons=[(1, "15m")]
    )
    
    # float_preds[1] shouldn't contain any NaNs
    assert not np.isnan(float_preds[1]).any()
    
def test_tracked_cell_advection_shape_and_nan_bounds():
    cell = StormCell(
        cell_id="c1", is_tracked=True,
        centroid_y=10.0, centroid_x=10.0,
        predicted_centroid_y=10.0, predicted_centroid_x=10.0,
        v_x=2.0, v_y=1.0, mean_intensity=1.0,
    )
    cell.E = 1.0
    cell.initialize_simulation_state()
    rain_rate = np.zeros((20, 20), dtype=np.float32)
    rain_rate[10, 10] = 5.0

    engine = _create_engine()
    float_preds = engine.extrapolate(
        rain_rate, tracked_cells=[cell], horizons=[(1, "15m")]
    )

    assert float_preds[1].shape == rain_rate.shape
    assert not np.isnan(float_preds[1]).any()


def test_step_specific_velocity_changes_as_cell_approaches_roi():
    engine = _create_engine()
    cells = [
        StormCell(is_tracked=True, centroid_y=50, centroid_x=49, v_y=0, v_x=1, volume=1),
        StormCell(is_tracked=True, centroid_y=50, centroid_x=0, v_y=0, v_x=10, volume=2),
    ]

    vx1, _, _, _ = engine._velocity_for_step(cells, 1, (50.0, 50.0), 5.0)
    vx5, _, _, _ = engine._velocity_for_step(cells, 5, (50.0, 50.0), 5.0)

    assert vx1 == 1.0
    assert vx5 == 10.0


def test_incoming_cells_dominate_moving_away_cells():
    engine = _create_engine()
    cells = [
        StormCell(is_tracked=True, centroid_y=50, centroid_x=45, v_y=0, v_x=2, volume=1),
        StormCell(is_tracked=True, centroid_y=50, centroid_x=55, v_y=0, v_x=8, volume=5),
    ]

    vx, _, _, weights = engine._velocity_for_step(cells, 1, (50.0, 50.0), 5.0)

    assert weights[0] > weights[1]
    assert vx == 2.0


def test_far_high_mass_cell_does_not_dominate_when_predicted_far_from_roi():
    engine = _create_engine()
    cells = [
        StormCell(is_tracked=True, centroid_y=10, centroid_x=10, v_y=0, v_x=1, volume=1),
        StormCell(is_tracked=True, centroid_y=200, centroid_x=200, v_y=0, v_x=20, volume=1000),
    ]

    vx, _, _, _ = engine._velocity_for_step(cells, 1, (10.0, 10.0), 2.0)

    assert vx == 1.0


def test_reset_feedback_restores_neutral_bias_state():
    engine = _create_engine()
    for _ in range(max(HORIZON_STEPS.values()) + 1):
        engine.update_feedback(1.0, {"15m": 2.0, "1h": 2.0, "2h": 2.0})

    assert any(abs(bias - 1.0) > 1e-6 for bias in engine._bias_by_step.values())

    engine.reset_feedback()

    assert engine._error_history == []
    assert engine.dynamic_bias_correction == 1.0
    assert engine._pid_bias == 1.0
    assert all(bias == 1.0 for bias in engine._bias_by_step.values())
    assert all(len(window) == 0 for window in engine._ratio_windows.values())


def test_feedback_uses_only_matured_forecasts():
    engine = _create_engine()

    engine.update_feedback(0.1, {"15m": 10.0, "1h": 10.0, "2h": 10.0})
    engine.update_feedback(0.0, {"15m": 0.1, "1h": 0.1, "2h": 0.1})

    assert engine._bias_by_step[HORIZON_STEPS["15m"]] == 1.0

    engine.update_feedback(5.0, {})

    assert engine._bias_by_step[HORIZON_STEPS["15m"]] < 1.0
    assert engine._bias_by_step[HORIZON_STEPS["1h"]] == 1.0
    assert engine._bias_by_step[HORIZON_STEPS["2h"]] == 1.0


def test_feedback_compares_1h_forecast_to_matching_cumulative_window():
    engine = _create_engine()

    engine.update_feedback(0.0, {"1h": 10.0})
    for actual in [1.0, 2.0, 3.0, 4.0]:
        engine.update_feedback(actual, {})

    assert engine._bias_by_step[HORIZON_STEPS["1h"]] == 1.0

    engine.update_feedback(5.0, {})

    window = engine._ratio_windows[HORIZON_STEPS["1h"]]
    assert np.isclose(window[-1], np.log(1.5))
    assert engine._bias_by_step[HORIZON_STEPS["1h"]] > 1.0


def test_wet_after_dry_miss_increases_matching_horizon_bias():
    engine = _create_engine()

    engine.update_feedback(0.0, {"15m": 0.0, "1h": 0.0, "2h": 0.0})
    engine.update_feedback(0.0, {"15m": 0.0, "1h": 0.0, "2h": 0.0})
    engine.update_feedback(1.0, {})

    assert engine._bias_by_step[HORIZON_STEPS["15m"]] > 1.0


def test_feedback_bias_corrects_cumulative_volumes_not_advected_maps():
    rain_rate = np.ones((4, 4), dtype=np.float32)
    neutral = _create_engine()
    biased = _create_engine()
    biased._bias_by_step[HORIZON_STEPS["15m"]] = 1.5

    neutral_preds = neutral.extrapolate(rain_rate, [], [(HORIZON_STEPS["15m"], "15m")])
    biased_preds = biased.extrapolate(rain_rate, [], [(HORIZON_STEPS["15m"], "15m")])

    assert np.allclose(biased_preds[HORIZON_STEPS["15m"]], neutral_preds[HORIZON_STEPS["15m"]])
    assert biased.correct_cumulative_volumes({"15m": 2.0})["15m"] == 3.0


def test_unreliable_centroids_blend_later_steps_toward_persistence():
    rain_rate = np.zeros((20, 20), dtype=np.float32)
    rain_rate[10, 10] = 5.0
    reliable = StormCell(is_tracked=True, centroid_y=10, centroid_x=10, v_y=0, v_x=3, volume=1)
    unreliable = StormCell(
        is_tracked=True, centroid_y=10, centroid_x=10, v_y=0, v_x=3, volume=1,
        prediction_error_pixels=999.0,
    )
    engine = _create_engine()

    reliable_pred = engine.extrapolate(rain_rate, [reliable], [(2, "15m")])[2]
    unreliable_pred = engine.extrapolate(rain_rate, [unreliable], [(2, "15m")])[2]

    assert np.sum(np.abs(unreliable_pred - rain_rate)) < np.sum(np.abs(reliable_pred - rain_rate))


def test_orchestrator_reset_tracking_resets_feedback():
    from orchestrator import Orchestrator

    orch = Orchestrator()
    engine = orch._advection_engine
    for _ in range(max(HORIZON_STEPS.values()) + 1):
        engine.update_feedback(1.0, {"15m": 2.0, "1h": 2.0, "2h": 2.0})

    orch.reset_tracking()

    assert engine._error_history == []
    assert all(bias == 1.0 for bias in engine._bias_by_step.values())
