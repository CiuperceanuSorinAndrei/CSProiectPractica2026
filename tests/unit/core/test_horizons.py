from datetime import UTC, datetime, timedelta
import numpy as np

from src.core.constants import DEFAULT_HORIZONS, HORIZON_NAMES, HORIZON_STEPS
from src.core.pipeline.frame_processor import FrameProcessor
from src.dashboard.frame_history import FrameHistory
from src.io.frame_preprocessor import FrameGeometry, FramePrep


class RecordingEngine:
    def __init__(self):
        self.horizons = None

    def extrapolate(self, rain_rate, tracked_cells, horizons, roi_mask=None):
        self.horizons = list(horizons)
        return {}

    def update_feedback(self, actual_map, preds):
        pass

    def correct_cumulative_volumes(self, volumes):
        return volumes

    def record_current_forecast(self, preds):
        pass


class EmptyTracker:
    def track(self, cells, rain_rate):
        return []


def test_horizon_constants_match_frame_processor_and_history():
    rain_rate = np.zeros((3, 3), dtype=np.float32)
    prep = FramePrep(rain_rate=rain_rate, filtered_cells=[], max_rain=0.0)
    geom = FrameGeometry(
        lon_grid=np.zeros((3, 3)),
        lat_grid=np.zeros((3, 3)),
        pixel_area_km2=np.ones((3, 3)),
        roi_mask=np.ones((3, 3), dtype=bool),
        y_slice=slice(0, 3),
        x_slice=slice(0, 3),
    )
    engine = RecordingEngine()

    FrameProcessor.process(prep, geom, EmptyTracker(), engine, run_mode="historic")

    hist = FrameHistory()
    assert HORIZON_STEPS == {"15m": 2, "1h": 5, "2h": 9}
    assert engine.horizons == list(DEFAULT_HORIZONS)
    assert tuple(hist.pred_volumes) == HORIZON_NAMES
    assert tuple(hist.pred_volumes_acc) == HORIZON_NAMES
    assert set(hist.reliability_counts[hist.thresholds[0]]) == set(HORIZON_STEPS)


def test_horizon_constants_include_15_minute_data_latency():
    assert HORIZON_STEPS["15m"] == 2
    assert HORIZON_STEPS["1h"] == 5
    assert HORIZON_STEPS["2h"] == 9


def test_live_horizons_compensate_for_frame_delay():
    rain_rate = np.zeros((3, 3), dtype=np.float32)
    prep = FramePrep(rain_rate=rain_rate, filtered_cells=[], max_rain=0.0)
    geom = FrameGeometry(
        lon_grid=np.zeros((3, 3)),
        lat_grid=np.zeros((3, 3)),
        pixel_area_km2=np.ones((3, 3)),
        roi_mask=np.ones((3, 3), dtype=bool),
        y_slice=slice(0, 3),
        x_slice=slice(0, 3),
    )
    engine = RecordingEngine()
    frame_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=15)

    FrameProcessor.process(prep, geom, EmptyTracker(), engine, frame_time=frame_time, run_mode="live")

    assert engine.horizons == [(2, "15m"), (5, "1h"), (9, "2h")]


def test_volume_sums_compare_matured_cumulative_forecasts():
    hist = FrameHistory()
    horizon = "15m"
    steps = HORIZON_STEPS[horizon]
    hist.true_volumes = [1.0, 2.0, 3.0, 4.0]
    hist.total_map_mm = sum(hist.true_volumes)
    hist.predicted_volume_accumulation[horizon] = 999.0
    hist.pred_volumes_acc[horizon] = [10.0, 20.0, 30.0, 40.0]

    actual, predicted = hist.volume_sums(horizon)

    assert steps == 2
    assert actual == (2.0 + 3.0) + (3.0 + 4.0)
    assert predicted == 10.0 + 20.0
