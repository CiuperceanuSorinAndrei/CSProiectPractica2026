from src.config import RAIN_THRESHOLD_MIN, RAIN_VMAX
from src.ui_helpers.base_map import resolve_scale


def test_rain_scale_defaults_match_visible_config_thresholds():
    assert resolve_scale(None, None) == (RAIN_THRESHOLD_MIN, RAIN_VMAX)


def test_rain_scale_allows_explicit_overrides():
    assert resolve_scale(2.0, 8.0) == (2.0, 8.0)
