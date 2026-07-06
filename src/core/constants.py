"""Shared core constants."""

HORIZON_STEPS = {"15m": 2, "1h": 5, "2h": 9}
HORIZON_NAMES = tuple(HORIZON_STEPS.keys())
DEFAULT_HORIZONS = tuple((step, name) for name, step in HORIZON_STEPS.items())
