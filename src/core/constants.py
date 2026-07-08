"""Shared core constants."""

# Horizons include a 15-minute padding to account for H-SAF arrival latency.
HORIZON_STEPS = {"15m": 2, "1h": 5, "2h": 9}
HORIZON_NAMES = tuple(HORIZON_STEPS.keys())
DEFAULT_HORIZONS = tuple((step, name) for name, step in HORIZON_STEPS.items())
