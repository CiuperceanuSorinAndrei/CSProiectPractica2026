"""Shared core constants."""

# H-SAF frames arrive about 15 minutes late. User-facing forecast labels are
# relative to "now", so each horizon includes one extra 15-minute source step.
HORIZON_STEPS = {"15m": 2, "1h": 5, "2h": 9}
HORIZON_NAMES = tuple(HORIZON_STEPS.keys())
DEFAULT_HORIZONS = tuple((step, name) for name, step in HORIZON_STEPS.items())
