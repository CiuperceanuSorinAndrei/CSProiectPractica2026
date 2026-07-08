# Verifies and generates missing volumetric data on application startup.
from __future__ import annotations

import os
import sys
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from src import config

# (label, output_file, build_script, required_credentials)
_STEPS = [
    ("DEM Curves + Basins", "data/geo/reservoirs/dem_augment.json",
     "scripts/build_reservoir_dem.py", ()),
    ("SWOT Levels", "data/geo/reservoirs/reservoir_levels.json",
     "scripts/build_reservoir_levels.py", (("EDL_USER", config.EDL_USER), ("EDL_PASS", config.EDL_PASS))),
    ("Sentinel-2 Levels", "data/geo/reservoirs/reservoir_levels_s2.json",
     "scripts/build_reservoir_levels_s2.py", (("SH_ID", config.SH_ID), ("SH_SECRET", config.SH_SECRET))),
]


def _present(rel_path: str) -> bool:
    p = os.path.join(_ROOT, rel_path)
    return os.path.exists(p) and os.path.getsize(p) > 2


def ensure_reservoir_data() -> None:
    # Generate missing data files without blocking execution on errors
    # Iterate through data generation steps
    for label, out_file, script, creds in _STEPS:
        if _present(out_file):
            continue
        # Check for required credentials
        missing = [name for name, val in creds if not val]
        if missing:
            print(f"[data] {label}: missing {out_file}. Set {', '.join(missing)} in .env "
                  f"to download. (skipped)")
            continue
        # Execute script and handle potential errors
        print(f"[data] {label}: missing {out_file} -> generating with {script} (may take a while)...", flush=True)
        try:
            subprocess.run([sys.executable, os.path.join(_ROOT, script)], cwd=_ROOT, check=False)
        except Exception as e:
            print(f"[data] {label}: generation failed ({e}). Continuing without it.", flush=True)
