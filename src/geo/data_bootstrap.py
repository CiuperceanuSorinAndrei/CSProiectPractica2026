"""Verifica datele volumetrice necesare si genereaza ce lipseste la pornirea aplicatiei.

La start, aplicatia are nevoie de:
  - curbele DEM + bazinele hidrografice (dem_augment.json)  -- fara credentiale;
  - nivelele curente SWOT (reservoir_levels.json)           -- necesita EDL_USER / EDL_PASS;
  - nivelele curente Sentinel-2 (reservoir_levels_s2.json)  -- necesita SH_ID / SH_SECRET.

Fisierele lipsa sunt generate ruland scriptul de build corespunzator. Credentialele se citesc
din mediu (incarcate din .env). Pasii fara credentiale se sar cu un avertisment, iar aplicatia
porneste cu ce date exista (celelalte lacuri raman in afara scopului pana la generare).
"""
from __future__ import annotations

import os
import sys
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from src import config

# (eticheta, fisier rezultat, script de build, credentiale necesare -> (nume, valoare))
_STEPS = [
    ("Curbe DEM + bazine", "data/geo/reservoirs/dem_augment.json",
     "scripts/build_reservoir_dem.py", ()),
    ("Nivele SWOT", "data/geo/reservoirs/reservoir_levels.json",
     "scripts/build_reservoir_levels.py", (("EDL_USER", config.EDL_USER), ("EDL_PASS", config.EDL_PASS))),
    ("Nivele Sentinel-2", "data/geo/reservoirs/reservoir_levels_s2.json",
     "scripts/build_reservoir_levels_s2.py", (("SH_ID", config.SH_ID), ("SH_SECRET", config.SH_SECRET))),
]


def _present(rel_path: str) -> bool:
    p = os.path.join(_ROOT, rel_path)
    return os.path.exists(p) and os.path.getsize(p) > 2


def ensure_reservoir_data() -> None:
    """Genereaza fisierele de date lipsa. Non-blocant pe erori: aplicatia porneste oricum."""
    for label, out_file, script, creds in _STEPS:
        if _present(out_file):
            continue
        missing = [name for name, val in creds if not val]
        if missing:
            print(f"[date] {label}: lipseste {out_file}. Setati {', '.join(missing)} in .env "
                  f"pentru a-l descarca. (sarit)")
            continue
        print(f"[date] {label}: lipseste {out_file} -> generez cu {script} (poate dura)...", flush=True)
        try:
            subprocess.run([sys.executable, os.path.join(_ROOT, script)], cwd=_ROOT, check=False)
        except Exception as e:
            print(f"[date] {label}: generare esuata ({e}). Continui fara.", flush=True)
