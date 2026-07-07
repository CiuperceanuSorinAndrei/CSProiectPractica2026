"""Stage-storage curve (level -> volume) for a reservoir.

Submerged volume (up to NNR) is anchored from an attribute (`vol_mil_m3`); the DEM cannot see underwater.
Above the waterline, we integrate the real terrain from the DEM to obtain the capacity in the
flood attenuation band (NNR -> crest) - exactly the volume the shapefile leaves 0 (`vol_atenua`).

Curve levels are *relative* to the waterline: `dh=0` means NNR (volume = v_nnr).
When DEM is missing (lake under one pixel, tile missing), it falls back to a parametric prismatic model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StageStorageCurve:
    levels_m: np.ndarray      # heights relative to NNR (0, step, 2*step, ...)
    volumes_m3: np.ndarray    # total volume at each level (monotonic increasing, volumes[0]=v_nnr)
    v_nnr_m3: float           # volume at NNR (anchor point)
    waterline_m: float        # absolute waterline elevation, from DEM or attribute
    source: str               # "dem" | "parametric"

    # ---- runtime queries -------------------------------------------------
    def volume_at_level(self, dh: float) -> float:
        """Total volume (m^3) at `dh` meters above NNR (interpolated, clamped at edges)."""
        return float(np.interp(dh, self.levels_m, self.volumes_m3))

    def level_for_added_volume(self, added_m3: float) -> float:
        """Level rise (m above NNR) produced by an `added_m3` volume added above NNR."""
        if added_m3 <= 0.0:
            return 0.0
        rel = self.volumes_m3 - self.v_nnr_m3            # volume above NNR, increasing from 0
        return float(np.interp(added_m3, rel, self.levels_m))

    def level_for_volume(self, volume_m3: float) -> float:
        """Elevation (m relative to NNR; negative = below NNR) for a given total volume."""
        return float(np.interp(volume_m3, self.volumes_m3, self.levels_m))

    def volume_for_wse(self, wse_m: float) -> float:
        """Total volume (m^3) for an absolute waterline elevation (m, same geoid as DEM)."""
        return self.volume_at_level(wse_m - self.waterline_m)

    def with_submerged_branch(self, surface_area_m2: float, step_m: float = 0.5) -> "StageStorageCurve":
        """Extends the curve below NNR (submerged part, invisible from DEM) using a conic model:
        area decreases linearly from `surface_area_m2` (at NNR) to 0 at the bottom, so V grows quadratically.
        Required to start simulation from a current level below NNR. Idempotent."""
        if self.levels_m[0] < 0 or surface_area_m2 <= 0 or self.v_nnr_m3 <= 0:
            return self
        depth = 2.0 * self.v_nnr_m3 / surface_area_m2     # depth NNR->bottom (cone: V=A*D/2)
        n = max(int(depth / step_m), 1)
        below_dh = np.linspace(-depth, 0.0, n + 1)[:-1]   # exclude 0 (already in above-NNR part)
        below_v = self.v_nnr_m3 * ((depth + below_dh) / depth) ** 2
        return StageStorageCurve(
            levels_m=np.concatenate([below_dh, self.levels_m]),
            volumes_m3=np.concatenate([below_v, self.volumes_m3]),
            v_nnr_m3=self.v_nnr_m3, waterline_m=self.waterline_m, source=self.source,
        )

    @property
    def capacity_to_crest_m3(self) -> float:
        """Volume from NNR to the highest modeled level (approximate crest)."""
        return float(self.volumes_m3[-1] - self.v_nnr_m3)

    def overtops(self, added_m3: float) -> bool:
        return added_m3 > self.capacity_to_crest_m3

    # ---- serialization (for JSON cache) -----------------------------------
    def to_dict(self) -> dict:
        return {
            "levels_m": [round(float(x), 3) for x in self.levels_m],
            "volumes_m3": [round(float(x), 1) for x in self.volumes_m3],
            "v_nnr_m3": self.v_nnr_m3,
            "waterline_m": self.waterline_m,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StageStorageCurve":
        return cls(
            levels_m=np.asarray(d["levels_m"], dtype=float),
            volumes_m3=np.asarray(d["volumes_m3"], dtype=float),
            v_nnr_m3=float(d["v_nnr_m3"]),
            waterline_m=float(d["waterline_m"]),
            source=str(d["source"]),
        )

    # ---- offline construction ------------------------------------------------
    @classmethod
    def from_dem(cls, window, polygon, v_nnr_m3: float,
                 max_rise_m: float = 25.0, step_m: float = 0.5) -> "StageStorageCurve | None":
        """Integrates DEM terrain above the waterline to obtain V(dh). Returns None if the lake has
        too few pixels (under ~3), in which case the caller falls back to the parametric model."""
        from scipy import ndimage

        dem = window.dem
        water = window.water_mask(polygon)
        if int(water.sum()) < 3:
            return None

        cell_area = window.cell_area_m2()
        finite = np.isfinite(dem)
        h0 = float(np.nanmedian(dem[water]))

        levels = np.arange(0.0, max_rise_m + step_m, step_m)
        volumes = np.empty_like(levels)
        base = np.maximum(dem, h0)
        for k, dh in enumerate(levels):
            h = h0 + dh
            cand = finite & (dem <= h)
            lab, _ = ndimage.label(cand)
            keep = set(np.unique(lab[water])) - {0}
            flooded = np.isin(lab, list(keep)) if keep else np.zeros_like(cand)
            depth = np.clip(h - base, 0.0, None)
            add = float((depth[flooded] * cell_area[flooded]).sum())
            volumes[k] = v_nnr_m3 + add

        volumes = np.maximum.accumulate(volumes)  # ensure monotonicity (numerical safety)
        return cls(levels_m=levels, volumes_m3=volumes, v_nnr_m3=v_nnr_m3,
                   waterline_m=h0, source="dem")

    @classmethod
    def from_attributes(cls, v_nnr_m3: float, surface_area_m2: float, waterline_m: float,
                        max_rise_m: float = 10.0, step_m: float = 0.5) -> "StageStorageCurve":
        """Parametric prismatic fallback: above NNR area remains ~constant (V grows linearly).
        Used when DEM is unavailable (tiny lakes, missing tiles)."""
        levels = np.arange(0.0, max_rise_m + step_m, step_m)
        area = max(surface_area_m2, 1.0)
        volumes = v_nnr_m3 + area * levels
        return cls(levels_m=levels, volumes_m3=volumes, v_nnr_m3=v_nnr_m3,
                   waterline_m=waterline_m, source="parametric")
