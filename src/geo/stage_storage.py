"""Curba stage-storage (nivel -> volum) a unui lac de acumulare.

Volumul submers (pana la NNR) este ancorat din atribut (`vol_mil_m3`); DEM-ul nu vede sub apa.
Peste luciul apei integram terenul real din DEM pentru a obtine capacitatea din banda de
atenuare (NNR -> coronament) - exact volumul pe care shapefile-ul il lasa 0 (`vol_atenua`).

Nivelele din curba sunt *relative* la luciul apei: `dh=0` inseamna NNR (volum = v_nnr).
Cand DEM-ul lipseste (lac sub un pixel, tile absent), se cade pe un model prismatic parametric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StageStorageCurve:
    levels_m: np.ndarray      # inaltimi relative la NNR (0, step, 2*step, ...)
    volumes_m3: np.ndarray    # volum total la fiecare nivel (crescator, volumes[0]=v_nnr)
    v_nnr_m3: float           # volum la NNR (ancora)
    waterline_m: float        # cota luciului apei (elevatie absoluta), din DEM sau atribut
    source: str               # "dem" | "parametric"

    # ---- interogari runtime -------------------------------------------------
    def volume_at_level(self, dh: float) -> float:
        """Volum total (m^3) la `dh` metri peste NNR (interpolat, plafonat la capete)."""
        return float(np.interp(dh, self.levels_m, self.volumes_m3))

    def level_for_added_volume(self, added_m3: float) -> float:
        """Cresterea de nivel (m peste NNR) produsa de un volum `added_m3` adaugat peste NNR."""
        if added_m3 <= 0.0:
            return 0.0
        rel = self.volumes_m3 - self.v_nnr_m3            # volum peste NNR, crescator de la 0
        return float(np.interp(added_m3, rel, self.levels_m))

    def level_for_volume(self, volume_m3: float) -> float:
        """Cota (m fata de NNR; negativ = sub NNR) pentru un volum total dat."""
        return float(np.interp(volume_m3, self.volumes_m3, self.levels_m))

    def volume_for_wse(self, wse_m: float) -> float:
        """Volum total (m^3) pentru o cota absoluta a luciului apei (m, acelasi geoid ca DEM-ul)."""
        return self.volume_at_level(wse_m - self.waterline_m)

    def with_submerged_branch(self, surface_area_m2: float, step_m: float = 0.5) -> "StageStorageCurve":
        """Extinde curba sub NNR (partea submersa, invizibila din DEM) cu un model conic:
        aria scade liniar de la `surface_area_m2` (la NNR) la 0 pe fund, deci V creste patratic.
        Necesara pentru a porni simularea de la un nivel curent sub NNR. Idempotenta."""
        if self.levels_m[0] < 0 or surface_area_m2 <= 0 or self.v_nnr_m3 <= 0:
            return self
        depth = 2.0 * self.v_nnr_m3 / surface_area_m2     # adancime NNR->fund (con: V=A*D/2)
        n = max(int(depth / step_m), 1)
        below_dh = np.linspace(-depth, 0.0, n + 1)[:-1]   # fara 0 (deja in partea de peste NNR)
        below_v = self.v_nnr_m3 * ((depth + below_dh) / depth) ** 2
        return StageStorageCurve(
            levels_m=np.concatenate([below_dh, self.levels_m]),
            volumes_m3=np.concatenate([below_v, self.volumes_m3]),
            v_nnr_m3=self.v_nnr_m3, waterline_m=self.waterline_m, source=self.source,
        )

    @property
    def capacity_to_crest_m3(self) -> float:
        """Volum de la NNR pana la ultimul nivel modelat (coronament aproximativ)."""
        return float(self.volumes_m3[-1] - self.v_nnr_m3)

    def overtops(self, added_m3: float) -> bool:
        return added_m3 > self.capacity_to_crest_m3

    # ---- serializare (pentru cache JSON) -----------------------------------
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

    # ---- constructie offline ------------------------------------------------
    @classmethod
    def from_dem(cls, window, polygon, v_nnr_m3: float,
                 max_rise_m: float = 25.0, step_m: float = 0.5) -> "StageStorageCurve | None":
        """Integreaza terenul DEM peste luciul apei ca sa obtina V(dh). None daca lacul are
        prea putini pixeli (sub ~3), caz in care apelantul cade pe modelul parametric."""
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

        volumes = np.maximum.accumulate(volumes)  # asigura monotonia (siguranta numerica)
        return cls(levels_m=levels, volumes_m3=volumes, v_nnr_m3=v_nnr_m3,
                   waterline_m=h0, source="dem")

    @classmethod
    def from_attributes(cls, v_nnr_m3: float, surface_area_m2: float, waterline_m: float,
                        max_rise_m: float = 10.0, step_m: float = 0.5) -> "StageStorageCurve":
        """Rezerva parametrica prismatica: peste NNR aria ramane ~constanta (V creste liniar).
        Folosita cand DEM-ul nu e disponibil (lacuri minuscule, tile lipsa)."""
        levels = np.arange(0.0, max_rise_m + step_m, step_m)
        area = max(surface_area_m2, 1.0)
        volumes = v_nnr_m3 + area * levels
        return cls(levels_m=levels, volumes_m3=volumes, v_nnr_m3=v_nnr_m3,
                   waterline_m=waterline_m, source="parametric")
