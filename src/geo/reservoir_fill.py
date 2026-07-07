"""Estimarea umplerii unui lac de acumulare din precipitatia acumulata.

Model:
  1. Debit de intrare = scurgere din bazinul hidrografic (delimitat din DEM):
         inflow = C_runoff * (MAP_mm / 1000) * suprafata_bazin
     (fara bazin se cade pe ploaia directa pe luciul apei).
  2. Volumul de pornire = nivelul curent real al lacului (SWOT WSE -> volum pe curba
     stage-storage), sau NNR daca lacul nu are observatie SWOT.
  3. Volumul de intrare se adauga peste cel de pornire si se citeste pe curba stage-storage
     noul nivel si gradul de umplere.

Suprafata bazinului, curba si nivelul curent vin din `ReservoirLoader` (precalculate de
scripts/build_reservoir_dem.py si scripts/build_reservoir_levels.py).
"""
from __future__ import annotations

from dataclasses import dataclass

MM_TO_M = 1.0e-3  # 1 mm precipitatie = 1 L/m^2 = 0.001 m adancime


@dataclass
class ReservoirFillResult:
    inflow_m3: float
    inflow_source: str            # "catchment" | "direct_rain"
    catchment_km2: float | None
    outflow_m3: float             # evacuare la baraj pe durata evenimentului
    evap_m3: float                # evaporare de suprafata pe durata evenimentului
    start_volume_m3: float
    start_fill_pct: float         # volum_start / volum_NNR * 100
    new_volume_m3: float
    new_fill_pct: float           # (start + intrare - iesiri) / volum_NNR * 100
    contribution_pct: float       # inflow / volum_NNR * 100 (marimea evenimentului)
    level_before_m: float | None  # cota fata de NNR (negativ = sub NNR)
    level_after_m: float | None
    delta_level_m: float | None
    overtops: bool                # depaseste coronamentul
    level_source: str             # "swot" | "assumed_nnr"
    level_as_of: str | None
    curve_source: str | None      # "dem" | "parametric"


class ReservoirFillEstimator:
    """Adauga scurgerea de bazin peste volumul curent si o citeste pe curba stage-storage."""

    @staticmethod
    def estimate(map_mm: float, reservoir: dict | None, runoff_coeff: float,
                 duration_hours: float | None = None, evap_mm_day: float = 0.0,
                 outflow_m3s: float = 0.0) -> ReservoirFillResult | None:
        """Bilant hidrologic al lacului: V_{t+1} = V_t + intrare - evacuare - evaporare, apoi citit
        pe curba stage-storage. None cand nu e selectat un lac / lipseste capacitatea (mod oras/cerc).

        Evacuarea si evaporarea se aplica doar cand se cunoaste `duration_hours` (durata evenimentului);
        cu rate 0 (implicit) bilantul se reduce la V_t + intrare.
        """
        if not reservoir:
            return None
        v_nnr = reservoir.get("max_volume_m3") or 0.0
        if v_nnr <= 0.0:
            return None

        # volum de pornire: nivelul curent real (SWOT/Sentinel-2), altfel NNR
        start_v = reservoir.get("current_volume_m3")
        level_source = reservoir.get("level_source", "assumed_nnr")
        if start_v is None:
            start_v = v_nnr

        # intrare: debit de bazin (sau ploaie directa pe lac)
        depth_m = max(float(map_mm or 0.0), 0.0) * MM_TO_M
        catch_km2 = reservoir.get("catchment_km2")
        if catch_km2 and catch_km2 > 0.0:
            inflow = runoff_coeff * depth_m * (catch_km2 * 1e6)
            inflow_source = "catchment"
        else:
            inflow = depth_m * (reservoir.get("surface_area_m2") or 0.0)
            inflow_source = "direct_rain"

        # iesiri: evacuare la baraj + evaporare de suprafata, pe durata evenimentului
        outflow_m3 = evap_m3 = 0.0
        if duration_hours and duration_hours > 0.0:
            outflow_m3 = max(outflow_m3s, 0.0) * duration_hours * 3600.0
            area = reservoir.get("surface_area_m2") or 0.0
            evap_m3 = max(evap_mm_day, 0.0) * MM_TO_M * area * (duration_hours / 24.0)

        new_v = max(start_v + inflow - outflow_m3 - evap_m3, 0.0)

        curve = reservoir.get("stage_storage")
        level_before = level_after = delta_level = curve_source = None
        overtops = False
        if curve is not None:
            level_before = curve.level_for_volume(start_v)
            level_after = curve.level_for_volume(new_v)
            delta_level = level_after - level_before
            overtops = new_v > v_nnr + curve.capacity_to_crest_m3
            curve_source = curve.source

        return ReservoirFillResult(
            inflow_m3=inflow, inflow_source=inflow_source, catchment_km2=catch_km2,
            outflow_m3=outflow_m3, evap_m3=evap_m3,
            start_volume_m3=start_v, start_fill_pct=100.0 * start_v / v_nnr,
            new_volume_m3=new_v, new_fill_pct=100.0 * new_v / v_nnr,
            contribution_pct=100.0 * inflow / v_nnr,
            level_before_m=level_before, level_after_m=level_after, delta_level_m=delta_level,
            overtops=overtops, level_source=level_source,
            level_as_of=reservoir.get("level_as_of"), curve_source=curve_source,
        )
