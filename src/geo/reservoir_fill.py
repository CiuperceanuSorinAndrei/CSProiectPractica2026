# Hydrological model estimating reservoir fill from accumulated precipitation
from __future__ import annotations

from dataclasses import dataclass

MM_TO_M = 1.0e-3  # 1 mm precipitation = 1 L/m^2 = 0.001 m depth


@dataclass
class ReservoirFillResult:
    inflow_m3: float
    inflow_source: str            # "catchment" | "direct_rain"
    catchment_km2: float | None
    outflow_m3: float             # dam release during event
    evap_m3: float                # surface evaporation during event
    start_volume_m3: float
    start_fill_pct: float         # start_volume / NNR_volume * 100
    new_volume_m3: float
    new_fill_pct: float           # (start + inflow - outflow) / NNR_volume * 100
    contribution_pct: float       # inflow / NNR_volume * 100 (event magnitude)
    level_before_m: float | None  # elevation relative to NNR (negative = below NNR)
    level_after_m: float | None
    delta_level_m: float | None
    overtops: bool                # exceeds crest
    level_source: str             # "swot" | "assumed_nnr"
    level_as_of: str | None
    curve_source: str | None      # "dem" | "parametric"


class ReservoirFillEstimator:
    # Estimates reservoir fill from accumulated precipitation

    @staticmethod
    def estimate(map_mm: float, reservoir: dict | None, runoff_coeff: float,
                 duration_hours: float | None = None, frame_time=None,
                 evap_mm_day: float | None = None, outflow_m3s: float | None = None) -> ReservoirFillResult | None:
        # Hydrological balance: V_{t+1} = V_t + inflow - outflow - evaporation
        if not reservoir:
            return None
        v_nnr = reservoir.get("max_volume_m3") or 0.0
        if v_nnr <= 0.0:
            return None

        # Start volume: current real level, else NNR
        start_v = reservoir.get("current_volume_m3")
        level_source = reservoir.get("level_source", "assumed_nnr")
        if start_v is None:
            start_v = v_nnr

        # Inflow: catchment runoff or direct rain on lake
        depth_m = max(float(map_mm or 0.0), 0.0) * MM_TO_M
        catch_km2 = reservoir.get("catchment_km2")
        if catch_km2 and catch_km2 > 0.0:
            inflow = runoff_coeff * depth_m * (catch_km2 * 1e6)
            inflow_source = "catchment"
        else:
            inflow = depth_m * (reservoir.get("surface_area_m2") or 0.0)
            inflow_source = "direct_rain"

        # Outflow: dam release + surface evaporation during event
        outflow_m3 = evap_m3 = 0.0
        if duration_hours and duration_hours > 0.0:
            # Dynamic base outflow: 5 L/s/km2 if catchment known, else 10 m3/s default.
            if outflow_m3s is None:
                outflow_m3s = (catch_km2 * 0.005) if catch_km2 else 10.0
            outflow_m3 = max(outflow_m3s, 0.0) * duration_hours * 3600.0
            
            if evap_mm_day is None:
                # Dynamic evaporation based on month for Romania (mm/day)
                evap_monthly_mm_day = {
                    1: 0.5, 2: 0.8, 3: 1.5, 4: 2.5, 5: 3.5, 6: 4.5,
                    7: 5.0, 8: 4.8, 9: 3.0, 10: 1.8, 11: 0.8, 12: 0.5
                }
                month = frame_time.month if frame_time else 6
                evap_mm_day = evap_monthly_mm_day.get(month, 2.5)
            
            area = reservoir.get("surface_area_m2") or 0.0
            evap_m3 = max(evap_mm_day, 0.0) * MM_TO_M * area * (duration_hours / 24.0)

        new_v = max(start_v + inflow - outflow_m3 - evap_m3, 0.0)

        # Calculate levels and overtops using stage-storage curve
        curve = reservoir.get("stage_storage")
        level_before = level_after = delta_level = curve_source = None
        overtops = False
        if curve is not None:
            level_before = curve.level_for_volume(start_v)
            level_after = curve.level_for_volume(new_v)
            delta_level = level_after - level_before
            overtops = new_v > v_nnr + curve.capacity_to_crest_m3
            curve_source = curve.source

        # Return calculation results
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
