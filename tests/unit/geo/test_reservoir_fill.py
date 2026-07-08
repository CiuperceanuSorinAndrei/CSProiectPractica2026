# Stage-storage and reservoir filling tests.
# Integration tests verify precalculated data (dem_augment.json / reservoir_levels.json) if available.
import os
import json

import numpy as np
import pytest

from src.geo.stage_storage import StageStorageCurve
from src.geo.reservoir_fill import ReservoirFillEstimator

AUGMENT = "data/geo/reservoirs/dem_augment.json"
LEVELS = "data/geo/reservoirs/reservoir_levels.json"
LEVELS_S2 = "data/geo/reservoirs/reservoir_levels_s2.json"


# --- StageStorageCurve --------------------------------------------------------------------

def _prism(v_nnr=1_000_000.0, area=100_000.0):
    return StageStorageCurve.from_attributes(v_nnr, area, waterline_m=100.0, max_rise_m=10.0, step_m=0.5)


def test_prism_volume_at_level():
    c = _prism()
    assert c.volume_at_level(0.0) == pytest.approx(1_000_000.0)
    assert c.volume_at_level(5.0) == pytest.approx(1_500_000.0)


def test_level_for_volume_inverts_volume_at_level():
    c = _prism()
    v = c.volume_at_level(3.0)
    assert c.level_for_volume(v) == pytest.approx(3.0)


def test_capacity_to_crest_and_overtops():
    c = _prism()
    assert c.capacity_to_crest_m3 == pytest.approx(1_000_000.0)
    assert c.overtops(1_200_000.0)
    assert not c.overtops(900_000.0)


def test_curve_roundtrip_dict():
    c = _prism()
    c2 = StageStorageCurve.from_dict(c.to_dict())
    assert c2.volume_at_level(3.0) == pytest.approx(c.volume_at_level(3.0))


def test_submerged_branch_extends_below_nnr():
    # Cone depth NNR->bottom = 2*V/A = 2*1e6/1e5 = 20 m; V(bottom)=0, V(NNR)=1e6
    c = _prism().with_submerged_branch(surface_area_m2=100_000.0)
    assert c.levels_m[0] == pytest.approx(-20.0)
    assert c.volumes_m3[0] == pytest.approx(0.0, abs=1.0)
    assert c.volume_at_level(0.0) == pytest.approx(1_000_000.0)         # Unchanged NNR
    # At half depth volume is (0.5)^2 = 25% of NNR
    assert c.volume_at_level(-10.0) == pytest.approx(250_000.0, rel=0.05)


def test_volume_for_wse_uses_waterline():
    c = _prism().with_submerged_branch(100_000.0)   # waterline_m = 100
    assert c.volume_for_wse(100.0) == pytest.approx(1_000_000.0)        # At NNR
    assert c.volume_for_wse(90.0) < 1_000_000.0                          # 10m below NNR -> less volume


# --- ReservoirFillEstimator ---------------------------------------------------------------

def _reservoir(catchment_km2=100.0, v_nnr=10_000_000.0, area_m2=1_000_000.0, current_volume_m3=None,
               level_source="assumed_nnr"):
    curve = StageStorageCurve.from_attributes(v_nnr, area_m2, 100.0, max_rise_m=20.0).with_submerged_branch(area_m2)
    return {
        "max_volume_m3": v_nnr, "surface_area_m2": area_m2, "catchment_km2": catchment_km2,
        "stage_storage": curve, "current_volume_m3": current_volume_m3, "level_source": level_source,
    }


def test_estimate_catchment_runoff_from_nnr_default():
    # No current level -> start at NNR. 10 mm, C=0.35, catchment 100 km2 -> inflow 350000 m3
    res = ReservoirFillEstimator.estimate(10.0, _reservoir(), runoff_coeff=0.35)
    assert res.inflow_source == "catchment"
    assert res.inflow_m3 == pytest.approx(350_000.0)
    assert res.level_source == "assumed_nnr"
    assert res.start_fill_pct == pytest.approx(100.0)                    # NNR
    assert res.new_fill_pct == pytest.approx(103.5)
    assert res.contribution_pct == pytest.approx(3.5)
    assert res.delta_level_m == pytest.approx(0.35, abs=0.02)            # +350000 / 1e6 m2 prisma


def test_estimate_starts_from_swot_current_level():
    # Lake at 60% NNR (SWOT) -> start there, not from NNR
    r = _reservoir(current_volume_m3=6_000_000.0, level_source="swot")
    res = ReservoirFillEstimator.estimate(10.0, r, 0.35)
    assert res.level_source == "swot"
    assert res.start_fill_pct == pytest.approx(60.0)
    assert res.new_fill_pct == pytest.approx(63.5)
    assert res.level_before_m < 0.0                                     # Below NNR
    assert res.delta_level_m > 0.0


def test_estimate_direct_rain_fallback_without_catchment():
    r = _reservoir(catchment_km2=None)
    res = ReservoirFillEstimator.estimate(10.0, r, 0.35)
    assert res.inflow_source == "direct_rain"
    assert res.inflow_m3 == pytest.approx(10_000.0)                     # Depth*area, no C


def test_estimate_overtops_flag():
    res = ReservoirFillEstimator.estimate(50.0, _reservoir(catchment_km2=100_000.0), 0.35)
    assert res.overtops is True


def test_water_balance_subtracts_outflow_and_evaporation():
    # Start 6e6; inflow=350000; outflow=864000; evap=40000 -> new = 5.446e6
    r = _reservoir(current_volume_m3=6_000_000.0, level_source="swot")
    res = ReservoirFillEstimator.estimate(10.0, r, 0.35, duration_hours=240, evap_mm_day=4.0, outflow_m3s=1.0)
    assert res.outflow_m3 == pytest.approx(864_000.0)
    assert res.evap_m3 == pytest.approx(40_000.0)
    assert res.new_volume_m3 == pytest.approx(5_446_000.0)
    assert res.new_fill_pct == pytest.approx(54.46)
    assert res.delta_level_m < 0.0                       # Outflow > Inflow -> level drops


def test_no_duration_means_no_losses():
    r = _reservoir(current_volume_m3=6_000_000.0, level_source="swot")
    # Without duration, outflow/evap don't apply
    res = ReservoirFillEstimator.estimate(10.0, r, 0.35, evap_mm_day=4.0, outflow_m3s=1.0)
    assert res.outflow_m3 == 0.0 and res.evap_m3 == 0.0
    assert res.new_volume_m3 == pytest.approx(6_350_000.0)


def test_new_volume_clamped_nonnegative():
    # Massive outflows cannot drop volume below 0
    r = _reservoir(current_volume_m3=1_000_000.0, catchment_km2=None, level_source="swot")
    res = ReservoirFillEstimator.estimate(0.0, r, 0.35, duration_hours=1000, outflow_m3s=100.0)
    assert res.new_volume_m3 == 0.0


@pytest.mark.parametrize("reservoir", [None, {"max_volume_m3": 0.0}, {"max_volume_m3": -1.0}])
def test_estimate_none_when_no_reservoir_or_capacity(reservoir):
    assert ReservoirFillEstimator.estimate(10.0, reservoir, 0.35) is None


# --- Optional integration tests with precalculated data ---

@pytest.mark.skipif(not os.path.exists(AUGMENT), reason="dem_augment.json not built")
def test_dem_augment_known_reservoir():
    aug = json.load(open(AUGMENT, encoding="utf-8"))
    if "Vidraru" not in aug:
        pytest.skip("Vidraru not processed yet")
    v = aug["Vidraru"]
    assert v["source"] == "dem"
    assert v["catchment_km2"] == pytest.approx(294, abs=25)
    curve = StageStorageCurve.from_dict(v["stage_storage"])
    assert curve.waterline_m > 700
    assert curve.volume_at_level(5.0) > curve.v_nnr_m3


@pytest.mark.skipif(not os.path.exists(LEVELS), reason="reservoir_levels.json not built")
def test_swot_levels_have_plausible_wse():
    lv = json.load(open(LEVELS, encoding="utf-8"))
    assert len(lv) > 0
    for name, d in lv.items():
        assert d["source"] == "swot"
        assert -100.0 < d["wse_m"] < 3000.0            # Plausible elevations in Romania


@pytest.mark.skipif(not os.path.exists(LEVELS), reason="reservoir_levels.json not built")
def test_covered_scope_is_data_driven():
    from src.geo.reservoir_loader import ReservoirLoader
    covered = ReservoirLoader.get_covered_reservoirs()
    all_res = ReservoirLoader.get_all_reservoirs()
    assert 0 < len(covered) < len(all_res)             # Limited scope vs full shapefile
    assert all(r["level_source"] in ("swot", "s2") for r in covered.values())
    assert all(r["current_volume_m3"] is not None for r in covered.values())


@pytest.mark.skipif(not os.path.exists(LEVELS_S2), reason="reservoir_levels_s2.json not built")
def test_s2_levels_present_and_plausible():
    lv = json.load(open(LEVELS_S2, encoding="utf-8"))
    assert len(lv) > 0
    for name, d in lv.items():
        assert d["source"] == "s2"
        assert -100.0 < d["wse_m"] < 3000.0
        assert 0.0 <= d.get("valid_frac", 1.0) <= 1.0
