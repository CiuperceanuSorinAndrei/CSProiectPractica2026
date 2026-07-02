import os

import pytest

from src.geo.reservoir_fill import ReservoirFillEstimator

SHAPEFILE = "data/geo/reservoirs/LacuriAcumulare.shp"


# --- accumulated_volume_m3: adancime (mm) * arie (m^2) -> volum (m^3) --------------------

def test_accumulated_volume_basic():
    # 10 mm (= 0.01 m) peste 1 000 000 m^2 = 10 000 m^3
    assert ReservoirFillEstimator.accumulated_volume_m3(10.0, 1_000_000.0) == pytest.approx(10_000.0)


def test_accumulated_volume_one_mm_over_one_m2_is_one_litre():
    # 1 L/m^2 peste 1 m^2 = 1 L = 0.001 m^3 (verifica factorul mm->m)
    assert ReservoirFillEstimator.accumulated_volume_m3(1.0, 1.0) == pytest.approx(0.001)


@pytest.mark.parametrize("map_mm, area", [(0.0, 1e6), (-5.0, 1e6), (10.0, 0.0), (10.0, -1.0), (None, 1e6), (10.0, None)])
def test_accumulated_volume_nonpositive_or_missing_is_zero(map_mm, area):
    assert ReservoirFillEstimator.accumulated_volume_m3(map_mm, area) == 0.0


# --- fill_percentage: volum acumulat ca procent din volumul maxim ------------------------

def test_fill_percentage_known_value():
    # 100 mm peste 1 km^2 (1e6 m^2) = 100 000 m^3; fata de 1 mil m^3 -> 10%
    pct = ReservoirFillEstimator.fill_percentage(100.0, 1_000_000.0, 1_000_000.0)
    assert pct == pytest.approx(10.0)


def test_fill_percentage_matches_suhaia_hand_calc():
    # Suhaia: ~9.99 km^2, 18 mil m^3. 50 mm -> 0.05 * 9.99e6 / 18e6 * 100 = 2.775%
    pct = ReservoirFillEstimator.fill_percentage(50.0, 9_989_791.6, 18_000_000.0)
    assert pct == pytest.approx(2.775, abs=1e-3)


def test_fill_percentage_can_exceed_100():
    # Un episod extrem poate depasi capacitatea; nu plafonam.
    pct = ReservoirFillEstimator.fill_percentage(1000.0, 10_000_000.0, 1_000_000.0)
    assert pct == pytest.approx(1000.0)
    assert pct > 100.0


def test_fill_percentage_zero_rain_is_zero():
    assert ReservoirFillEstimator.fill_percentage(0.0, 1_000_000.0, 1_000_000.0) == 0.0


@pytest.mark.parametrize("max_vol", [0.0, -1.0, None])
def test_fill_percentage_none_when_capacity_unknown(max_vol):
    # Fara o capacitate valida nu putem raporta -> None (cardul ramane fara procent).
    assert ReservoirFillEstimator.fill_percentage(100.0, 1_000_000.0, max_vol) is None


# --- fill_percentage_for: varianta convenabila pe intrarea ReservoirLoader ----------------

def test_fill_percentage_for_reads_reservoir_dict():
    reservoir = {"surface_area_m2": 1_000_000.0, "max_volume_m3": 1_000_000.0}
    assert ReservoirFillEstimator.fill_percentage_for(100.0, reservoir) == pytest.approx(10.0)


def test_fill_percentage_for_none_reservoir():
    # Modul oras/cerc (fara lac) nu produce procent.
    assert ReservoirFillEstimator.fill_percentage_for(100.0, None) is None


def test_fill_percentage_for_missing_keys_returns_none():
    # Lac fara volum maxim cunoscut -> None (get(...) implicit 0.0 -> capacitate invalida).
    assert ReservoirFillEstimator.fill_percentage_for(100.0, {"surface_area_m2": 1_000_000.0}) is None


# --- Integrare cu shapefile-ul real (se sare daca lipsesc datele) -------------------------

@pytest.mark.skipif(not os.path.exists(SHAPEFILE), reason="Shapefile-ul cu lacuri nu este disponibil")
def test_loader_attaches_volume_and_area_for_known_reservoir():
    from src.geo.reservoir_loader import ReservoirLoader

    reservoirs = ReservoirLoader.get_all_reservoirs(SHAPEFILE)
    assert "Suhaia" in reservoirs, "lacul de referinta Suhaia ar trebui sa existe in shapefile"

    suhaia = reservoirs["Suhaia"]
    # Volumul maxim cunoscut al Suhaiei: 18 milioane m^3.
    assert suhaia["vol_mil_m3"] == pytest.approx(18.0)
    assert suhaia["max_volume_m3"] == pytest.approx(18_000_000.0)
    # Suprafata luciului ~9.99 km^2 (derivata din geometria proiectata, in m^2).
    assert suhaia["surface_area_m2"] == pytest.approx(9.99e6, rel=0.01)

    # Contract end-to-end: 50 mm acumulate -> ~2.78% din capacitate.
    pct = ReservoirFillEstimator.fill_percentage_for(50.0, suhaia)
    assert pct == pytest.approx(2.78, abs=0.05)
