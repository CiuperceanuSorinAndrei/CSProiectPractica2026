"""Teste hermetice pentru rutarea scurgerii (priority-flood + D8 + trasare amonte).

Folosesc DEM-uri sintetice mici, deterministe - fara descarcari DEM.
"""
import numpy as np
import pytest
from shapely.geometry import box

from src.geo.dem_source import DemWindow
from src.geo.catchment import (
    priority_flood_fill, d8_receivers, upstream_mask, delineate_catchment,
)


def test_d8_drains_down_south_slope():
    # panta pura spre sud: fiecare celula se scurge exact spre celula de sub ea
    dem = np.array([[40, 40, 40],
                    [30, 30, 30],
                    [20, 20, 20],
                    [10, 10, 10]], dtype=float)
    rec = d8_receivers(priority_flood_fill(dem))
    nx = 3
    # (0,1) -> (1,1) -> (2,1) -> (3,1)
    assert rec[0 * nx + 1] == 1 * nx + 1
    assert rec[1 * nx + 1] == 2 * nx + 1
    assert rec[2 * nx + 1] == 3 * nx + 1


def test_upstream_mask_collects_column():
    dem = np.array([[40, 40, 40],
                    [30, 30, 30],
                    [20, 20, 20],
                    [10, 10, 10]], dtype=float)
    rec = d8_receivers(priority_flood_fill(dem))
    nx = 3
    seed = 3 * nx + 1                     # celula (3,1)
    seen = upstream_mask(rec, [seed], dem.size)
    # amonte de (3,1) = coloana 1 (randurile 0..3)
    expected = {r * nx + 1 for r in range(4)}
    assert set(np.flatnonzero(seen)) == expected


def test_pit_is_filled_no_interior_sink():
    # groapa interioara la (1,1)=1, inconjurata de teren mai inalt; exutor la coltul (2,2)=2
    dem = np.array([[3, 3, 3],
                    [3, 1, 3],
                    [3, 3, 2]], dtype=float)
    filled = priority_flood_fill(dem)
    # groapa e ridicata la nivelul de deversare (~2), deci nu mai e un minim local
    assert filled[1, 1] == pytest.approx(2.0, abs=0.05)
    assert filled[1, 1] > dem[1, 1]


def _slope_window():
    dem = np.array([[40, 40, 40],
                    [30, 30, 30],
                    [20, 20, 20],
                    [10, 10, 10],
                    [0, 0, 0]], dtype=np.float32)
    return DemWindow(dem=dem, lon0=25.0, lat0=45.0, px=0.01)


def test_delineate_synthetic_slope():
    w = _slope_window()
    # lac peste celula (3,1): centru la lon 25.015, lat 44.965
    cx = 25.0 + 1.5 * 0.01
    cy = 45.0 - 3.5 * 0.01
    poly = box(cx - 0.004, cy - 0.004, cx + 0.004, cy + 0.004)
    res = delineate_catchment(w, poly, downsample=1)
    # bazinul lacului (3,1) = coloana de deasupra (randurile 0..3) = 4 celule vs 1 celula lac
    assert round(res["catchment_km2"] / res["lake_km2"]) == 4
    assert res["edge_clipped"] is True     # bazinul atinge randul de sus (marginea)
