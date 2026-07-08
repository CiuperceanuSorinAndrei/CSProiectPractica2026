# Copernicus GLO-30 DEM access: downloads and mosaics .tif COG tiles without GDAL.
from __future__ import annotations

import os
import math
import urllib.request
import functools
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile

GLO30_PX = 1.0 / 3600.0            # deg/pixel (uniform in 0-50 N band)
_TILE = 3600                        # pixels/side for a 1 degree tile
_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
_M_PER_DEG = 111320.0


def _tile_id(south: int, west: int) -> str:
    ns = f"N{south:02d}" if south >= 0 else f"S{abs(south):02d}"
    ew = f"E{west:03d}" if west >= 0 else f"W{abs(west):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


@dataclass
class DemWindow:
    # Geographic DEM window. lon0/lat0 is NW corner of pixel [0,0].
    dem: np.ndarray      # float32 [row, col], row 0 = north
    lon0: float
    lat0: float
    px: float

    @property
    def shape(self):
        return self.dem.shape

    def centers_1d(self):
        ny, nx = self.dem.shape
        lon = self.lon0 + (np.arange(nx) + 0.5) * self.px
        lat = self.lat0 - (np.arange(ny) + 0.5) * self.px
        return lon, lat

    def cell_area_m2(self) -> np.ndarray:
        # Area of each pixel in m^2 (decreases with latitude)
        _, lat = self.centers_1d()
        dy = self.px * _M_PER_DEG
        dx = self.px * _M_PER_DEG * np.cos(np.radians(lat))
        return (dy * dx)[:, None] * np.ones((1, self.dem.shape[1]), dtype=np.float64)

    def water_mask(self, polygon) -> np.ndarray:
        # Mask of pixels whose center falls inside the lake polygon
        import shapely
        lon, lat = self.centers_1d()
        LON, LAT = np.meshgrid(lon, lat)
        return shapely.contains_xy(polygon, LON.ravel(), LAT.ravel()).reshape(self.dem.shape)


class DemSource:
    # Downloads and caches GLO-30 tiles, providing mosaics for arbitrary windows

    def __init__(self, cache_dir: str, timeout: float = 120.0):
        self.cache_dir = cache_dir
        self.timeout = timeout
        os.makedirs(cache_dir, exist_ok=True)

    def _tile_path(self, south: int, west: int) -> str | None:
        # Local tile path (downloads if missing). None if not on server.
        # Check local cache
        name = _tile_id(south, west)
        path = os.path.join(self.cache_dir, name + ".tif")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        # Download from S3 with temporary file
        url = f"{_BASE}/{name}/{name}.tif"
        tmp = path + ".part"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp, open(tmp, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            os.replace(tmp, path)
            return path
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            return None  # Tile missing (e.g. over ocean) or network error

    @functools.lru_cache(maxsize=32)
    def _read_tile(self, south: int, west: int) -> np.ndarray | None:
        path = self._tile_path(south, west)
        if path is None:
            return None
        with tifffile.TiffFile(path) as tf:
            return tf.pages[0].asarray()

    def mosaic(self, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> DemWindow | None:
        # Mosaic tiles covering bbox and crop to bbox. None if outside coverage.
        # Calculate bounding tiles
        w0, w1 = math.floor(lon_min), math.floor(lon_max)
        s0, s1 = math.floor(lat_min), math.floor(lat_max)

        # Pre-fetch tiles in parallel to saturate network
        pairs = [(south, west) for south in range(s1, s0 - 1, -1) for west in range(w0, w1 + 1)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda p: self._tile_path(*p), pairs))

        # Initialize mosaic array
        nx_t, ny_t = (w1 - w0 + 1), (s1 - s0 + 1)
        big = np.full((ny_t * _TILE, nx_t * _TILE), np.nan, dtype=np.float32)
        got = False
        for ti, south in enumerate(range(s1, s0 - 1, -1)):     # north -> south
            for tj, west in enumerate(range(w0, w1 + 1)):       # west -> east
                arr = self._read_tile(south, west)
                if arr is not None:
                    big[ti * _TILE:(ti + 1) * _TILE, tj * _TILE:(tj + 1) * _TILE] = arr[:_TILE, :_TILE]
                    got = True
        if not got:
            return None

        # Crop mosaic to exact geographic bounds
        block_lon0 = float(w0)
        block_lat0 = float(s1 + 1)
        c0 = max(int(round((lon_min - block_lon0) / GLO30_PX)), 0)
        c1 = min(int(round((lon_max - block_lon0) / GLO30_PX)), big.shape[1])
        r0 = max(int(round((block_lat0 - lat_max) / GLO30_PX)), 0)
        r1 = min(int(round((block_lat0 - lat_min) / GLO30_PX)), big.shape[0])
        if c1 <= c0 or r1 <= r0:
            return None
        return DemWindow(
            dem=big[r0:r1, c0:c1].copy(),
            lon0=block_lon0 + c0 * GLO30_PX,
            lat0=block_lat0 - r0 * GLO30_PX,
            px=GLO30_PX,
        )
