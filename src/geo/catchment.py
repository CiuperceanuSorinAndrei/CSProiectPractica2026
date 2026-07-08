# Catchment delineation using priority-flood depression filling and upstream tracing.
from __future__ import annotations

import heapq
from collections import deque

import numpy as np
import cv2
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon

from src.geo.dem_source import DemWindow, _M_PER_DEG

_NB = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    # Subsample array by taking kxk block means
    ny, nx = a.shape
    ny -= ny % k; nx -= nx % k
    return a[:ny, :nx].reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))


def priority_flood_fill(dem: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    # Fill DEM depressions using Barnes priority-flood algorithm
    # Initialize arrays and priority queue
    ny, nx = dem.shape
    filled = np.where(np.isfinite(dem), dem, 1e9).astype(np.float64)
    out = np.full((ny, nx), np.inf)
    pq: list[tuple[float, int]] = []
    # Seed priority queue with border cells
    for c in range(nx):
        for r in (0, ny - 1):
            out[r, c] = filled[r, c]; heapq.heappush(pq, (filled[r, c], r * nx + c))
    for r in range(ny):
        for c in (0, nx - 1):
            if out[r, c] == np.inf:
                out[r, c] = filled[r, c]; heapq.heappush(pq, (filled[r, c], r * nx + c))
    # Process queue and fill depressions
    while pq:
        e, idx = heapq.heappop(pq)
        r, c = divmod(idx, nx)
        for dr, dc in _NB:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ny and 0 <= nc < nx and out[nr, nc] == np.inf:
                out[nr, nc] = filled[nr, nc] if filled[nr, nc] > e + eps else e + eps
                heapq.heappush(pq, (out[nr, nc], nr * nx + nc))
    return out


def d8_receivers(filled: np.ndarray) -> np.ndarray:
    # Calculate D8 flow receivers via steepest descent
    # Initialize receiver matrix
    ny, nx = filled.shape
    rec = np.full(ny * nx, -1, dtype=np.int64)
    best = np.zeros((ny, nx))
    pad = np.pad(filled, 1, constant_values=np.inf)
    rows = np.arange(ny)[:, None]; cols = np.arange(nx)[None, :]
    # Find steepest descent neighbor
    for dr, dc in _NB:
        dist = (dr * dr + dc * dc) ** 0.5
        neigh = pad[1 + dr:1 + dr + ny, 1 + dc:1 + dc + nx]
        drop = (filled - neigh) / dist
        rr, cc = rows + dr, cols + dc
        valid = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx) & (drop > best)
        idx = np.where(valid, (rr * nx + cc), rec.reshape(ny, nx))
        rec = np.where(valid.ravel(), idx.ravel(), rec)
        best = np.where(valid, drop, best)
    return rec


def upstream_mask(rec: np.ndarray, seeds, n: int) -> np.ndarray:
    # Trace upstream drainage paths from seed cells
    # Build reverse flow graph
    donors: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        p = rec[i]
        if p >= 0:
            donors[p].append(i)
    # Initialize traversal queue
    seen = np.zeros(n, dtype=bool)
    dq = deque()
    for s in seeds:
        if not seen[s]:
            seen[s] = True; dq.append(s)
    # Traverse upstream from seeds
    while dq:
        u = dq.popleft()
        for d in donors[u]:
            if not seen[d]:
                seen[d] = True; dq.append(d)
    return seen


def delineate_catchment(window: DemWindow, polygon, downsample: int = 3) -> dict:
    # Compute catchment draining into lake polygon
    # Subsample DEM and compute coordinates
    dem = block_mean(window.dem, downsample)
    px = window.px * downsample
    ny, nx = dem.shape
    n = ny * nx

    lat = window.lat0 - (np.arange(ny) + 0.5) * px
    dy = px * _M_PER_DEG
    dx = px * _M_PER_DEG * np.cos(np.radians(lat))
    cell_km2 = ((dy * dx) / 1e6)[:, None] * np.ones((1, nx))

    # Find lake cells as seeds
    import shapely
    lon_c = window.lon0 + (np.arange(nx) + 0.5) * px
    LON, LAT = np.meshgrid(lon_c, lat)
    water = shapely.contains_xy(polygon, LON.ravel(), LAT.ravel()).reshape(dem.shape)
    seeds = list(np.flatnonzero(water))
    if not seeds:
        # Select closest cell to centroid if lake is smaller than pixel
        cy, cx = polygon.centroid.y, polygon.centroid.x
        r = int(np.clip((window.lat0 - cy) / px, 0, ny - 1))
        c = int(np.clip((cx - window.lon0) / px, 0, nx - 1))
        seeds = [r * nx + c]

    # Run catchment delineation
    filled = priority_flood_fill(dem)
    rec = d8_receivers(filled)
    seen = upstream_mask(rec, seeds, n)

    # Format and check boundary
    seen2d = seen.reshape(ny, nx)
    edge_clipped = bool(seen2d[0, :].any() or seen2d[-1, :].any()
                        or seen2d[:, 0].any() or seen2d[:, -1].any())
    # Extract catchment boundary polygon from binary mask
    catchment_wkt = None
    mask_uint8 = seen2d.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        polygons = []
        for cnt in contours:
            if len(cnt) >= 3:
                # Convert pixel coords to geographic
                coords_geo = []
                for pt in cnt:
                    col, row = int(pt[0][0]), int(pt[0][1])
                    pt_lon = window.lon0 + (col + 0.5) * px
                    pt_lat = window.lat0 - (row + 0.5) * px
                    coords_geo.append((pt_lon, pt_lat))
                if len(coords_geo) >= 3:
                    coords_geo.append(coords_geo[0])  # Close ring
                    polygons.append(ShapelyPolygon(coords_geo))
        if len(polygons) == 1:
            catchment_wkt = polygons[0].wkt
        elif len(polygons) > 1:
            catchment_wkt = MultiPolygon(polygons).wkt

    return {
        "catchment_km2": float(cell_km2.ravel()[seen].sum()),
        "lake_km2": float(cell_km2[water].sum()),
        "edge_clipped": edge_clipped,
        "n_cells": int(n),
        "catchment_wkt": catchment_wkt,
    }
