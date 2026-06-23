from __future__ import annotations

import numpy as np
from shapely.geometry import Point, shape
from shapely.prepared import prep


class PolygonIntersection:

    @staticmethod
    def mask_fraction_inside(
        mask: np.ndarray,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        polygon_geojson: dict,
    ) -> float:
        """Fractia de pixeli din masca aflati in interiorul poligonului GeoJSON.

        Foloseste Shapely prepared geometry pentru a accelera testul contains
        pentru query-uri multiple.
        """
        polygon = shape(polygon_geojson)
        if mask.shape != lon_grid.shape or mask.shape != lat_grid.shape:
            return 0.0

        points = np.argwhere(mask > 0)
        if len(points) == 0:
            return 0.0

        prepared_poly = prep(polygon)
        hits = sum(
            1
            for y_idx, x_idx in points
            if prepared_poly.contains(Point(lon_grid[y_idx, x_idx], lat_grid[y_idx, x_idx]))
        )
        return hits / len(points)
