from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import shape


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

        # V24 Fix: Vectorizare shapely 2.0 in loc de for loop nativ Python cu shapely.Point instantiat iterativ
        lons = lon_grid[points[:, 0], points[:, 1]]
        lats = lat_grid[points[:, 0], points[:, 1]]
        
        point_geoms = shapely.points(lons, lats)
        hits = np.sum(shapely.contains(polygon, point_geoms))
        
        return float(hits) / len(points)
