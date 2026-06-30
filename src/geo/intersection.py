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

    @staticmethod
    def create_polygon_mask(polygon, lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
        """Creeaza o masca booleana (acelasi shape ca grid-ul) pentru interiorul unui poligon."""
        if polygon is None or lon_grid.shape != lat_grid.shape:
            return np.zeros_like(lon_grid, dtype=bool)
            
        # Aplatizam grid-ul pentru shapely.points
        flat_lons = lon_grid.ravel()
        flat_lats = lat_grid.ravel()
        
        point_geoms = shapely.points(flat_lons, flat_lats)
        mask_flat = shapely.contains(polygon, point_geoms)
        
        return mask_flat.reshape(lon_grid.shape)

    @staticmethod
    def create_fractional_mask(polygon, lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
        """Creeaza o masca fractionara [0.0, 1.0] bazata pe acoperirea exacta a poligonului pe fiecare pixel."""
        mask_frac = np.zeros_like(lon_grid, dtype=np.float32)
        if polygon is None or lon_grid.shape != lat_grid.shape:
            return mask_frac
            
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        buffer = 0.05
        
        y_idx, x_idx = np.where(
            (lon_grid >= min_lon - buffer) & (lon_grid <= max_lon + buffer) &
            (lat_grid >= min_lat - buffer) & (lat_grid <= max_lat + buffer)
        )
        
        if len(y_idx) == 0:
            return mask_frac
            
        from shapely.geometry import Polygon
        
        lat_diff_y = np.gradient(lat_grid, axis=0)
        lon_diff_x = np.gradient(lon_grid, axis=1)
        
        for i, j in zip(y_idx, x_idx):
            c_lon = lon_grid[i, j]
            c_lat = lat_grid[i, j]
            
            dlon = abs(lon_diff_x[i, j]) / 2.0
            dlat = abs(lat_diff_y[i, j]) / 2.0
            
            pixel_poly = Polygon([
                (c_lon - dlon, c_lat - dlat),
                (c_lon + dlon, c_lat - dlat),
                (c_lon + dlon, c_lat + dlat),
                (c_lon - dlon, c_lat + dlat)
            ])
            
            if polygon.intersects(pixel_poly):
                intersection = polygon.intersection(pixel_poly)
                mask_frac[i, j] = intersection.area / pixel_poly.area
                
        return mask_frac
