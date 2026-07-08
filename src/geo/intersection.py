from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import shape


class PolygonIntersection:

    @staticmethod
    def create_fractional_mask(polygon, lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
        # Create fractional mask [0.0, 1.0] based on exact polygon coverage
        # Validate inputs and initialize fractional mask
        mask_frac = np.zeros_like(lon_grid, dtype=np.float32)
        if polygon is None or lon_grid.shape != lat_grid.shape:
            return mask_frac
            
        # Define bounding box filter to optimize calculations
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        buffer = 0.05
        
        y_idx, x_idx = np.where(
            (lon_grid >= min_lon - buffer) & (lon_grid <= max_lon + buffer) &
            (lat_grid >= min_lat - buffer) & (lat_grid <= max_lat + buffer)
        )
        
        if len(y_idx) == 0:
            return mask_frac
            
        # Compute local grid gradients and point geometries
        from shapely.geometry import Polygon
        
        lat_diff_y = np.gradient(lat_grid, axis=0)
        lon_diff_x = np.gradient(lon_grid, axis=1)
        
        bbox_lons = lon_grid[y_idx, x_idx]
        bbox_lats = lat_grid[y_idx, x_idx]
        point_geoms = shapely.points(bbox_lons, bbox_lats)
        
        # Fast binary classification
        inside_mask = shapely.contains(polygon, point_geoms)
        
        # Buffer distance to identify boundary pixels
        max_dlon = np.max(np.abs(lon_diff_x[y_idx, x_idx])) if len(y_idx) > 0 else 0.0
        max_dlat = np.max(np.abs(lat_diff_y[y_idx, x_idx])) if len(y_idx) > 0 else 0.0
        pixel_radius = max(max_dlon, max_dlat)
        
        exterior = polygon.exterior
        near_boundary_mask = shapely.dwithin(point_geoms, exterior, pixel_radius * 1.5)
        
        # Calculate fractional coverage for boundary pixels
        for idx, (i, j) in enumerate(zip(y_idx, x_idx)):
            if inside_mask[idx] and not near_boundary_mask[idx]:
                mask_frac[i, j] = 1.0
                continue
                
            if not near_boundary_mask[idx]:
                continue
                
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
