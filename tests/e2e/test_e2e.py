import pytest
import numpy as np
from src.core.pipeline.frame_processor import FrameProcessor
from src.core.tracking.storm_tracker import StormTracker
from src.core.domain import StormCell
from src.io.frame_preprocessor import FramePrep, FrameGeometry

def test_frame_processor_e2e_pipeline():
    """
    End-To-End test validating that the processing engine can handle preprocessing,
    tracking, Kalman filtering, trend calculation, and S-PROG advection
    without raising any exceptions (TypeErrors / AttributeErrors).
    """
    tracker = StormTracker()

    # Generate a sequence of 4 frames with a moving rain cell
    for frame_idx in range(4):
        # 1. Create empty rain matrix
        rain_rate = np.zeros((100, 100), dtype=np.float32)
        
        # 2. Simulate a "storm" moving bottom-right (vx=2, vy=2)
        center_y = 30 + frame_idx * 2
        center_x = 40 + frame_idx * 2
        radius = 10
        
        # Draw rain around the center
        y_grid, x_grid = np.mgrid[0:100, 0:100]
        dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        rain_rate[dist < radius] = 10.0 # 10 mm/h
        
        # 3. Simulate preprocessed cell
        c = StormCell(
            id=-1, # Unassigned
            centroid_x=float(center_x),
            centroid_y=float(center_y),
            area_pixels=int(np.pi * radius**2),
            geo_lon=25.0 + frame_idx * 0.1,
            geo_lat=45.0 + frame_idx * 0.1,
            max_intensity=10.0
        )
        # Add required tracking fields
        c.coords = np.argwhere(rain_rate > 0)
        
        prep = FramePrep(rain_rate=rain_rate, filtered_cells=[c], max_rain=10.0)
        
        geom = FrameGeometry(
            lon_grid=np.zeros((100, 100)),
            lat_grid=np.zeros((100, 100)),
            pixel_area_km2=np.full((100, 100), 9.0),
            roi_mask=np.ones((100, 100), dtype=bool),
            y_slice=slice(0, 100),
            x_slice=slice(0, 100)
        )
        
        # 4. Run FrameProcessor
        try:
            from src.core.nowcast.advection_engine import AdvectionEngine
            from src.core.nowcast.kinematic_advector import KinematicAdvector
            
            engine = AdvectionEngine(KinematicAdvector())
            
            result = FrameProcessor.process(prep, geom, tracker, engine)
            
            # Verify the contract (DTOs)
            assert hasattr(result, "tracked_cells")
            assert hasattr(result, "roi_map_mm")
            assert hasattr(result, "predicted_roi_map_mm")
            
            # First few frames might lack predictions (history needed for velocity)
            if frame_idx >= 2:
                # A cell should be tracked by now
                assert len(result.tracked_cells) > 0
                cell_dto = result.tracked_cells[0]
                assert "id" in cell_dto
                assert "v_x" in cell_dto
                
        except Exception as e:
            pytest.fail(f"FrameProcessor failed at frame {frame_idx} with error: {str(e)}")
