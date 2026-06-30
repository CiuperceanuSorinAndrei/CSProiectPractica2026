import pytest
import numpy as np
from src.core.frame_processor import FrameProcessor
from src.core.storm_tracker import StormTracker
from src.core.domain import StormCell
from src.io.frame_preprocessor import FramePrep, FrameGeometry

def test_frame_processor_e2e_pipeline():
    """
    Test End-To-End care valideaza ca motorul de procesare poate prelua preprocesarea, 
    urmarirea, filtrarea Kalman, calcularea tendintelor si advectia S-PROG 
    fara a genera vreo exceptie (TypeErrors / AttributeErrors).
    """
    tracker = StormTracker()
    predictions_queue = []

    # Generam o secventa de 4 cadre cu o celula de ploaie care se misca
    for frame_idx in range(4):
        # 1. Cream matrice de ploaie goala
        rain_rate = np.zeros((100, 100), dtype=np.float32)
        
        # 2. Simulam o "furtuna" care se misca dreapta-jos (vx=2, vy=2)
        center_y = 30 + frame_idx * 2
        center_x = 40 + frame_idx * 2
        radius = 10
        
        # Desenam ploaie in jurul centrului
        y_grid, x_grid = np.mgrid[0:100, 0:100]
        dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        rain_rate[dist < radius] = 10.0 # 10 mm/h
        
        # 3. Simulam celula preprocesata
        c = StormCell(
            id=-1, # Unassigned
            centroid_x=float(center_x),
            centroid_y=float(center_y),
            area_pixels=int(np.pi * radius**2),
            geo_lon=25.0 + frame_idx * 0.1,
            geo_lat=45.0 + frame_idx * 0.1,
            max_intensity=10.0
        )
        # Adaugam campurile necesare pentru track
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
        
        # 4. Rulam FrameProcessor
        try:
            result = FrameProcessor.process(prep, geom, tracker, predictions_queue)
            
            # Verificam contractul (DTOs)
            assert hasattr(result, "tracked_cells")
            assert hasattr(result, "roi_volume_m3")
            assert hasattr(result, "predicted_roi_volume_m3")
            
            # La primele cadre s-ar putea sa nu avem previziuni (nevoie de history pentru v)
            if frame_idx >= 2:
                # O celula ar trebui sa fie trackuita acum
                assert len(result.tracked_cells) > 0
                cell_dto = result.tracked_cells[0]
                assert "id" in cell_dto
                assert "v_x" in cell_dto
                
        except Exception as e:
            pytest.fail(f"FrameProcessor a esuat la cadrul {frame_idx} cu eroarea: {str(e)}")
