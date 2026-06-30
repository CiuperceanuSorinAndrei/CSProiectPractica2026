import pytest
import numpy as np

from src.core.domain import StormCell
from src.diagnostics.autopsier import Autopsier, FailureCluster

def test_autopsier_over_persistence():
    # Predictia a persistat dar celula reala a murit (None)
    pred_cell = StormCell(id=1, centroid_y=10.0, centroid_x=10.0, area_pixels=50, max_intensity=1.0, mean_intensity=0.5, coords=np.array([]))
    pred_cell.lifecycle_phase = 'MATURITY'
    pred_cell.predicted_area_kalman = 40.0
    
    cluster = Autopsier.classify_cluster(pred_cell, None)
    assert cluster == FailureCluster.OVER_PERSISTENCE
    
    attr = Autopsier.attribute_error(pred_cell, cluster)
    assert attr.decay_pct == 80.0
    
def test_autopsier_early_collapse():
    # Predictia crede ca celula e la inceput, dar realitatea are arie dubla
    pred_cell = StormCell(id=2, centroid_y=20.0, centroid_x=20.0, area_pixels=10, max_intensity=1.0, mean_intensity=0.5, coords=np.array([]))
    pred_cell.lifecycle_phase = 'BIRTH'
    pred_cell.predicted_area_kalman = 15.0
    
    act_cell = StormCell(id=2, centroid_y=20.0, centroid_x=20.0, area_pixels=40, max_intensity=2.0, mean_intensity=1.5, coords=np.array([]))
    
    cluster = Autopsier.classify_cluster(pred_cell, act_cell)
    assert cluster == FailureCluster.EARLY_COLLAPSE
    
def test_autopsier_position_drift():
    # Centroidul e ratat enorm
    pred_cell = StormCell(id=3, centroid_y=50.0, centroid_x=50.0, area_pixels=20, max_intensity=1.0, mean_intensity=0.5, coords=np.array([]))
    pred_cell.lifecycle_phase = 'MATURITY'
    act_cell = StormCell(id=3, centroid_y=100.0, centroid_x=100.0, area_pixels=25, max_intensity=1.0, mean_intensity=0.5, coords=np.array([]))
    
    cluster = Autopsier.classify_cluster(pred_cell, act_cell)
    assert cluster == FailureCluster.POSITION_DRIFT
    
def test_autopsier_evaluation_report():
    pred_cells = [
        StormCell(id=1, centroid_y=10.0, centroid_x=10.0, area_pixels=50, max_intensity=1.0, mean_intensity=0.5, coords=np.array([])),
        StormCell(id=2, centroid_y=20.0, centroid_x=20.0, area_pixels=10, max_intensity=1.0, mean_intensity=0.5, coords=np.array([]))
    ]
    pred_cells[0].predicted_area_kalman = 60.0
    pred_cells[1].predicted_area_kalman = 10.0
    
    act_cells = [
        StormCell(id=1, centroid_y=10.0, centroid_x=10.0, area_pixels=20, max_intensity=1.0, mean_intensity=0.5, coords=np.array([])),
        StormCell(id=3, centroid_y=30.0, centroid_x=30.0, area_pixels=15, max_intensity=1.0, mean_intensity=0.5, coords=np.array([])) # missed
    ]
    
    matches = {1: 1} # P_1 -> A_1, P_2 nu gaseste match, A_3 ratata
    
    report = Autopsier.evaluate(step=2, predicted_cells=pred_cells, actual_cells=act_cells, iou_matches=matches)
    
    assert report.forecast_step == 2
    assert len(report.worst_cells) == 2 # 2 celule prezise verificate
    assert len(report.missed_cells) == 1
    assert report.missed_cells[0].id == 3
