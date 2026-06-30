from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

from src.core.domain import StormCell

class FailureCluster(Enum):
    OVER_PERSISTENCE = "over_persistence"
    EARLY_COLLAPSE = "early_collapse"
    POSITION_DRIFT = "position_drift"
    NONE = "none"

@dataclass
class ErrorAttribution:
    advection_pct: float = 0.0
    decay_pct: float = 0.0
    birth_pct: float = 0.0
    noise_pct: float = 0.0

@dataclass
class CellDiagnostic:
    cell_id: int
    predicted_vol: float
    actual_vol: float
    vol_error_pct: float
    rmse: float
    cluster: FailureCluster
    attribution: ErrorAttribution

@dataclass
class DiagnosticReport:
    forecast_step: int
    worst_cells: list[CellDiagnostic] = field(default_factory=list)
    false_alarms: list[StormCell] = field(default_factory=list)
    missed_cells: list[StormCell] = field(default_factory=list)

class Autopsier:
    @staticmethod
    def classify_cluster(pred_cell: StormCell, actual_cell: StormCell | None) -> FailureCluster:
        """Clasifică eroarea pe baza comportamentului cinematic și termodinamic."""
        if actual_cell is None:
            # Daca predictia persista dar celula a murit real
            phase = getattr(pred_cell, 'lifecycle_phase', 'MATURITY')
            if phase in ['MATURITY', 'DISSIPATION'] and pred_cell.predicted_area_kalman > 0:
                return FailureCluster.OVER_PERSISTENCE
            return FailureCluster.NONE

        if getattr(pred_cell, 'lifecycle_phase', 'MATURITY') == 'BIRTH' and pred_cell.predicted_area_kalman < actual_cell.area_pixels * 0.5:
            return FailureCluster.EARLY_COLLAPSE
            
        dist = np.hypot(pred_cell.centroid_x - actual_cell.centroid_x, pred_cell.centroid_y - actual_cell.centroid_y)
        if dist > max(10, np.sqrt(actual_cell.area_pixels) * 0.5):
            return FailureCluster.POSITION_DRIFT
            
        return FailureCluster.NONE

    @staticmethod
    def attribute_error(pred_cell: StormCell, cluster: FailureCluster) -> ErrorAttribution:
        """Decompune eroarea responsabilă per componentă internă."""
        attr = ErrorAttribution()
        
        if cluster == FailureCluster.OVER_PERSISTENCE:
            attr.decay_pct = 80.0
            attr.noise_pct = 20.0
        elif cluster == FailureCluster.EARLY_COLLAPSE:
            attr.birth_pct = 70.0
            attr.decay_pct = 30.0
        elif cluster == FailureCluster.POSITION_DRIFT:
            attr.advection_pct = 85.0
            attr.noise_pct = 15.0
            
        return attr

    @classmethod
    def evaluate(cls, step: int, predicted_cells: list[StormCell], actual_cells: list[StormCell], iou_matches: dict[int, int]) -> DiagnosticReport:
        """Rulează autopsia pentru un pas predictiv (generează raport)."""
        report = DiagnosticReport(forecast_step=step)
        
        actual_dict = {c.id: c for c in actual_cells}
        
        diagnostics = []
        for p_cell in predicted_cells:
            # Gasim corespondentul daca exista (din matching/tracker pipeline)
            a_id = iou_matches.get(p_cell.id)
            a_cell = actual_dict.get(a_id) if a_id else None
            
            # Calcul eroare volumetrică
            pred_vol = float(p_cell.predicted_area_kalman)
            act_vol = float(a_cell.area_pixels) if a_cell else 0.0
            
            vol_error_pct = (pred_vol - act_vol) / max(1.0, act_vol) * 100.0
            rmse = abs(pred_vol - act_vol)
            
            # Root Cause & Error Attribution
            cluster = cls.classify_cluster(p_cell, a_cell)
            attr = cls.attribute_error(p_cell, cluster)
            
            if a_cell is None and pred_vol > 15:
                report.false_alarms.append(p_cell)
                
            diag = CellDiagnostic(
                cell_id=p_cell.id,
                predicted_vol=pred_vol,
                actual_vol=act_vol,
                vol_error_pct=vol_error_pct,
                rmse=rmse,
                cluster=cluster,
                attribution=attr
            )
            diagnostics.append(diag)
            
        for a_cell in actual_cells:
            if a_cell.id not in iou_matches.values():
                report.missed_cells.append(a_cell)
                
        # Sortam worst cells dupa RMSE
        report.worst_cells = sorted(diagnostics, key=lambda d: d.rmse, reverse=True)[:20]
        
        return report
