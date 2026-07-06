"""Pachet core: detectie celule, tracking cinematic, nowcasting si metrici de validare.

Sub-pachete:
  - detection: detectie celule intr-un singur cadru
  - tracking:  matching KD-Tree + filtre Kalman + optical flow + lifecycle
  - nowcast:   extrapolare prin advectie + energetica Reaction-Diffusion
  - metrics:   scoruri de validare (CSI/FAR/POD/FSS) + integrare volumetrica
  - pipeline:  orchestrare per-cadru + cache de preprocesare

`domain` (StormCell) ramane la radacina, fiind partajat de toate sub-pachetele.
"""
from .detection.storm_cell_detector import StormCellDetector
from .tracking.storm_tracker import StormTracker

__all__ = ["StormCellDetector", "StormTracker"]
