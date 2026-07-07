"""Core package: cell detection, kinematic tracking, nowcasting, and validation metrics.

Sub-packages:
  - detection: single-frame cell detection
  - tracking:  KD-Tree matching + Kalman filters + optical flow + lifecycle
  - nowcast:   advection extrapolation + Reaction-Diffusion energetics
  - metrics:   validation scores (CSI/FAR/POD/FSS) + volumetric integration
  - pipeline:  per-frame orchestration + preprocessing cache

`domain` (StormCell) remains at root, shared by all sub-packages.
"""
from .detection.storm_cell_detector import StormCellDetector
from .tracking.storm_tracker import StormTracker

__all__ = ["StormCellDetector", "StormTracker"]
