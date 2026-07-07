"""Facade for the tracking and frame processing systems. Ensures thread safety."""
from __future__ import annotations
import threading
from typing import Optional, Any

from src.core.tracking.storm_tracker import StormTracker
from src.core.pipeline.cache_manager import CacheManager
from src.core.pipeline.frame_processor import FrameProcessor, FrameResult
from src.core.nowcast.advection_engine import AdvectionEngine
from src.core.nowcast.kinematic_advector import KinematicAdvector
from src.config import MAX_TRACKING_DISTANCE_PX

class ServerBusy(Exception):
    """Raised when a frame is already being processed."""
    pass

class Orchestrator:
    """Facade for tracking and frame processing services."""

    def __init__(self) -> None:
        self._tracker = StormTracker(max_dist_pixels=MAX_TRACKING_DISTANCE_PX)
        self._lock = threading.Lock()
        self._cache_manager = CacheManager(self._lock)
        self._advection_engine = AdvectionEngine(
            KinematicAdvector()
        )

    def reset_tracking(self) -> None:
        """Clears the tracking state."""
        with self._lock:
            self._tracker.reset()
            self._advection_engine.reset_feedback()

    def process_frame(
        self,
        file_path: str,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float, 
        polygon: Optional[Any] = None,
        frame_time: Optional[str] = None, 
        run_mode: str = "historic"
    ) -> FrameResult | None:
        self._cache_manager.update_activity()
        
        if not self._lock.acquire(timeout=0.5):
            raise ServerBusy()

        try:
            prep = self._cache_manager.get_frame_prep(
                file_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km, polygon
            )
            if prep is None or self._cache_manager.geometry is None:
                return None
                
            result = FrameProcessor.process(
                prep, self._cache_manager.geometry, self._tracker, self._advection_engine,
                frame_time=frame_time, run_mode=run_mode
            )
            return result
        finally:
            self._lock.release()

    def start_warmup(
        self,
        file_paths: list[str],
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float, 
        polygon: Optional[Any] = None
    ) -> None:
        self._cache_manager.start_warmup(
            file_paths, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km, polygon
        )

    def stop_warmup(self) -> None:
        self._cache_manager.stop_warmup()

    def warm_status(self) -> tuple[int, int]:
        return self._cache_manager.warm_status()
