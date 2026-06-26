"""Orchestrator: coordoneaza procesarea cadrelor si pastreaza starea intre ele.

Preprocesarea stateless (citire netCDF4 + crop + detectie) e delegata catre
frame_preprocessor si memoizata; aici se face partea secventiala (tracking Kalman +
metrici + volum) si se coordoneaza warm-up-ul de fundal. Starea cinematica e detinuta
de StormTracker; aici pastram coada de predictii pentru metricile CSI/FAR/POD.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
import threading
import time

import numpy as np

from src.core.storm_tracker import StormTracker
from src.core.advection_engine import AdvectionEngine
from src.core.evaluator import Evaluator
from frame_preprocessor import FrameGeometry, FramePrep, compute_geometry, preprocess

from config import RAIN_THRESHOLD_MIN, MAX_TRACKING_DISTANCE_PX


class ServerBusy(Exception):
    """Ridicata cand un alt cadru este deja in procesare (lock-ul orchestratorului e ocupat)."""
    pass

@dataclass
class FrameResult:
    """Rezultatul procesarii unui cadru complet."""
    tracked_cells: list[dict[str, Any]]
    rain_rate: np.ndarray
    rain_rate_masked: np.ma.MaskedArray
    lon_grid: np.ndarray
    lat_grid: np.ndarray
    max_rain: float
    mean_centroid_error: float
    mean_size_error: float
    num_tracked: int
    roi_volume_m3: float
    predicted_roi_volume_m3: float
    predicted_volumes_horizons: dict[str, float]
    instant_predicted_volumes: dict[str, float]
    global_csi: dict[str, float]
    global_far: dict[str, float]
    global_pod: dict[str, float]
    global_fss: dict[str, float]

_PREP_CACHE_MAXSIZE = 256
_WARMUP_GRACE_S = 0.6
_WARMUP_POLL_S = 0.05

class Orchestrator:
    """Coordoneaza procesarea cadrelor si pastreaza starea cinematica intre ele."""

    def __init__(self) -> None:
        self._tracker = StormTracker(max_dist_pixels=MAX_TRACKING_DISTANCE_PX)
        self._lock = threading.Lock()
        self._predictions_queue = []

        # Cache geometrie si preprocesare
        self._geom_key: tuple | None = None
        self._geom: FrameGeometry | None = None
        self._prep_cache: OrderedDict[str, FramePrep] = OrderedDict()

        # Warm-up de fundal
        self._last_activity: float = 0.0
        self._warm_lock = threading.Lock()
        self._warm_thread: threading.Thread | None = None
        self._warm_stop: threading.Event | None = None
        self._warm_geom_key: tuple | None = None
        self._warm_complete_key: tuple | None = None
        self._warm_total: int = 0

    def reset_tracking(self) -> None:
        """Goleste complet starea de tracking (Kalman + coada predictii)."""
        with self._lock:
            self._tracker.reset()
            self._predictions_queue.clear()

    def process_frame(
        self,
        file_path: str,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float,
    ) -> FrameResult | None:
        self._last_activity = time.monotonic()
        # V24 Fix: Asteptam scurt dupa lacat pentru a preveni sufocarea UI-ului (ServerBusy)
        # cauzata de thread-ul de warmup din fundal.
        if not self._lock.acquire(timeout=0.5):
            raise ServerBusy()

        try:
            prep = self._get_frame_prep(
                file_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km,
            )
            if prep is None:
                return None
            return self._track_and_assemble(prep, self._geom)
        finally:
            self._lock.release()

    def _get_frame_prep(
        self,
        file_path: str,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float,
    ) -> FramePrep | None:
        geom_key = (lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
        if geom_key != self._geom_key:
            self._geom_key = geom_key
            self._geom = None
            self._prep_cache.clear()

        cached = self._prep_cache.get(file_path)
        if cached is not None and self._geom is not None:
            self._prep_cache.move_to_end(file_path)
            return cached

        return self._compute_prep(
            file_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km,
        )

    def _compute_prep(
        self,
        file_path: str,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float,
    ) -> FramePrep | None:
        bbox = (lon_min, lon_max, lat_min, lat_max)
        if self._geom is None:
            self._geom = compute_geometry(file_path, bbox, (center_lat, center_lon), radius_km)
            if self._geom is None:
                return None
        prep = preprocess(file_path, self._geom, bbox)
        if prep is None:
            return None
        self._prep_cache[file_path] = prep
        if len(self._prep_cache) > _PREP_CACHE_MAXSIZE:
            self._prep_cache.popitem(last=False)
        return prep

    def _track_and_assemble(self, prep: FramePrep, geom: FrameGeometry) -> FrameResult:
        rain_rate = prep.rain_rate
        roi_mask = geom.roi_mask

        # Copii superficiale pentru a nu muta celulele memoizate
        cells_for_tracking = [dict(c) for c in prep.filtered_cells]
        tracked_cells, flow = self._tracker.track(cells_for_tracking, rain_rate)

        horizons = [(2, "30m"), (4, "1h"), (8, "2h")]
        csi, far, pod, fss = Evaluator.calculate_global_metrics(
            rain_rate, roi_mask, self._predictions_queue, horizons
        )

        sparse_preds, float_preds = AdvectionEngine.extrapolate(
            rain_rate, flow, tracked_cells, horizons
        )

        self._predictions_queue.append(sparse_preds)
        if len(self._predictions_queue) > 25:
            self._predictions_queue.pop(0)

        roi_volume_m3, predicted_volumes, instant_predicted_volumes = Evaluator.calculate_volumes(
            rain_rate, float_preds, roi_mask, geom.lat_grid, horizons
        )

        valid_errors = [c.get("prediction_error_pixels", 0.0) for c in tracked_cells if c.get("is_tracked", False)]
        size_errors = [c.get("size_error_percent", 0.0) for c in tracked_cells if c.get("is_tracked", False)]

        rain_rate_masked = np.ma.masked_where(rain_rate < RAIN_THRESHOLD_MIN, rain_rate)
        
        return FrameResult(
            tracked_cells=tracked_cells,
            rain_rate=rain_rate,
            rain_rate_masked=rain_rate_masked,
            lon_grid=geom.lon_grid,
            lat_grid=geom.lat_grid,
            max_rain=prep.max_rain,
            mean_centroid_error=float(np.mean(valid_errors)) if valid_errors else 0.0,
            mean_size_error=float(np.mean(size_errors)) if size_errors else 0.0,
            num_tracked=len([c for c in tracked_cells if c.get("is_tracked", False)]),
            roi_volume_m3=roi_volume_m3,
            predicted_roi_volume_m3=predicted_volumes.get("1h", 0.0),
            predicted_volumes_horizons=predicted_volumes,
            instant_predicted_volumes=instant_predicted_volumes,
            global_csi=csi,
            global_far=far,
            global_pod=pod,
            global_fss=fss,
        )

    def start_warmup(
        self,
        file_paths: list[str],
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float,
    ) -> None:
        geom_key = (lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
        with self._warm_lock:
            alive = self._warm_thread is not None and self._warm_thread.is_alive()
            if self._warm_complete_key == geom_key or (self._warm_geom_key == geom_key and alive):
                return
            if self._warm_stop is not None:
                self._warm_stop.set()
                # V24 Fix: Ne asiguram ca thread-ul vechi e inchis curat inainte sa deschidem altul
                if self._warm_thread and self._warm_thread.is_alive():
                    self._warm_thread.join(timeout=1.0)
            stop = threading.Event()
            geom_args = (lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
            self._warm_stop = stop
            self._warm_geom_key = geom_key
            self._warm_total = len(file_paths)
            self._warm_thread = threading.Thread(
                target=self._warmup_loop,
                args=(list(file_paths), geom_args, geom_key, stop),
                daemon=True,
            )
            self._warm_thread.start()

    def stop_warmup(self) -> None:
        with self._warm_lock:
            if self._warm_stop is not None:
                self._warm_stop.set()
            self._warm_thread = None
            self._warm_geom_key = None
            self._warm_complete_key = None
            self._warm_total = 0

    def warm_status(self) -> tuple[int, int]:
        total = self._warm_total
        if total <= 0:
            return 0, 0
        return min(len(self._prep_cache), total), total

    def _warmup_loop(self, file_paths: list[str], geom_args: tuple, geom_key: tuple,
                     stop: threading.Event) -> None:
        i = 0
        while i < len(file_paths):
            if stop.is_set():
                return
            if time.monotonic() - self._last_activity < _WARMUP_GRACE_S:
                time.sleep(_WARMUP_POLL_S)
                continue
            if not self._lock.acquire(blocking=False):
                time.sleep(_WARMUP_POLL_S)
                continue
            try:
                if not self._warm_one(file_paths[i], geom_key, geom_args):
                    return
            finally:
                self._lock.release()
            i += 1

        with self._warm_lock:
            if not stop.is_set():
                self._warm_complete_key = geom_key

    def _warm_one(self, file_path: str, geom_key: tuple, geom_args: tuple) -> bool:
        if geom_key != self._geom_key:
            return False
        if file_path not in self._prep_cache:
            self._compute_prep(file_path, *geom_args)
        return True
