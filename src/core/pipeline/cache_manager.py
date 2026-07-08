import threading
import time
from collections import OrderedDict

from src.io.frame_preprocessor import FrameGeometry, FramePrep, compute_geometry, preprocess

_PREP_CACHE_MAXSIZE = 256
_WARMUP_GRACE_S = 0.6
_WARMUP_POLL_S = 0.05

class CacheManager:
    """Manages preprocessing cache, background warmup threads, and frame geometry."""
    
    def __init__(self, execution_lock: threading.Lock):
        self._execution_lock = execution_lock
        
        self._geom_key: tuple | None = None
        self._geom: FrameGeometry | None = None
        self._prep_cache: OrderedDict[str, FramePrep] = OrderedDict()
        
        # Background warm-up.
        self._last_activity: float = 0.0
        self._warm_lock = threading.Lock()
        self._warm_thread: threading.Thread | None = None
        self._warm_stop: threading.Event | None = None
        self._warm_geom_key: tuple | None = None
        self._warm_complete_key: tuple | None = None
        self._warm_total: int = 0

    @property
    def geometry(self) -> FrameGeometry | None:
        return self._geom

    def update_activity(self):
        self._last_activity = time.monotonic()

    def get_frame_prep(
        self, file_path: str, lon_min: float, lon_max: float, lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float, polygon=None, catchment_polygon=None
    ) -> FramePrep | None:
        poly_hash = hash(str(polygon)) if polygon else 0
        catch_hash = hash(str(catchment_polygon)) if catchment_polygon else 0
        geom_key = (lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km, poly_hash, catch_hash)
        if geom_key != self._geom_key:
            self._geom_key = geom_key
            self._geom = None
            self._prep_cache.clear()

        cached = self._prep_cache.get(file_path)
        if cached is not None and self._geom is not None:
            self._prep_cache.move_to_end(file_path)
            return cached

        return self._compute_prep(
            file_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km, polygon, catchment_polygon
        )

    def _compute_prep(
        self, file_path: str, lon_min: float, lon_max: float, lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float, polygon=None, catchment_polygon=None
    ) -> FramePrep | None:
        bbox = (lon_min, lon_max, lat_min, lat_max)
        if self._geom is None:
            self._geom = compute_geometry(file_path, bbox, (center_lat, center_lon), radius_km, polygon=polygon, catchment_polygon=catchment_polygon)
            if self._geom is None:
                return None
        prep = preprocess(file_path, self._geom, bbox)
        if prep is None:
            return None
        self._prep_cache[file_path] = prep
        if len(self._prep_cache) > _PREP_CACHE_MAXSIZE:
            self._prep_cache.popitem(last=False)
        return prep

    def start_warmup(
        self, file_paths: list[str], lon_min: float, lon_max: float, lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float, polygon=None, catchment_polygon=None
    ) -> None:
        poly_hash = hash(str(polygon)) if polygon else 0
        catch_hash = hash(str(catchment_polygon)) if catchment_polygon else 0
        geom_key = (lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km, poly_hash, catch_hash)
        with self._warm_lock:
            alive = self._warm_thread is not None and self._warm_thread.is_alive()
            if self._warm_complete_key == geom_key or (self._warm_geom_key == geom_key and alive):
                return
            if self._warm_stop is not None:
                self._warm_stop.set()
                if self._warm_thread and self._warm_thread.is_alive():
                    self._warm_thread.join(timeout=1.0)
            stop = threading.Event()
            geom_args = (lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km, polygon, catchment_polygon)
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

    def _warmup_loop(self, file_paths: list[str], geom_args: tuple, geom_key: tuple, stop: threading.Event) -> None:
        i = 0
        while i < len(file_paths):
            if stop.is_set():
                return
            if time.monotonic() - self._last_activity < _WARMUP_GRACE_S:
                if stop.wait(_WARMUP_POLL_S):
                    return
                continue
            if not self._execution_lock.acquire(timeout=_WARMUP_POLL_S):
                continue
            try:
                if not self._warm_one(file_paths[i], geom_key, geom_args):
                    return
            finally:
                self._execution_lock.release()
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
