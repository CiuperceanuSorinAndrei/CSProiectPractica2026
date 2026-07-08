# Per-user state management (Sessions) for the Dashboard.
import time
import uuid

from src.core.orchestrator import Orchestrator
from src.dashboard.frame_history import FrameHistory


class SessionManager:
    # Manages Orchestrator and FrameHistory instances per session.
    
    def __init__(self):
        self._orchestrators: dict[str, Orchestrator] = {}
        self._histories: dict[str, FrameHistory] = {}
        self._last_dataset_id = {}
        self._last_access: dict[str, float] = {}

    def _cleanup_old_sessions(self):
        """Prevent Memory Leaks by removing sessions older than 1 hour."""
        now = time.time()
        expired = [sid for sid, t in self._last_access.items() if now - t > 3600]
        for sid in expired:
            self._orchestrators.pop(sid, None)
            self._histories.pop(sid, None)
            self._last_dataset_id.pop(sid, None)
            self._last_access.pop(sid, None)

    def get_state(self, session_id: str) -> tuple[Orchestrator, FrameHistory]:
        self._cleanup_old_sessions()
        if not session_id:
            session_id = "default"
        self._last_access[session_id] = time.time()
        if session_id not in self._orchestrators:
            self._orchestrators[session_id] = Orchestrator()
            self._histories[session_id] = FrameHistory()
        return self._orchestrators[session_id], self._histories[session_id]

    def reset_session(self, session_id: str) -> None:
        if session_id in self._orchestrators:
            self._orchestrators[session_id].reset_tracking()
            self._histories[session_id].reset()

    def process_to_frame(
        self, session_id: str, frame_idx: int, nc_files: list[str],
        bbox: tuple[float, float, float, float], center: tuple[float, float],
        radius_km: float, run_mode: str, time_range: dict, store,
        polygon=None, catchment_polygon=None, frame_time=None
    ):
        """Processes logical frame (accumulation/re-rendering/jump) and maintains state."""
        self._last_access[session_id] = time.time()
        self._cleanup_old_sessions()

        lon_min, lon_max, lat_min, lat_max = bbox
        center_lat, center_lon = center

        dataset_id = (run_mode, str(time_range), str(bbox), center, radius_km, id(polygon))
        session_dataset = self._last_dataset_id.get(session_id)
        is_new_dataset = (session_dataset != dataset_id)

        orch, hist = self.get_state(session_id)

        def run(idx, f_time=None):
            return orch.process_frame(
                store.path(nc_files[idx]),
                lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km,
                polygon=polygon, catchment_polygon=catchment_polygon,
                frame_time=f_time, run_mode=run_mode
            )

        if is_new_dataset or frame_idx < hist.last_frame_idx:
            def warmup():
                paths = [store.path(f) for f in nc_files]
                orch.start_warmup(
                    paths, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km,
                    polygon=polygon, catchment_polygon=catchment_polygon
                )

            self._last_dataset_id[session_id] = dataset_id
            result = self._replay_from_start(run, warmup, orch, hist, frame_idx)
        elif frame_idx == hist.last_frame_idx + 1:  # consecutive frame
            result = run(frame_idx, frame_time)
            if result is None:
                return None
            hist.accumulate(result)
        elif frame_idx == hist.last_frame_idx:  # same frame
            return hist.last_result
        else:  
            self._accumulate_range(run, hist, max(0, hist.last_frame_idx + 1), frame_idx)
            result = run(frame_idx, frame_time)
            if result is None:
                return None
            hist.accumulate(result)

        hist.last_frame_idx = frame_idx
        return result

    @staticmethod
    def _accumulate_range(run, hist, start: int, stop: int) -> None:
        """Runs and accumulates intermediate frames in the interval [start, stop)."""
        for i in range(start, stop):
            inter = run(i)
            if inter:
                hist.accumulate(inter)

    @staticmethod
    def _replay_from_start(run, warmup, orch, hist, frame_idx: int):
        """Full reset + re-run from frame 0 to frame_idx (new dataset or rewind)."""
        orch.reset_tracking()
        hist.reset()
        SessionManager._accumulate_range(run, hist, 0, frame_idx)
        result = run(frame_idx) # f_time not strictly needed for history replay
        if result is not None:
            hist.accumulate(result)
        warmup()
        return result
