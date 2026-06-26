"""Gestionarea starii per-utilizator (Sesiuni) pentru Dashboard."""
import time
import uuid

from orchestrator import Orchestrator
from src.dashboard.frame_history import FrameHistory


class SessionManager:
    """Gestioneaza instantele de Orchestrator si FrameHistory per sesiune."""
    
    def __init__(self):
        self._orchestrators: dict[str, Orchestrator] = {}
        self._histories: dict[str, FrameHistory] = {}
        self._last_dataset_id = {}
        self._last_access: dict[str, float] = {}

    def _cleanup_old_sessions(self):
        """Previnem Memory Leak stergand sesiunile mai vechi de 1 ora."""
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
        if not session_id:
            session_id = "default"
        if session_id not in self._orchestrators:
            self._orchestrators[session_id] = Orchestrator()
            self._histories[session_id] = FrameHistory()
        return self._orchestrators[session_id], self._histories[session_id]

    def reset_session(self, session_id: str) -> None:
        if session_id in self._orchestrators:
            self._orchestrators[session_id].reset_tracking()
            self._histories[session_id].reset()

    def process_to_frame(
        self, session_id: str, frame_idx: int, nc_files: list, bbox: tuple, 
        center: tuple, radius_km: float, run_mode: str, time_range: dict, store
    ):
        """Proceseaza cadru logic (acumulare/re-randare/salt) si mentine starea."""
        lon_min, lon_max, lat_min, lat_max = bbox
        center_lat, center_lon = center

        # V24 Fix: Includem si bbox in dataset_id pentru a forta resetarea trackerului daca utilizatorul da zoom!
        dataset_id = (run_mode, str(time_range), str(bbox))
        session_dataset = self._last_dataset_id.get(session_id)
        is_new_dataset = (session_dataset != dataset_id)

        orch, hist = self.get_state(session_id)

        def run(idx):
            return orch.process_frame(
                store.path(nc_files[idx]),
                lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km,
            )

        is_consecutive = (frame_idx == hist.last_frame_idx + 1)
        is_same_frame = (frame_idx == hist.last_frame_idx)

        if is_new_dataset or frame_idx < hist.last_frame_idx:
            orch.reset_tracking()
            hist.reset()
            self._last_dataset_id[session_id] = dataset_id
            
            paths = [store.path(f) for f in nc_files]
            orch.start_warmup(paths, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
            
            for i in range(0, frame_idx):
                inter = run(i)
                if inter:
                    hist.accumulate(inter)
            result = run(frame_idx)
            if result is not None:
                hist.accumulate(result)
        elif is_consecutive:
            result = run(frame_idx)
            if result is None:
                return None
            hist.accumulate(result)
        elif is_same_frame:
            # V24 Fix: Returnam ultimul rezultat in loc sa rulam run(frame_idx) din nou, 
            # ceea ce distrugea tracker-ul (viteza 0).
            return hist.last_result
        else:
            for i in range(max(0, hist.last_frame_idx + 1), frame_idx):
                inter = run(i)
                if inter:
                    hist.accumulate(inter)
            result = run(frame_idx)
            if result is None:
                return None
            hist.accumulate(result)

        hist.last_frame_idx = frame_idx
        return result
