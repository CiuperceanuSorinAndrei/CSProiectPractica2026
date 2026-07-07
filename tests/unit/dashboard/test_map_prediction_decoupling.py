from src.dashboard.dashboard_callbacks import DashboardCallbacks
from src.dashboard.session_manager import SessionManager


class FakeOrchestrator:
    def __init__(self):
        self.reset_count = 0
        self.process_calls = []
        self.warmup_calls = []

    def reset_tracking(self):
        self.reset_count += 1

    def process_frame(self, *args, **kwargs):
        self.process_calls.append((args, kwargs))
        return {"frame": len(self.process_calls)}

    def start_warmup(self, *args, **kwargs):
        self.warmup_calls.append((args, kwargs))


class FakeHistory:
    def __init__(self):
        self.last_frame_idx = -1
        self.last_result = None
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.last_frame_idx = -1
        self.last_result = None

    def accumulate(self, result):
        self.last_result = result


class FakeStore:
    @staticmethod
    def path(name):
        return name


def test_prediction_area_is_auto_roi_not_map_zoom(monkeypatch):
    polygon = object()

    class FakeReservoirLoader:
        @staticmethod
        def get_all_reservoirs():
            return {
                "Lake": {
                    "center": (45.0, 25.0),
                    "polygon": polygon,
                    "radius_km": 40.0,
                }
            }

    monkeypatch.setattr(
        "src.geo.reservoir_loader.ReservoirLoader",
        FakeReservoirLoader,
    )

    city_center, city_polygon, city_prediction_area = DashboardCallbacks._resolve_roi(
        "predefined", "Manual (Introducere coordonate)", None, 44.0, 26.0, 30.0
    )
    reservoir_center, reservoir_polygon, reservoir_prediction_area = DashboardCallbacks._resolve_roi(
        "reservoir", None, "Lake", None, None, 30.0
    )

    assert city_center == (44.0, 26.0)
    assert city_polygon is None
    assert city_prediction_area == 300.0
    assert reservoir_center == (45.0, 25.0)
    assert reservoir_polygon is polygon
    assert reservoir_prediction_area == 300.0


def test_session_dataset_key_ignores_visual_zoom_but_tracks_roi_radius(monkeypatch):
    manager = SessionManager()
    orch = FakeOrchestrator()
    hist = FakeHistory()
    monkeypatch.setattr(manager, "get_state", lambda session_id: (orch, hist))
    monkeypatch.setattr(manager, "_cleanup_old_sessions", lambda: None)

    files = ["frame.nc"]
    bbox = (20.0, 30.0, 40.0, 50.0)
    center = (45.0, 25.0)
    store = FakeStore()

    first = manager.process_to_frame("session", 0, files, bbox, center, 30.0, "historic", {}, store)
    second = manager.process_to_frame("session", 0, files, bbox, center, 30.0, "historic", {}, store)
    third = manager.process_to_frame("session", 0, files, bbox, center, 60.0, "historic", {}, store)

    assert first == {"frame": 1}
    assert second == first
    assert third == {"frame": 2}
    assert orch.reset_count == 2
    assert hist.reset_count == 2
