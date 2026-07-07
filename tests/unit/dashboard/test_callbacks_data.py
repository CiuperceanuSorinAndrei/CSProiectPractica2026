from datetime import date

from src.dashboard.callbacks_data import DataCallbacks


class FakeDataService:
    def __init__(self, result):
        self.result = result

    def download_range(self, _start, _end):
        return self.result


class FakeStore:
    def filtered(self, _time_range, run_mode="historic"):
        return ["frame.nc"] if run_mode == "historic" else []


class FakeDashboard:
    def __init__(self, result):
        self.app = None
        self._data_service = FakeDataService(result)
        self._store = FakeStore()


def _callback(result):
    callbacks = DataCallbacks(FakeDashboard(result))
    callbacks._apply_settings = lambda *args, **kwargs: None
    return callbacks


def test_download_historic_reports_already_available_for_zero_tuple():
    msg, max_frame, frame, time_range = _callback((0, 0))._download_historic(
        1, date(2026, 7, 7).isoformat(), date(2026, 7, 7).isoformat(), 0, 1,
        "", "", "", "", "", "", 15,
    )

    assert msg == "Data already available locally. Ready!"
    assert max_frame == 0
    assert frame == 0
    assert time_range["start"].startswith("2026-07-07T00:00:00")


def test_download_historic_reports_partial_downloads():
    msg, max_frame, frame, time_range = _callback((3, 1))._download_historic(
        1, date(2026, 7, 7).isoformat(), date(2026, 7, 7).isoformat(), 0, 1,
        "", "", "", "", "", "", 15,
    )

    assert msg == "Downloaded 1/3 missing files. Check connection or server availability."
    assert max_frame == 0
    assert frame == 0
    assert time_range["end"].startswith("2026-07-07T01:59:59")
