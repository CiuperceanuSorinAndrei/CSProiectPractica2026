# Dash App: orchestrator, data service, state, callbacks
import os
from datetime import datetime as dt

import dash
import dash_bootstrap_components as dbc

from src.io.cloud_data_service import CloudDataService
from src.io.server_settings import ServerSettings
from src.config import BASE_DIR

from src.dashboard.constants import MANUAL_LOCATION, DEFAULT_TIME_RANGE
from src.dashboard.frame_store import FrameStore
from src.dashboard.dashboard_layout import DashboardLayout
from src.dashboard.session_manager import SessionManager
from src.dashboard.report_builder import ReportBuilder

from src.dashboard.callbacks_ui import UICallbacks
from src.dashboard.callbacks_data import DataCallbacks
from src.dashboard.callbacks_render import RenderCallbacks
class NowcastingDashboard:

    def __init__(self):
        self._settings = ServerSettings.load()
        self._store = FrameStore(self._settings.local_dir, self._settings.file_format)
        self._data_service = CloudDataService(self._settings)
        self._session_manager = SessionManager()

        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            assets_folder=os.path.join(BASE_DIR, "assets"),
            update_title=None,
            suppress_callback_exceptions=True
        )
        self.app.title = "Precipitation Volume Estimation"
        
        def serve_layout():
            return DashboardLayout(self._store).build()
            
        self.app.layout = serve_layout
        
        # Register callbacks
        UICallbacks(self).register()
        DataCallbacks(self).register()
        RenderCallbacks(self).register()
    @property
    def server(self):
        return self.app.server

    def run(self, debug: bool = True, port: int = 8050) -> None:
        print("Starting Dash server... Open http://127.0.0.1:8050 in your browser!")
        self.app.run(debug=debug, port=port, use_reloader=False)
