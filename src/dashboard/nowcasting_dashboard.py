"""Aplicatia Dash: detine orchestrator-ul, serviciul de date, starea si callbacks."""
import os
from datetime import datetime as dt

import matplotlib
matplotlib.use('Agg')

import dash
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from src.io.cloud_data_service import CloudDataService
from src.io.server_settings import ServerSettings
from src.ui_helpers.plotting import StormMapPlotter
from config import PREDEFINED_LOCATIONS, BASE_DIR

from src.dashboard.constants import MANUAL_LOCATION, DEFAULT_TIME_RANGE
from src.dashboard.frame_store import FrameStore
from src.dashboard.dashboard_layout import DashboardLayout
from src.dashboard.session_manager import SessionManager
from src.dashboard.report_builder import ReportBuilder

from src.dashboard.dashboard_callbacks import DashboardCallbacks

class NowcastingDashboard:
    """Aplicatia Dash simplificata."""

    def __init__(self):
        self._settings = ServerSettings.load()
        self._store = FrameStore(self._settings.local_dir, self._settings.file_format)
        self._data_service = CloudDataService(self._settings)
        self._session_manager = SessionManager()

        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            assets_folder=os.path.join(BASE_DIR, "assets"),
            # Nu inlocui titlul cu "Updating..." la fiecare callback (ex. poll-ul de progres),
            # altfel titlul paginii palpaie continuu.
            update_title=None,
        )
        self.app.title = "Estimarea volumului de precipitații"
        
        def serve_layout():
            return DashboardLayout(self._store).build()
            
        self.app.layout = serve_layout
        
        # Inregistram callback-urile din modulul separat
        DashboardCallbacks(self).register()

    @property
    def server(self):
        return self.app.server

    def run(self, debug: bool = True, port: int = 8050) -> None:
        print("Porneste serverul Dash... Deschide http://127.0.0.1:8050 in browser!")
        # Dezactivam use_reloader pentru ca descarcarea de date locale sa nu cauzeze restartarea serverului in timpul descarcarii!
        self.app.run(debug=debug, port=port, use_reloader=False)
