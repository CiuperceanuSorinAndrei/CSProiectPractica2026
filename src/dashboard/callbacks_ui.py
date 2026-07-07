"""UI toggle callbacks for the NowcastingDashboard."""
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc

from src.dashboard.constants import MANUAL_LOCATION


class UICallbacks:
    """Registers pure UI state toggles (no data processing)."""

    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.app = dashboard.app

    def register(self):
        app = self.app

        app.callback(
            Output("predefined-loc-div", "style"),
            Output("reservoir-loc-div", "style"),
            Output("manual-coords-div", "style"),
            Output("radius-input-div", "style"),
            Input("location-type", "value"),
            Input("location-select", "value"),
        )(self._toggle_location_inputs)

        app.callback(
            Output("animation-interval", "disabled"),
            Input("btn-play", "n_clicks"),
            State("animation-interval", "disabled"),
            prevent_initial_call=True,
        )(self._toggle_play)

        app.callback(
            Output("live-polling-interval", "disabled"),
            Output("btn-play", "disabled"),
            Output("btn-reset", "disabled"),
            Output("frame-slider", "disabled"),
            Output("start-date", "disabled"),
            Output("end-date", "disabled"),
            Output("start-hour", "disabled"),
            Output("end-hour", "disabled"),
            Output("btn-download", "disabled"),
            Input("run-mode-select", "value"),
        )(self._toggle_live_mode)

        app.callback(
            Output("historic-controls-container", "style"),
            Output("playback-controls-container", "style"),
            Input("run-mode-select", "value")
        )(self._toggle_ui_elements)

        app.callback(
            Output("server-config-collapse", "is_open"),
            Output("toggle-server-config", "children"),
            Input("toggle-server-config", "n_clicks"),
            State("server-config-collapse", "is_open"),
            prevent_initial_call=True,
        )(self._toggle_server_config)

    @staticmethod
    def _toggle_ui_elements(run_mode):
        if run_mode == "live":
            return {"display": "none"}, {"display": "none"}
        return {"display": "block"}, {"display": "block"}

    @staticmethod
    def _toggle_location_inputs(loc_type, loc_select):
        show_predefined = {"display": "block"} if loc_type == "predefined" else {"display": "none"}
        show_reservoir = {"display": "block"} if loc_type == "reservoir" else {"display": "none"}
        show_manual = {"display": "block"} if (loc_type == "predefined" and loc_select == MANUAL_LOCATION) else {"display": "none"}
        show_radius = show_predefined
        return show_predefined, show_reservoir, show_manual, show_radius

    @staticmethod
    def _toggle_play(n_clicks, currently_disabled):
        return not currently_disabled

    @staticmethod
    def _toggle_live_mode(mode):
        is_live = (mode == "live")
        return not is_live, is_live, is_live, False, is_live, is_live, is_live, is_live, is_live

    @staticmethod
    def _toggle_server_config(n, is_open):
        new_open = not is_open
        return new_open, ("▾ Server Configuration" if new_open else "▸ Server Configuration")
