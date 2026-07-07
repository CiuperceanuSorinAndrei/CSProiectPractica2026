"""Data acquisition callbacks (FTP polling, historic downloads)."""
from datetime import datetime as dt
import dash
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from src.io.server_settings import ServerSettings


class DataCallbacks:
    """Registers data ingestion callbacks (live polling, historic downloads)."""

    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.app = dashboard.app

    def register(self):
        app = self.app

        app.callback(
            Output("live-status", "children"),
            Output("frame-slider", "max", allow_duplicate=True),
            Output("frame-slider", "value", allow_duplicate=True),
            Input("live-polling-interval", "n_intervals"),
            Input("run-mode-select", "value"),
            Input("btn-apply-config", "n_clicks"),
            State("frame-slider", "value"),
            State("frame-slider", "max"),
            State("srv-host", "value"),
            State("srv-remote-dir", "value"),
            State("srv-local-dir", "value"),
            State("srv-user", "value"),
            State("srv-pass", "value"),
            State("srv-format", "value"),
            State("time-delta", "value"),
            prevent_initial_call=True,
        )(self._poll_live_data)

        app.callback(
            Output("download-status", "children"),
            Output("frame-slider", "max", allow_duplicate=True),
            Output("frame-slider", "value", allow_duplicate=True),
            Output("active-time-range", "data"),
            Input("btn-download", "n_clicks"),
            State("start-date", "date"),
            State("end-date", "date"),
            State("start-hour", "value"),
            State("end-hour", "value"),
            State("srv-host", "value"),
            State("srv-remote-dir", "value"),
            State("srv-local-dir", "value"),
            State("srv-user", "value"),
            State("srv-pass", "value"),
            State("srv-format", "value"),
            State("time-delta", "value"),
            prevent_initial_call=True,
        )(self._download_historic)

    def _apply_settings(self, host, remote_dir, local_dir, username, password, file_format, time_delta):
        s = ServerSettings.from_inputs(host, remote_dir, local_dir, file_format, time_delta, username, password)
        if s == self.dashboard._settings:
            return
        s.save()
        self.dashboard._settings = s
        self.dashboard._data_service.reconfigure(s)
        self.dashboard._store.reconfigure(s.local_dir, s.file_format)

    def _poll_live_data(self, n_int, mode, btn_apply, current_val, current_max,
                        host, remote_dir, local_dir, username, password, file_format, time_delta):
        if mode != "live":
            raise PreventUpdate
        
        try:
            self._apply_settings(host, remote_dir, local_dir, username, password, file_format, time_delta)
            self.dashboard._data_service.fetch_latest()
        except Exception:
            error_msg = dbc.Alert("Connection error. Check server settings and credentials.", color="danger", className="py-2 mb-0")
            return error_msg, dash.no_update, dash.no_update
            
        files = self.dashboard._store.filtered(time_range=None, run_mode="live")
        if not files:
            raise PreventUpdate
        
        new_max = len(files) - 1
        
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        if triggered_id in ("run-mode-select", "btn-apply-config"):
            return "", new_max, new_max
        
        if current_val is not None and current_max is not None and current_val < current_max:
            return "", new_max, dash.no_update
        return "", new_max, new_max

    def _download_historic(self, n, start_d, end_d, start_h, end_h,
                           host, remote_dir, local_dir, username, password, file_format, time_delta):
        self._apply_settings(host, remote_dir, local_dir, username, password, file_format, time_delta)
        if not start_d or not end_d:
            return "Select dates!", dash.no_update, dash.no_update, dash.no_update
        if start_h is None or end_h is None:
            return "Set a valid hour (0-23)!", dash.no_update, dash.no_update, dash.no_update

        try:
            h_s, h_e = int(start_h), int(end_h)
            if not (0 <= h_s <= 23) or not (0 <= h_e <= 23):
                return "Hours must be between 0 and 23!", dash.no_update, dash.no_update, dash.no_update
            start_dt = dt.fromisoformat(start_d).replace(hour=h_s, minute=0, second=0)
            end_dt = dt.fromisoformat(end_d).replace(hour=h_e, minute=59, second=59)
        except Exception:
            return "Invalid date/time format!", dash.no_update, dash.no_update, dash.no_update

        if start_dt >= end_dt:
            return "Start time must precede Stop time!", dash.no_update, dash.no_update, dash.no_update

        try:
            missing_count, downloaded_count = self.dashboard._data_service.download_range(start_dt, end_dt)
            if missing_count == 0:
                msg = "Data already available locally. Ready!"
            elif downloaded_count == missing_count:
                msg = f"Downloaded {downloaded_count} new files. Ready!"
            else:
                msg = f"Downloaded {downloaded_count}/{missing_count} missing files. Check connection or server availability."
        except Exception:
            error_msg = dbc.Alert("Connection error. Check server settings and credentials.", color="danger", className="py-2 mb-0")
            return error_msg, dash.no_update, dash.no_update, dash.no_update

        time_range = {"start": start_dt.isoformat(), "end": end_dt.isoformat()}
        filtered = self.dashboard._store.filtered(time_range, run_mode="historic")
        return msg, max(len(filtered) - 1, 0), 0, time_range
