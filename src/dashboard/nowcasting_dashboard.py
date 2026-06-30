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
        self._register_callbacks()

    @property
    def server(self):
        return self.app.server

    def run(self, debug: bool = True, port: int = 8050) -> None:
        print("Porneste serverul Dash... Deschide http://127.0.0.1:8050 in browser!")
        self.app.run(debug=debug, port=port)

    # ---- callbacks --------------------------------------------------------
    def _toggle_ui_elements(self, run_mode):
        if run_mode == "live":
            return {"display": "none"}, {"display": "none"}
        return {"display": "block"}, {"display": "block"}

    def _register_callbacks(self) -> None:
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
            Output("frame-slider", "value"),
            Output("is-processing", "data", allow_duplicate=True),
            Output("animation-interval", "disabled", allow_duplicate=True),
            Input("animation-interval", "n_intervals"),
            State("is-processing", "data"),
            State("frame-slider", "value"),
            State("frame-slider", "max"),
            prevent_initial_call=True,
        )(self._auto_advance_frame)

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
            Output("frame-slider", "max", allow_duplicate=True),
            Output("frame-slider", "value", allow_duplicate=True),
            Input("live-polling-interval", "n_intervals"),
            Input("run-mode-select", "value"),
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
            Output("historic-controls-container", "style"),
            Output("playback-controls-container", "style"),
            Input("run-mode-select", "value")
        )(self._toggle_ui_elements)

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

        app.callback(
            Output("map-image", "src"),
            Output("val-historic-vol", "children"),
            Output("val-current-vol", "children"),
            Output("val-predicted-vol", "children"),
            Output("val-max-rain", "children"),
            Output("val-metrics-30m", "children"),
            Output("val-metrics-1h", "children"),
            Output("val-metrics-2h", "children"),
            Output("val-metrics-total", "children"),
            Output("val-tracked", "children"),
            Output("val-in-roi", "children"),
            Output("frame-label", "children"),
            Output("final-report-div", "children"),
            Output("diagnostics-div", "children"),
            Output("is-processing", "data"),
            Output("input-warnings", "children"),
            Output("map-zoom-input", "value"),
            Output("roi-radius-input", "value"),
            Output("img-loading-sentinel", "children"),  # tine spinner-ul aprins cat ruleaza callback-ul
            Input("frame-slider", "value"),
            Input("location-select", "value"),
            Input("location-type", "value"),
            Input("reservoir-select", "value"),
            Input("manual-lat", "value"),
            Input("manual-lon", "value"),
            Input("map-zoom-input", "value"),
            Input("roi-radius-input", "value"),
            State("run-mode-select", "value"),
            State("active-time-range", "data"),
            State("session-id", "data"),
        )(self._update_dashboard)

        app.callback(
            Output("frame-slider", "value", allow_duplicate=True),
            Input("btn-reset", "n_clicks"),
            State("session-id", "data"),
            prevent_initial_call=True,
        )(self._handle_reset)

        app.callback(
            Output("warmup-status", "children"),
            Input("warmup-poll", "n_intervals"),
            State("session-id", "data"),
        )(self._update_warmup_status)

        app.callback(
            Output("server-config-collapse", "is_open"),
            Output("toggle-server-config", "children"),
            Input("toggle-server-config", "n_clicks"),
            State("server-config-collapse", "is_open"),
            prevent_initial_call=True,
        )(self._toggle_server_config)

    # ---- simple callbacks --------------------------------------------------
    @staticmethod
    def _toggle_location_inputs(loc_type, loc_select):
        show_predefined = {"display": "block"} if loc_type == "predefined" else {"display": "none"}
        show_reservoir = {"display": "block"} if loc_type == "reservoir" else {"display": "none"}
        show_manual = {"display": "block"} if (loc_type == "predefined" and loc_select == MANUAL_LOCATION) else {"display": "none"}
        # Ascundem raza cand e poligon
        show_radius = show_predefined
        return show_predefined, show_reservoir, show_manual, show_radius

    @staticmethod
    def _toggle_play(n_clicks, currently_disabled):
        return not currently_disabled

    @staticmethod
    def _auto_advance_frame(n, is_processing, current_frame, max_frame):
        if is_processing:
            raise PreventUpdate
        if current_frame < max_frame:
            return current_frame + 1, True, dash.no_update
        return current_frame, False, True

    @staticmethod
    def _toggle_live_mode(mode):
        is_live = (mode == "live")
        return not is_live, is_live, is_live, False, is_live, is_live, is_live, is_live, is_live

    def _poll_live_data(self, n_int, mode, current_val, current_max,
                        host, remote_dir, local_dir, username, password, file_format, time_delta):
        if mode != "live":
            raise PreventUpdate
        self._apply_settings(host, remote_dir, local_dir, username, password, file_format, time_delta)
        self._data_service.fetch_latest()
        files = self._store.filtered(time_range=None, run_mode="live")
        if not files:
            raise PreventUpdate
        
        new_max = len(files) - 1
        
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        if triggered_id == "run-mode-select":
            return new_max, new_max
        
        if current_val is not None and current_max is not None and current_val < current_max:
            return new_max, dash.no_update
        return new_max, new_max

    def _handle_reset(self, n_clicks, session_id):
        if n_clicks:
            self._session_manager.reset_session(session_id)
            return 0
        return dash.no_update

    def _update_warmup_status(self, _n, session_id):
        """Progresul pre-incarcarii in fundal (warm-up); gol cand nu ruleaza / s-a terminat."""
        orch, _ = self._session_manager.get_state(session_id)
        done, total = orch.warm_status()
        if total <= 0 or done >= total:
            return ""
        return f"⏳ Pre-încărcare cache: {done}/{total} cadre"

    @staticmethod
    def _toggle_server_config(n, is_open):
        """Arata/ascunde blocul de configurare server + actualizeaza chevronul butonului."""
        new_open = not is_open
        return new_open, ("▾ Configurare Server" if new_open else "▸ Configurare Server")

    def _apply_settings(self, host, remote_dir, local_dir, username, password, file_format, time_delta) -> None:
        """Aplica setarile de server din UI: reconfigureaza serviciul de date + store-ul si
        persista pe disc campurile non-credentiale (NU user/parola). No-op daca nimic nu s-a schimbat."""
        s = ServerSettings.from_inputs(host, remote_dir, local_dir, file_format, time_delta, username, password)
        if s == self._settings:
            return
        s.save()  # persista host / directoare / format / interval (NU user/parola)
        self._settings = s
        self._data_service.reconfigure(s)
        self._store = FrameStore(s.local_dir, s.file_format)

    def _download_historic(self, n, start_d, end_d, start_h, end_h,
                           host, remote_dir, local_dir, username, password, file_format, time_delta):
        self._apply_settings(host, remote_dir, local_dir, username, password, file_format, time_delta)
        if not start_d or not end_d:
            return "Selectează datele!", dash.no_update, dash.no_update, dash.no_update
        if start_h is None or end_h is None:
            return "Setați o oră validă (0-23)!", dash.no_update, dash.no_update, dash.no_update

        try:
            h_s, h_e = int(start_h), int(end_h)
            if not (0 <= h_s <= 23) or not (0 <= h_e <= 23):
                return "Orele trebuie să fie între 0 și 23!", dash.no_update, dash.no_update, dash.no_update
            start_dt = dt.fromisoformat(start_d).replace(hour=h_s)
            end_dt = dt.fromisoformat(end_d).replace(hour=h_e)
        except Exception:
            return "Format dată/oră invalid!", dash.no_update, dash.no_update, dash.no_update

        if start_dt >= end_dt:
            return "Timpul de Start trebuie să preceadă timpul de Stop!", dash.no_update, dash.no_update, dash.no_update

        new_count = self._data_service.download_range(start_dt, end_dt)
        msg = (f"✓ S-au descărcat {new_count} fișiere noi. Gata de folosire!" if new_count
               else "✓ Datele există deja local. Gata de folosire!")

        time_range = {"start": start_dt.isoformat(), "end": end_dt.isoformat()}
        filtered = self._store.filtered(time_range, run_mode="historic")
        return msg, max(len(filtered) - 1, 0), 0, time_range

    # ---- main dashboard callback ------------------------------------------
    def _update_dashboard(
        self, frame_idx, loc_choice, loc_type, res_select, m_lat, m_lon, map_zoom, radius_km, run_mode, tr_data, session_id
    ):
        import numpy as np
        from dash.exceptions import PreventUpdate
        from src.geo.reservoir_loader import ReservoirLoader
        from src.dashboard.constants import MAP_ZOOM_MIN, MAP_ZOOM_MAX, MAP_ZOOM_DEFAULT, ROI_RADIUS_MIN, ROI_RADIUS_MAX, ROI_RADIUS_DEFAULT
        
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        # Caching Inteligent: Evitam recompilarea hartii cand userul modifica input-uri inactive
        if triggered_id in ("location-select", "manual-lat", "manual-lon") and loc_type != "predefined":
            raise PreventUpdate
        if triggered_id == "reservoir-select" and loc_type != "reservoir":
            raise PreventUpdate

        raw_zoom, raw_radius = map_zoom, radius_km
        zoom = min(max(map_zoom, MAP_ZOOM_MIN), MAP_ZOOM_MAX) if map_zoom is not None else MAP_ZOOM_DEFAULT
        radius = min(max(radius_km, ROI_RADIUS_MIN), ROI_RADIUS_MAX) if radius_km is not None else ROI_RADIUS_DEFAULT

        warnings = []
        if raw_zoom is None:
            warnings.append(dbc.Alert("Valoare invalidă pentru Arie. S-a folosit valoarea implicită (500 km).", color="danger", className="small mb-2"))
        elif raw_zoom > MAP_ZOOM_MAX or raw_zoom < MAP_ZOOM_MIN:
            warnings.append(dbc.Alert(f"Aria introdusă ({raw_zoom} km) a fost respinsă.", color="danger", className="small mb-2 fw-bold"))

        if raw_radius is None:
            warnings.append(dbc.Alert("Valoare invalidă pentru Rază. S-a folosit valoarea implicită (30 km).", color="danger", className="small mb-2"))
        elif raw_radius > ROI_RADIUS_MAX or raw_radius < ROI_RADIUS_MIN:
            warnings.append(dbc.Alert(f"Raza introdusă ({raw_radius} km) a fost respinsă.", color="danger", className="small mb-2 fw-bold"))

        nc_files = self._store.filtered(tr_data, run_mode)
        if not nc_files:
            return ("assets/placeholder.png", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                    "Fără date", None, None, False, warnings, zoom, radius, "")

        frame_idx = min(max(frame_idx, 0), len(nc_files) - 1)
        label = self._store.label(nc_files[frame_idx])
        
        polygon = None
        if loc_type == "reservoir":
            reservoirs = ReservoirLoader.get_all_reservoirs()
            if res_select in reservoirs:
                res_data = reservoirs[res_select]
                center = res_data["center"]
                polygon = res_data["polygon"]
                # Ajustam zoom-ul hartii automat pentru a prinde tot lacul daca e prea mare
                zoom = max(zoom, res_data["radius_km"] * 2.5)
            else:
                center = (45.0, 25.0)
        else:
            if loc_choice == MANUAL_LOCATION:
                center = (float(m_lat), float(m_lon))
            else:
                cfg = PREDEFINED_LOCATIONS[loc_choice]
                center = (float(cfg["lat"]), float(cfg["lon"]))
            
        center_lat, center_lon = center
        delta_lat = zoom / 111.0
        delta_lon = zoom / (111.0 * np.cos(np.radians(center_lat)))
        bbox = (center_lon - delta_lon, center_lon + delta_lon, center_lat - delta_lat, center_lat + delta_lat)

        from orchestrator import ServerBusy
        try:
            result = self._session_manager.process_to_frame(
                session_id, frame_idx, nc_files, bbox, center, radius, run_mode, tr_data, self._store, polygon=polygon
            )
        except ServerBusy:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        
        if result is None:
            return ("assets/placeholder.png", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare",
                    f"Eroare procesare {label}", None, None, False, warnings, zoom, radius, "")

        title = f"[LIVE NOWCAST] {label} UTC" if run_mode == "live" else f"{label} UTC"
        
        # Plot map
        fig, ax, _ = StormMapPlotter.create_figure(
            lon_grid=result.lon_grid,
            lat_grid=result.lat_grid,
            rain_rate_masked=result.rain_rate_masked,
            extent=bbox,
            title=title,
            roi_center=center,
            roi_radius_km=radius,
            polygon=polygon
        )
        StormMapPlotter.draw_overlays(
            ax=ax,
            tracked_cells=result.tracked_cells,
            lon_grid=result.lon_grid,
            lat_grid=result.lat_grid
        )
        import io
        import base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor='#212529')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        src = f"data:image/png;base64,{encoded}"
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Reports
        diagnostics = ReportBuilder.build_diagnostics(result.tracked_cells)
        hist_vol, curr_vol, pred_vol, max_rain, m_30m, m_1h, m_2h, m_tot, tracked, in_roi = ReportBuilder.format_metrics(session_id, result, self._session_manager)
        lbl_frame = f"Cadru: {label} UTC ({frame_idx + 1}/{len(nc_files)})"
        final_report = ReportBuilder.build_final_report(session_id, run_mode, frame_idx, len(nc_files), self._session_manager)

        return (src, hist_vol, curr_vol, pred_vol, max_rain,
                m_30m, m_1h, m_2h, m_tot, tracked, in_roi,
                lbl_frame, final_report, diagnostics, False, warnings, zoom, radius, "")
