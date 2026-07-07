"""Modul care incapsuleaza logica de callbacks pentru NowcastingDashboard."""
from datetime import datetime as dt
import dash
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from src.dashboard.constants import MANUAL_LOCATION
from src.io.server_settings import ServerSettings
from src.dashboard.report_builder import ReportBuilder
from src.ui_helpers.plotting import StormMapPlotter


class DashboardCallbacks:
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.app = dashboard.app
        # Shortcut referințe la dependențele dashboard-ului
        self.store = dashboard._store
        self.data_service = dashboard._data_service
        self.session_manager = dashboard._session_manager
        self.settings = dashboard._settings

    def register(self):
        self._register_ui_callbacks()
        self._register_data_callbacks()
        self._register_render_callbacks()

    def _register_ui_callbacks(self):
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

    def _register_data_callbacks(self):
        app = self.app

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

    def _register_render_callbacks(self):
        app = self.app

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
            Output("img-loading-sentinel", "children"),
            Input("frame-slider", "value"),
            Input("location-select", "value"),
            Input("location-type", "value"),
            Input("reservoir-select", "value"),
            Input("manual-lat", "value"),
            Input("manual-lon", "value"),
            Input("map-zoom-input", "value"),
            Input("roi-radius-input", "value"),
            Input("evap-input", "value"),
            Input("outflow-input", "value"),
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
        self.dashboard._data_service.fetch_latest()
        files = self.dashboard._store.filtered(time_range=None, run_mode="live")
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
            self.dashboard._session_manager.reset_session(session_id)
            return 0
        return dash.no_update

    def _update_warmup_status(self, _n, session_id):
        orch, _ = self.dashboard._session_manager.get_state(session_id)
        done, total = orch.warm_status()
        if total <= 0 or done >= total:
            return ""
        return f"⏳ Pre-încărcare cache: {done}/{total} cadre"

    @staticmethod
    def _toggle_server_config(n, is_open):
        new_open = not is_open
        return new_open, ("▾ Configurare Server" if new_open else "▸ Configurare Server")

    def _apply_settings(self, host, remote_dir, local_dir, username, password, file_format, time_delta):
        s = ServerSettings.from_inputs(host, remote_dir, local_dir, file_format, time_delta, username, password)
        if s == self.dashboard._settings:
            return
        s.save()
        self.dashboard._settings = s
        self.dashboard._data_service.reconfigure(s)
        self.dashboard._store.reconfigure(s.local_dir, s.file_format)

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
            start_dt = dt.fromisoformat(start_d).replace(hour=h_s, minute=0, second=0)
            # Ora Stop e inclusiva pentru toata ora: pana la HH:45 (nu doar HH:00), altfel
            # ultimele 3 cadre din ora selectata (HH:15/HH:30/HH:45) ar fi taiate.
            end_dt = dt.fromisoformat(end_d).replace(hour=h_e, minute=59, second=59)
        except Exception:
            return "Format dată/oră invalid!", dash.no_update, dash.no_update, dash.no_update

        if start_dt >= end_dt:
            return "Timpul de Start trebuie să preceadă timpul de Stop!", dash.no_update, dash.no_update, dash.no_update

        new_count = self.dashboard._data_service.download_range(start_dt, end_dt)
        msg = (f"✓ S-au descărcat {new_count} fișiere noi. Gata de folosire!" if new_count
               else "✓ Datele există deja local. Gata de folosire!")

        time_range = {"start": start_dt.isoformat(), "end": end_dt.isoformat()}
        filtered = self.dashboard._store.filtered(time_range, run_mode="historic")
        return msg, max(len(filtered) - 1, 0), 0, time_range

    def _update_dashboard(self, frame_idx, loc_choice, loc_type, res_select, m_lat, m_lon, map_zoom, radius_km, evap_val, outflow_val, run_mode, tr_data, session_id):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        if triggered_id in ("location-select", "manual-lat", "manual-lon") and loc_type != "predefined":
            raise PreventUpdate
        if triggered_id == "reservoir-select" and loc_type != "reservoir":
            raise PreventUpdate

        zoom, radius, warnings = self._validate_zoom_radius(map_zoom, radius_km)

        nc_files = self.dashboard._store.filtered(tr_data, run_mode)
        if not nc_files:
            return ("assets/placeholder.png", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                    "Fără date", None, None, False, warnings, zoom, radius, "")

        frame_idx = min(max(frame_idx, 0), len(nc_files) - 1)
        label = self.dashboard._store.label(nc_files[frame_idx])

        center, polygon, zoom = self._resolve_roi_center(loc_type, loc_choice, res_select, m_lat, m_lon, zoom)
        bbox = self._compute_bbox(center, zoom)

        from orchestrator import ServerBusy
        try:
            result = self.dashboard._session_manager.process_to_frame(
                session_id, frame_idx, nc_files, bbox, center, radius, run_mode, tr_data, self.dashboard._store, polygon=polygon
            )
        except ServerBusy:
            raise PreventUpdate

        if result is None:
            return ("assets/placeholder.png", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare",
                    f"Eroare procesare {label}", None, None, False, warnings, zoom, radius, "")

        title = f"[LIVE NOWCAST] {label} UTC" if run_mode == "live" else f"{label} UTC"
        src = self._render_map_png(result, bbox, center, radius, polygon, title)

        diagnostics = ReportBuilder.build_diagnostics(result.tracked_cells)
        reservoir = self._selected_reservoir(loc_type, res_select)
        if reservoir is not None:
            # nivel de pornire potrivit intervalului selectat (Sentinel-2 la cerere), altfel static
            from src.geo.reservoir_level_service import with_interval_level
            reservoir = with_interval_level(reservoir, res_select, tr_data)
        hist_vol, curr_vol, pred_vol, max_rain, m_30m, m_1h, m_2h, m_tot, tracked, in_roi = ReportBuilder.format_metrics(
            session_id, result, self.dashboard._session_manager, reservoir=reservoir,
            evap_mm_day=float(evap_val or 0.0), outflow_m3s=float(outflow_val or 0.0))
        lbl_frame = f"Cadru: {label} UTC ({frame_idx + 1}/{len(nc_files)})"
        final_report = ReportBuilder.build_final_report(session_id, run_mode, frame_idx, len(nc_files), self.dashboard._session_manager)

        return (src, hist_vol, curr_vol, pred_vol, max_rain,
                m_30m, m_1h, m_2h, m_tot, tracked, in_roi,
                lbl_frame, final_report, diagnostics, False, warnings, zoom, radius, "")

    @staticmethod
    def _validate_zoom_radius(map_zoom, radius_km):
        from src.dashboard.constants import (
            MAP_ZOOM_MIN, MAP_ZOOM_MAX, MAP_ZOOM_DEFAULT,
            ROI_RADIUS_MIN, ROI_RADIUS_MAX, ROI_RADIUS_DEFAULT,
        )

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

        return zoom, radius, warnings

    @staticmethod
    def _selected_reservoir(loc_type, res_select):
        """Intrarea ReservoirLoader pentru lacul selectat (cu suprafata + volum maxim), sau None
        cand ROI-ul nu e un lac de acumulare. Dictionarul de lacuri este memoizat, deci ieftin."""
        if loc_type != "reservoir":
            return None
        from src.geo.reservoir_loader import ReservoirLoader
        return ReservoirLoader.get_covered_reservoirs().get(res_select)

    @staticmethod
    def _resolve_roi_center(loc_type, loc_choice, res_select, m_lat, m_lon, zoom):
        from src.geo.reservoir_loader import ReservoirLoader
        from config import PREDEFINED_LOCATIONS

        polygon = None
        if loc_type == "reservoir":
            reservoirs = ReservoirLoader.get_covered_reservoirs()
            if res_select in reservoirs:
                res_data = reservoirs[res_select]
                center = res_data["center"]
                polygon = res_data["polygon"]
                zoom = max(zoom, res_data["radius_km"] * 2.5)
            else:
                center = (45.0, 25.0)
        else:
            if loc_choice == MANUAL_LOCATION:
                center = (float(m_lat), float(m_lon))
            else:
                cfg = PREDEFINED_LOCATIONS[loc_choice]
                center = (float(cfg["lat"]), float(cfg["lon"]))

        return center, polygon, zoom

    @staticmethod
    def _compute_bbox(center, zoom):
        import numpy as np
        center_lat, center_lon = center
        delta_lat = zoom / 111.0
        delta_lon = zoom / (111.0 * np.cos(np.radians(center_lat)))
        return (center_lon - delta_lon, center_lon + delta_lon, center_lat - delta_lat, center_lat + delta_lat)

    @staticmethod
    def _render_map_png(result, bbox, center, radius, polygon, title):
        import io
        import base64
        import matplotlib.pyplot as plt

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
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor='#212529')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)
        return f"data:image/png;base64,{encoded}"
