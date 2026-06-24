"""Aplicatia Dash: detine orchestrator-ul, serviciul de date, starea si callbacks."""
import os
import io
import base64
from datetime import datetime as dt

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactiv pentru viteza si stabilitate in server
import matplotlib.pyplot as plt

import dash
from dash import html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from orchestrator import Orchestrator
from src.io.cloud_data_service import CloudDataService
from src.ui_helpers.plotting import StormMapPlotter
from config import PREDEFINED_LOCATIONS, RAIN_VMAX, RAIN_THRESHOLD_MIN, BASE_DIR

from src.dashboard.constants import (
    DATA_DIR, MANUAL_LOCATION,
    MAP_ZOOM_MIN, MAP_ZOOM_MAX, MAP_ZOOM_DEFAULT,
    ROI_RADIUS_MIN, ROI_RADIUS_MAX, ROI_RADIUS_DEFAULT,
)
from src.dashboard.frame_store import FrameStore
from src.dashboard.frame_history import FrameHistory
from src.dashboard.dashboard_layout import DashboardLayout


class NowcastingDashboard:
    """Aplicatia Dash: detine orchestrator-ul, serviciul de date, starea si callbacks."""

    def __init__(self):
        self._store = FrameStore(DATA_DIR)
        self._history = FrameHistory()
        self._orch = Orchestrator()
        self._data_service = CloudDataService()

        # assets_folder pointat explicit la folderul assets din radacina proiectului
        # (aplicatia traieste in src/dashboard/, deci default-ul Dash nu l-ar gasi).
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            assets_folder=os.path.join(BASE_DIR, "assets"),
        )
        self.app.title = "Estimarea volumului de precipitatii"
        self.app.layout = DashboardLayout(self._store).build()
        self._register_callbacks()

    @property
    def server(self):
        return self.app.server

    def run(self, debug: bool = True, port: int = 8050) -> None:
        print("Pornește serverul Dash... Deschide http://127.0.0.1:8050 în browser!")
        self.app.run(debug=debug, port=port)

    # ---- callback registration --------------------------------------------
    def _register_callbacks(self) -> None:
        app = self.app

        app.callback(
            Output("manual-coords-div", "style"),
            Input("location-select", "value"),
        )(self._toggle_manual_coords)

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
            prevent_initial_call=True,
        )(self._poll_live_data)

        app.callback(
            Output("download-status", "children"),
            Output("frame-slider", "max", allow_duplicate=True),
            Output("frame-slider", "value", allow_duplicate=True),
            Output("active-time-range", "data"),
            Input("btn-download", "n_clicks"),
            State("start-date", "value"),
            State("end-date", "value"),
            State("start-hour", "value"),
            State("end-hour", "value"),
            prevent_initial_call=True,
        )(self._download_historic)

        app.callback(
            Output("map-image", "src"),
            Output("val-historic-vol", "children"),
            Output("val-current-vol", "children"),
            Output("val-predicted-vol", "children"),
            Output("val-max-rain", "children"),
            Output("val-metrics", "children"),
            Output("val-tracked", "children"),
            Output("val-in-roi", "children"),
            Output("frame-label", "children"),
            Output("final-report-div", "children"),
            Output("diagnostics-div", "children"),
            Output("is-processing", "data"),
            Output("input-warnings", "children"),
            Output("map-zoom-slider", "value"),
            Output("roi-radius-slider", "value"),
            Input("frame-slider", "value"),
            Input("location-select", "value"),
            Input("manual-lat", "value"),
            Input("manual-lon", "value"),
            Input("map-zoom-slider", "value"),
            Input("roi-radius-slider", "value"),
            State("run-mode-select", "value"),
            State("active-time-range", "data"),
        )(self._update_dashboard)

        app.callback(
            Output("frame-slider", "value", allow_duplicate=True),
            Input("btn-reset", "n_clicks"),
            prevent_initial_call=True,
        )(self._handle_reset)

    # ---- simple callbacks --------------------------------------------------
    @staticmethod
    def _toggle_manual_coords(loc):
        return {"display": "block"} if loc == MANUAL_LOCATION else {"display": "none"}

    @staticmethod
    def _toggle_play(n_clicks, currently_disabled):
        return not currently_disabled

    @staticmethod
    def _auto_advance_frame(n, is_processing, current_frame, max_frame):
        # Daca serverul inca proceseaza un cadru, sarim peste acest tick (evita blocaje/sarituri).
        if is_processing:
            raise PreventUpdate
        if current_frame < max_frame:
            return current_frame + 1, True, dash.no_update
        # La final, oprim intervalul ca raportul sa poata fi citit
        return current_frame, False, True

    @staticmethod
    def _toggle_live_mode(mode):
        is_live = (mode == "live")
        # primul output e live-polling-interval (activ in LIVE); restul sunt controale istorice (dezactivate in LIVE)
        return not is_live, is_live, is_live, is_live, is_live, is_live, is_live, is_live, is_live

    def _poll_live_data(self, n_int, mode):
        if mode != "live":
            raise PreventUpdate
        self._data_service.fetch_latest()
        files = self._store.list()
        if not files:
            raise PreventUpdate
        max_idx = len(files) - 1
        return max_idx, max_idx

    def _handle_reset(self, n_clicks):
        if n_clicks:
            self._orch.reset_tracking()
            self._history.reset()
            return 0
        return dash.no_update

    def _download_historic(self, n, start_d, end_d, start_h, end_h):
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
            return "Timpul de Start trebuie să fie înaintea Stop!", dash.no_update, dash.no_update, dash.no_update

        new_count = self._data_service.download_range(start_dt, end_dt)
        msg = (f"✓ S-au descărcat {new_count} fișiere noi. Gata de folosire!" if new_count
               else "✓ Datele există deja local. Gata de folosire!")

        time_range = {"start": start_dt.isoformat(), "end": end_dt.isoformat()}
        filtered = self._store.filtered(time_range, run_mode="historic")
        return msg, max(len(filtered) - 1, 0), 0, time_range

    # ---- main dashboard callback ------------------------------------------
    def _update_dashboard(self, frame_idx, loc_choice, m_lat, m_lon, map_zoom, radius_km, run_mode, time_range):
        map_zoom, radius_km, warnings = self._validate_view_inputs(map_zoom, radius_km)

        nc_files = self._store.filtered(time_range, run_mode)
        if not nc_files:
            return self._no_data_response()

        frame_idx = min(max(frame_idx, 0), len(nc_files) - 1)
        label = FrameStore.label(nc_files[frame_idx])
        center = self._resolve_center(loc_choice, m_lat, m_lon)
        bbox = self._compute_bbox(center, map_zoom)

        result = self._process_to_frame(frame_idx, nc_files, bbox, center, radius_km)
        if result is None:
            return self._error_response(label)

        title = f"🔴 LIVE NOWCAST: {label} UTC" if run_mode == "live" else f"{label} UTC"
        src = self._render_map(result, bbox, center, radius_km, title)
        diagnostics = self._build_diagnostics(result.tracked_cells)
        hist_vol, curr_vol, pred_vol, max_rain, metrics, tracked, in_roi = self._format_metrics(result)
        lbl_frame = f"Cadru: {label} UTC ({frame_idx + 1}/{len(nc_files)})"
        final_report = self._build_final_report(run_mode, frame_idx, len(nc_files))

        return (src, hist_vol, curr_vol, pred_vol, max_rain, metrics, tracked, in_roi,
                lbl_frame, final_report, diagnostics, False, warnings, map_zoom, radius_km)

    # ---- update_dashboard helpers -----------------------------------------
    def _process_to_frame(self, frame_idx, nc_files, bbox, center, radius_km):
        """Proceseaza cadrul curent gestionand starea: avans consecutiv (acumuleaza),
        acelasi cadru (doar re-randare), sau salt (proceseaza cadrele omise). Returneaza
        FrameResult sau None (eroare/server ocupat)."""
        lon_min, lon_max, lat_min, lat_max = bbox
        center_lat, center_lon = center

        def run(idx):
            return self._orch.process_frame(
                self._store.path(nc_files[idx]),
                lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km,
            )

        hist = self._history
        is_consecutive = (frame_idx == hist.last_frame_idx + 1)
        is_same_frame = (frame_idx == hist.last_frame_idx)

        if is_consecutive:
            result = run(frame_idx)
            if result is None:
                raise PreventUpdate
            hist.accumulate(result)
        elif is_same_frame:
            # Zoom/rază schimbate: reprocesăm pentru vizualizare, dar NU atingem istoricul.
            result = run(frame_idx)
            if result is None:
                raise PreventUpdate
        else:
            # Salt: procesăm cadrele omise (înainte) pentru a păstra continuitatea volumului.
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

    @staticmethod
    def _validate_view_inputs(map_zoom, radius_km):
        raw_zoom, raw_radius = map_zoom, radius_km
        zoom = min(max(map_zoom, MAP_ZOOM_MIN), MAP_ZOOM_MAX) if map_zoom is not None else MAP_ZOOM_DEFAULT
        radius = min(max(radius_km, ROI_RADIUS_MIN), ROI_RADIUS_MAX) if radius_km is not None else ROI_RADIUS_DEFAULT

        warnings = []
        if raw_zoom is None:
            warnings.append(dbc.Alert("Valoare invalidă pentru Arie. S-a folosit valoarea implicită (500 km).",
                                      color="danger", style={"padding": "0.5rem"}, className="small mb-2"))
        elif raw_zoom > MAP_ZOOM_MAX or raw_zoom < MAP_ZOOM_MIN:
            warnings.append(dbc.Alert(f"Aria introdusă ({raw_zoom} km) a fost respinsă. Valoarea maximă permisă este 700 km (minim 100 km).",
                                      color="danger", style={"padding": "0.5rem"}, className="small mb-2 fw-bold"))

        if raw_radius is None:
            warnings.append(dbc.Alert("Valoare invalidă pentru Rază. S-a folosit valoarea implicită (30 km).",
                                      color="danger", style={"padding": "0.5rem"}, className="small mb-2"))
        elif raw_radius > ROI_RADIUS_MAX or raw_radius < ROI_RADIUS_MIN:
            warnings.append(dbc.Alert(f"Raza introdusă ({raw_radius} km) a fost respinsă. Valoarea maximă permisă este 200 km (minim 5 km).",
                                      color="danger", style={"padding": "0.5rem"}, className="small mb-2 fw-bold"))

        return zoom, radius, warnings

    @staticmethod
    def _resolve_center(loc_choice, m_lat, m_lon):
        if loc_choice == MANUAL_LOCATION:
            return float(m_lat), float(m_lon)
        cfg = PREDEFINED_LOCATIONS[loc_choice]
        return float(cfg["lat"]), float(cfg["lon"])

    @staticmethod
    def _compute_bbox(center, map_zoom):
        center_lat, center_lon = center
        delta_lat = map_zoom / 111.0
        delta_lon = map_zoom / (111.0 * np.cos(np.radians(center_lat)))
        return (center_lon - delta_lon, center_lon + delta_lon,
                center_lat - delta_lat, center_lat + delta_lat)

    @staticmethod
    def _render_map(result, bbox, center, radius_km, title) -> str:
        fig, ax, _ = StormMapPlotter.create_figure(
            result.lon_grid, result.lat_grid, result.rain_rate_masked,
            extent=bbox, vmin=RAIN_THRESHOLD_MIN, vmax=RAIN_VMAX, title=title,
            roi_center=center, roi_radius_km=radius_km,
        )
        StormMapPlotter.draw_overlays(ax, result.tracked_cells, result.lon_grid, result.lat_grid)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")

    @staticmethod
    def _build_diagnostics(tracked_cells) -> html.Div:
        diag_rows = []
        for cell in tracked_cells:
            vt = cell.get("volume_trend", 1.0)
            stare = "Extindere" if vt > 1.12 else "Disipare" if vt < 0.88 else "Stabil"
            err = f"{cell.get('prediction_error_pixels', 0.0):.1f} px" if cell["is_tracked"] else "Nou"
            diag_rows.append(html.Tr([
                html.Td(cell["cell_id"]),
                html.Td(f"{cell['geo_lat']:.2f}, {cell['geo_lon']:.2f}"),
                html.Td(str(cell.get("area_pixels", 0))),
                html.Td(err),
                html.Td(stare),
            ]))

        diag_table = dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("ID Sistem"), html.Th("Locație"), html.Th("Arie (px)"),
                    html.Th("Eroare Centroid"), html.Th("Evoluție"),
                ])),
                html.Tbody(diag_rows),
            ],
            bordered=True, hover=True, size="sm", className="table-dark mb-0",
        )

        return html.Div([
            html.H6("Diagnostic Corelat", className="fw-bold text-muted mb-2"),
            html.Div(diag_table, style={"maxHeight": "250px", "overflowY": "auto",
                                        "border": "1px solid var(--c-border)", "borderRadius": "5px"}),
        ], className="mb-4")

    def _format_metrics(self, result):
        hist_vol = f"{self._history.total_volume_m3 / 1000.0:.2f} mii m³"
        curr_vol = f"{result.roi_volume_m3 / 1000.0:.2f} mii m³"
        pred_vol = f"{result.predicted_roi_volume_m3 / 1000.0:.2f} mii m³"
        max_rain = f"{result.max_rain:.2f}"
        metrics = (f"{result.global_csi:.2f} / {result.global_far:.2f} / {result.global_pod:.2f}"
                   if result.global_csi is not None else "N/A (Fără ploaie)")
        tracked = f"{result.num_tracked}"
        in_roi = f"{sum(1 for c in result.tracked_cells if c.get('in_roi'))}"
        return hist_vol, curr_vol, pred_vol, max_rain, metrics, tracked, in_roi

    def _build_final_report(self, run_mode, frame_idx, n_files):
        if run_mode == "live" or frame_idx != n_files - 1:
            return ""
        avg_csi, avg_far, avg_pod = self._history.averages()
        return dbc.Alert(
            [
                html.H4("Simulare Istorică Încheiată", className="alert-heading"),
                html.P("Raport de Performanță al Algoritmului pentru intervalul analizat:"),
                html.Hr(),
                dbc.Row([
                    dbc.Col(html.Strong(f"Volum Total Precipitat (Bazin): {self._history.total_volume_m3 / 1000.0:.0f} mii m³")),
                    dbc.Col(html.Strong(f"Acuratețe Detecție (CSI): {avg_csi:.2f}")),
                    dbc.Col(html.Strong(f"Rată Alarme False (FAR): {avg_far:.2f}")),
                    dbc.Col(html.Strong(f"Probabilitate Detecție (POD): {avg_pod:.2f}")),
                ]),
            ],
            color="success",
        )

    # ---- canned responses --------------------------------------------------
    @staticmethod
    def _no_data_response():
        return ("", "0.00", "0.00", "0.00", "0.0", "N/A", "0", "0", "Fără Date", "", "",
                False, dash.no_update, dash.no_update, dash.no_update)

    @staticmethod
    def _error_response(label):
        return ("", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare",
                f"Eroare fisier {label}", "", "", False, dash.no_update, dash.no_update, dash.no_update)
