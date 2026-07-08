import dash
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from datetime import datetime as dt
import numpy as np

from src.dashboard.constants import MANUAL_LOCATION
from src.dashboard.report_builder import ReportBuilder
from src.ui_helpers.plotting import StormMapPlotter


class RenderCallbacks:
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.app = dashboard.app

    def register(self):
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
    def _auto_advance_frame(n, is_processing, current_frame, max_frame):
        if is_processing:
            raise PreventUpdate
        if current_frame < max_frame:
            return current_frame + 1, True, dash.no_update
        return current_frame, False, True

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
        return f"Preloading cache: {done}/{total} frames"

    def _update_dashboard(self, frame_idx, loc_choice, loc_type, res_select, m_lat, m_lon, map_zoom, radius_km, run_mode, tr_data, session_id):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        if triggered_id in ("location-select", "manual-lat", "manual-lon") and loc_type != "predefined":
            raise PreventUpdate
        if triggered_id == "reservoir-select" and loc_type != "reservoir":
            raise PreventUpdate

        zoom, radius, warnings = self._validate_zoom_radius(map_zoom, radius_km)

        nc_files = self.dashboard._store.filtered(tr_data, run_mode)
        if not nc_files:
            return ("assets/placeholder.png", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                    "No data", None, None, False, warnings, zoom, radius, "")

        frame_idx = min(max(frame_idx, 0), len(nc_files) - 1)
        label = self.dashboard._store.label(nc_files[frame_idx])

        center, polygon, prediction_area = self._resolve_roi(loc_type, loc_choice, res_select, m_lat, m_lon, radius)
        map_bbox = self._compute_bbox(center, zoom)
        prediction_bbox = self._compute_bbox(center, prediction_area)

        try:
            frame_time = dt.strptime(label, "%Y%m%d_%H%M")
        except ValueError:
            frame_time = None
            
        from src.core.orchestrator import ServerBusy
        try:
            result = self.dashboard._session_manager.process_to_frame(
                session_id, frame_idx, nc_files, prediction_bbox, center, radius, run_mode, tr_data, self.dashboard._store, polygon=polygon, frame_time=frame_time
            )
        except ServerBusy:
            raise PreventUpdate

        if result is None:
            return ("assets/placeholder.png", "Error", "Error", "Error", "Error", "Error", "Error",
                    f"Processing error {label}", None, None, False, warnings, zoom, radius, "")

        title = f"[LIVE NOWCAST] {label} UTC" if run_mode == "live" else f"{label} UTC"
        src = self._render_map_png(result, map_bbox, center, radius, polygon, title)

        diagnostics = ReportBuilder.build_diagnostics(result.tracked_cells)
        reservoir = self._selected_reservoir(loc_type, res_select)
        if reservoir is not None:
            from src.geo.reservoir_level_service import with_interval_level
            reservoir = with_interval_level(reservoir, res_select, tr_data)
        
        hist_vol, curr_vol, pred_vol, max_rain, tracked, in_roi = ReportBuilder.format_metrics(
            session_id, result, self.dashboard._session_manager, reservoir=reservoir, frame_time=frame_time)
        
        lbl_frame = f"Frame: {label} UTC ({frame_idx + 1}/{len(nc_files)})"
        final_report = ReportBuilder.build_final_report(session_id, run_mode, frame_idx, len(nc_files), self.dashboard._session_manager, reservoir)

        return (src, hist_vol, curr_vol, pred_vol, max_rain,
                tracked, in_roi,
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
            warnings.append(dbc.Alert("Invalid value for Area. Defaulting to 500 km.", color="danger", className="small mb-2"))
        elif raw_zoom > MAP_ZOOM_MAX or raw_zoom < MAP_ZOOM_MIN:
            warnings.append(dbc.Alert(f"Input Area ({raw_zoom} km) was rejected.", color="danger", className="small mb-2 fw-bold"))

        if raw_radius is None:
            warnings.append(dbc.Alert("Invalid value for Radius. Defaulting to 30 km.", color="danger", className="small mb-2"))
        elif raw_radius > ROI_RADIUS_MAX or raw_radius < ROI_RADIUS_MIN:
            warnings.append(dbc.Alert(f"Input Radius ({raw_radius} km) was rejected.", color="danger", className="small mb-2 fw-bold"))

        return zoom, radius, warnings

    @staticmethod
    def _selected_reservoir(loc_type, res_select):
        if loc_type != "reservoir":
            return None
        from src.geo.reservoir_loader import ReservoirLoader
        return ReservoirLoader.get_all_reservoirs().get(res_select)

    @staticmethod
    def _resolve_roi(loc_type, loc_choice, res_select, m_lat, m_lon, radius):
        from src.geo.reservoir_loader import ReservoirLoader
        from src.config import PREDEFINED_LOCATIONS

        polygon = None
        roi_extent_km = radius
        if loc_type == "reservoir":
            reservoirs = ReservoirLoader.get_all_reservoirs()
            if res_select in reservoirs:
                res_data = reservoirs[res_select]
                center = res_data["center"]
                polygon = res_data["polygon"]
                roi_extent_km = res_data["radius_km"]
            else:
                center = (45.0, 25.0)
        else:
            if loc_choice == MANUAL_LOCATION:
                center = (float(m_lat), float(m_lon))
            else:
                cfg = PREDEFINED_LOCATIONS[loc_choice]
                center = (float(cfg["lat"]), float(cfg["lon"]))

        prediction_area = RenderCallbacks._prediction_area_from_roi(roi_extent_km)
        return center, polygon, prediction_area

    @staticmethod
    def _prediction_area_from_roi(roi_extent_km):
        from src.dashboard.constants import MAP_ZOOM_MAX
        return min(max(float(roi_extent_km) * 2.5, 300.0), float(MAP_ZOOM_MAX))

    @staticmethod
    def _compute_bbox(center, zoom):
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
        fig.savefig(buf, format="png", dpi=100, facecolor='#212529')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)
        return f"data:image/png;base64,{encoded}"
