"""Builder for metrics and HTML reports in the Dashboard."""
from dash import html
import dash_bootstrap_components as dbc
from src.core.constants import HORIZON_NAMES
from src.dashboard.session_manager import SessionManager
from src.geo.reservoir_fill import ReservoirFillEstimator
from src.config import RUNOFF_COEFFICIENT, EVAP_MM_PER_DAY, RESERVOIR_OUTFLOW_M3S


class ReportBuilder:

    @staticmethod
    def _avg_metric(hist, key: str, horizon: str) -> float:
        """Average of strictly positive values for a metric (csi/far/pod/fss) at a given horizon."""
        vals = [m.get(horizon, 0) for m in hist.metrics_history[key] if m.get(horizon, 0) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def _fill_lines(value_str: str, map_mm: float, reservoir: dict | None, duration_hours: float | None = None,
                    evap_mm_day: float = EVAP_MM_PER_DAY, outflow_m3s: float = RESERVOIR_OUTFLOW_M3S):
        res = ReservoirFillEstimator.estimate(
            map_mm, reservoir, RUNOFF_COEFFICIENT,
            duration_hours=duration_hours, evap_mm_day=evap_mm_day, outflow_m3s=outflow_m3s)
        if res is None:
            return value_str

        if res.level_source == "assumed_nnr":
            parts = [f"NNR +{res.contribution_pct:.2f}% of volume"]
            if res.delta_level_m is not None:
                parts.append(f"{res.delta_level_m:+.2f} m")
        elif res.new_fill_pct <= 100.0:
            parts = [f"{res.start_fill_pct:.0f}% → {res.new_fill_pct:.0f}% of volume"]
            if res.delta_level_m is not None:
                parts.append(f"{res.delta_level_m:+.2f} m")
        elif res.overtops:
            parts = [f"{res.start_fill_pct:.0f}% → full ⚠ overflow (inflow {res.contribution_pct:.0f}% of volume)"]
        else:
            parts = [f"{res.start_fill_pct:.0f}% → full (+{res.level_after_m:.2f} m above NNR)"]

        losses = res.outflow_m3 + res.evap_m3
        if losses > 0.0 and res.level_source != "assumed_nnr":
            parts.append(f"−{losses / 1e6:.2f} mil m³ outflows")

        children = [value_str, html.Div(" · ".join(parts), className="small text-muted fw-normal mt-1")]
        src = ReportBuilder._source_label(reservoir, res)
        if src:
            children.append(html.Div(src, className="text-muted fw-normal", style={"fontSize": "0.7rem"}))
        return children

    @staticmethod
    def _source_label(reservoir: dict, res) -> str | None:
        name = {"lake": "SWOT lake", "river": "SWOT river", "s2": "Sentinel-2"}.get(reservoir.get("level_product"))
        if name:
            as_of = reservoir.get("level_as_of")
            return f"{name} · {as_of[:10]}" if as_of else name
        if res.level_source == "assumed_nnr":
            return "assumed NNR"
        return None

    @staticmethod
    def format_metrics(session_id: str, result, session_manager: SessionManager, reservoir: dict | None = None,
                       evap_mm_day: float = EVAP_MM_PER_DAY, outflow_m3s: float = RESERVOIR_OUTFLOW_M3S):
        _, hist = session_manager.get_state(session_id)

        hist_vol_str = f"{hist.total_map_mm:.2f} L/m²"

        curr_vol = result.roi_map_mm
        curr_vol_str = f"{curr_vol:.2f} L/m²"

        vols = result.predicted_volumes_horizons
        pred_vol_str = (
            f"15m: {vols.get('15m', 0):.2f} | "
            f"1h: {vols.get('1h', 0):.2f} | "
            f"2h: {vols.get('2h', 0):.2f} L/m²"
        )

        # Below the volume metrics (L/m²) we display the percentage of the selected reservoir's max volume.
        event_hours = (hist.frames_processed or 0) * 0.25
        hist_vol_str = ReportBuilder._fill_lines(hist_vol_str, hist.total_map_mm, reservoir, event_hours, evap_mm_day, outflow_m3s)
        curr_vol_str = ReportBuilder._fill_lines(curr_vol_str, curr_vol, reservoir, None, evap_mm_day, outflow_m3s)
        pred_vol_str = ReportBuilder._fill_lines(pred_vol_str, vols.get("1h", 0.0), reservoir, None, evap_mm_day, outflow_m3s)

        from dash import html
        pred_vol_str = html.Span([
            html.Span(f"{vols.get('15m', 0):.2f} "), html.Small("15m", className="text-muted me-2", style={"fontSize": "0.6em"}),
            html.Span(f"{vols.get('1h', 0):.2f} "), html.Small("1h", className="text-muted me-2", style={"fontSize": "0.6em"}),
            html.Span(f"{vols.get('2h', 0):.2f} "), html.Small("2h", className="text-muted", style={"fontSize": "0.6em"}),
        ])

        max_rain_str = f"{result.max_rain:.2f}"

        tracked = f"{result.num_tracked} Active"
        in_roi = "No"
        if curr_vol > 0:
            in_roi = "Yes (Rain detected)"

        return hist_vol_str, curr_vol_str, pred_vol_str, max_rain_str, tracked, in_roi

    @staticmethod
    def build_diagnostics(tracked_cells: list) -> html.Div:
        import math
        rows = []
        for cell in tracked_cells:
            if cell.get("is_tracked", False):
                short_id = str(cell.get('cell_id', '???'))[:4]
                
                # Calculate speed in km/h: sqrt(vx^2 + vy^2) pixels/frame * 3 km/pixel / (15/60) h = * 12
                vx = cell.get('v_x', 0)
                vy = cell.get('v_y', 0)
                speed_kmh = math.sqrt(vx**2 + vy**2) * 12
                
                # Calculate direction
                if vx == 0 and vy == 0:
                    direction = "Stationary"
                else:
                    angle = math.degrees(math.atan2(-vy, vx)) # -vy because image Y is down
                    if angle < 0: angle += 360
                    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
                    direction = dirs[round(angle / 45) % 8]
                
                # Trend
                trend = cell.get("volume_trend", 1.0)
                if trend > 1.05:
                    trend_str = "Increasing"
                elif trend < 0.95:
                    trend_str = "Decreasing"
                else:
                    trend_str = "Stable"
                    
                # Phase
                phase = cell.get("lifecycle_phase", "MATURITY")
                phase_map = {"FORMATION": "Formation", "MATURITY": "Maturity", "DISSIPATION": "Dissipation", "ACTIVE": "Active"}
                phase_en = phase_map.get(phase, phase.capitalize())
                
                rows.append(html.Tr([
                    html.Td(short_id),
                    html.Td(f"{speed_kmh:.0f} km/h"),
                    html.Td(direction),
                    html.Td(phase_en),
                    html.Td(trend_str),
                ]))
        if not rows:
            return html.Div(html.I("No active cells at this moment."), className="text-muted small")
        return html.Div([
            html.H6("Active Storms Telemetry", className="fw-bold text-primary"),
            dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Storm ID"),
                        html.Th("Speed"),
                        html.Th("Direction"),
                        html.Th("Stage"),
                        html.Th("Intensity Evolution"),
                    ])),
                    html.Tbody(rows),
                ],
                bordered=True, hover=True, color="dark",
                className="kalman-diag mb-0",
            ),
        ])

    @staticmethod
    def _build_kinematic_rows(hist) -> list:
        """Table rows with average CSI/FAR/POD/FSS across horizons."""
        rows = []
        for horizon in HORIZON_NAMES:
            c = ReportBuilder._avg_metric(hist, "csi", horizon)
            f = ReportBuilder._avg_metric(hist, "far", horizon)
            p = ReportBuilder._avg_metric(hist, "pod", horizon)
            fs = ReportBuilder._avg_metric(hist, "fss", horizon)

            rows.append(html.Tr([
                html.Td(horizon), html.Td(f"{c:.2f}"), html.Td(f"{f:.2f}"),
                html.Td(f"{p:.2f}"), html.Td(f"{fs:.2f}")
            ]))
        return rows

    @staticmethod
    def _build_volume_rows(hist) -> list:
        """Table rows with real vs predicted accumulated MAP (L/m²) per horizon."""
        vol_rows = []

        for horizon in HORIZON_NAMES:
            vol_real_sum, vol_pred_sum = hist.volume_sums(horizon)
            delta_pct = ((vol_pred_sum - vol_real_sum) / vol_real_sum * 100.0) if vol_real_sum > 0.1 else 0.0

            vol_rows.append(html.Tr([
                html.Td(horizon), html.Td(f"{vol_real_sum:.2f} L/m²"),
                html.Td(f"{vol_pred_sum:.2f} L/m²"), html.Td(f"{delta_pct:+.1f}%")
            ]))
        return vol_rows

    @staticmethod
    def build_final_report(session_id: str, run_mode: str, frame_idx: int, total_frames: int, session_manager: SessionManager) -> html.Div | None:
        _, hist = session_manager.get_state(session_id)
        if hist.frames_processed == 0:
            return None

        if run_mode == "historic" and frame_idx < total_frames - 1:
            return None

        vol_rows = ReportBuilder._build_volume_rows(hist)

        reliability = hist.get_reliability_metrics()
        rel_rows = []
        for t, metrics in reliability.items():
            # For each threshold, display a special header row
            rel_rows.append(html.Tr([
                html.Td(f"Accumulation > {t} L/m²", colSpan=4, className="fw-bold bg-secondary text-light text-center")
            ]))
            for horizon in HORIZON_NAMES:
                pod = metrics[horizon]["pod"]
                far = metrics[horizon]["far"]
                cmae = metrics[horizon]["cmae"]
                rel_rows.append(html.Tr([
                    html.Td(horizon),
                    html.Td(f"{pod:.0f}%" if pod > 0 else "0%"),
                    html.Td(f"{far:.0f}%" if far > 0 else "0%"),
                    html.Td(f"± {cmae:.1f}%" if cmae > 0 else "-")
                ]))

        title_text = "Live Performance (Cumulative Stats)" if run_mode == "live" else "Historic Simulation Finished (Hydrological Mode)"
        alert_color = "info" if run_mode == "live" else "success"

        return html.Div([
            dbc.Alert(
                [
                    html.H4(title_text, className="alert-heading fw-bold"),
                    html.P("Catchment Level Volumetric Performance:"),
                    html.Hr(),
                    html.H6("Catchment Precipitation Accumulation", className="fw-bold mt-3"),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Accumulation Horizon"), 
                            html.Th("Actual (L/m²)"), 
                            html.Th("Predicted (L/m²)"), 
                            html.Th("Volumetric Bias (MAPE %)")
                        ])),
                        html.Tbody(vol_rows)
                    ], bordered=True, color="dark", hover=True, size="sm", className="mb-4"),
                    html.H6("Catchment Warning Confidence (Cumulative Volume Accuracy)", className="fw-bold mt-3"),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Horizon"), 
                            html.Th("Success (POD)"), 
                            html.Th("False Alarms (FAR)"),
                            html.Th("Error (CMAPE)")
                        ])),
                        html.Tbody(rel_rows)
                    ], bordered=True, color="dark", hover=True, size="sm", className="mb-2"),
                    html.Small("*CMAPE = Conditional Mean Absolute Percentage Error (only when rain is confirmed).", className="text-muted d-block mt-1")
                ],
                color=alert_color,
            )
        ])
