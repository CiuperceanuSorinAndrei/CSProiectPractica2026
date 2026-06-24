"""H-SAF Nowcasting Dashboard — Dash UI (Full Python Enterprise)."""
import os
import io
import base64
import datetime
from datetime import timedelta, datetime as dt

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactiv pentru viteza si stabilitate in server
import matplotlib.pyplot as plt

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from orchestrator import Orchestrator
from src.io.cloud_data_service import CloudDataService
from src.ui_helpers.plotting import StormMapPlotter

from config import (
    PREDEFINED_LOCATIONS,
    DEFAULT_RADIUS_KM,
    RAIN_VMAX,
)

# ---------------------------------------------------------------------------
# Initializare Stare Globala
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Orchestrator si serviciul de date sunt globale pentru acest PoC local
orch = Orchestrator()
data_service = CloudDataService(time_frames=[0, 15, 30, 45])

# Variabile de istoric tinute global
historic_state = {
    "last_frame_idx": -1,
    "total_volume_m3": 0.0,
    "csi": [],
    "far": [],
    "pod": [],
}


def get_nc_files():
    return sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".nc"))


def _parse_file_label(filename: str) -> str:
    """Extrage label-ul uman din numele fisierului H60."""
    try:
        parts = filename.split("_")
        d, o = parts[1], parts[2]
        return f"{d[6:]}-{d[4:6]} {o[:2]}:{o[2:]}"
    except (IndexError, ValueError):
        return filename


# ---------------------------------------------------------------------------
# Instantiere Aplicatie Dash
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Estimarea volumului de precipitatii"

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
sidebar = html.Div(
    [
        html.H4("Estimarea volumului de precipitatii din produse satelitare", className="text-primary fw-bold mb-3", style={"fontSize": "1.2rem"}),
        html.Hr(),

        html.H6("Mod de Rulare", className="fw-bold"),
        dbc.RadioItems(
            id="run-mode-select",
            options=[
                {"label": "Istoric", "value": "historic"},
                {"label": "LIVE", "value": "live"}
            ],
            value="historic",
            inline=True,
            className="mb-4"
        ),

        html.H6("Regiune de Interes (ROI)", className="fw-bold"),
        dbc.Label("Alege Locație"),
        dbc.Select(
            id="location-select",
            options=[{"label": k, "value": k} for k in PREDEFINED_LOCATIONS.keys()],
            value=list(PREDEFINED_LOCATIONS.keys())[0],
            className="mb-3",
        ),

        # Coordonate manuale (ascunse initial)
        html.Div(
            id="manual-coords-div",
            children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Latitudine"),
                        dbc.Input(id="manual-lat", type="number", value=44.33, step=0.1),
                    ]),
                    dbc.Col([
                        dbc.Label("Longitudine"),
                        dbc.Input(id="manual-lon", type="number", value=23.79, step=0.1),
                    ]),
                ], className="mb-3")
            ],
            style={"display": "none"}
        ),

        dbc.Label("Arie Vizualizare Hartă (km)"),
        dcc.Slider(100, 1000, 50, value=500, id="map-zoom-slider", className="mb-3"),

        dbc.Label("Rază Bazin/Oraș (km) - Volum"),
        dcc.Slider(10, 200, 5, value=30, id="roi-radius-slider", className="mb-4"),

        html.H6("Ingestie Date Istorice", className="fw-bold"),
        dbc.Row([
            dbc.Col(dcc.DatePickerRange(
                id='date-picker-range',
                start_date=datetime.date(2026, 6, 13),
                end_date=datetime.date(2026, 6, 14),
                display_format='YYYY-MM-DD',
                className="mb-2 w-100"
            ), width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Ora Start", className="small"),
                dbc.Input(id="start-hour", type="number", min=0, max=23, value=22, size="sm")
            ]),
            dbc.Col([
                dbc.Label("Ora Stop", className="small"),
                dbc.Input(id="end-hour", type="number", min=0, max=23, value=23, size="sm")
            ])
        ], className="mb-3"),
        dbc.Button("Descarcă Perioada Istorică", id="btn-download", color="secondary", outline=True, className="w-100 mb-3", size="sm"),
        html.Div(id="download-status", className="small text-success mb-3 fw-bold"),

        html.Hr(),
        html.H6("Control Timp", className="fw-bold"),

        dbc.Label(id="frame-label", children="Cadru Selectat: N/A", className="fw-bold text-primary"),
        dcc.Slider(0, max(len(get_nc_files()) - 1, 0), 1, value=0, id="frame-slider", className="mb-3"),

        dbc.Label("Viteza cadre (milisecunde)", className="small"),
        dcc.Slider(200, 3000, 200, value=1000, id="speed-slider", className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Button("Play/Pauză", id="btn-play", color="success", className="w-100")),
            dbc.Col(dbc.Button("Reset", id="btn-reset", color="danger", outline=True, className="w-100")),
        ]),

        dcc.Interval(id="animation-interval", interval=1000, n_intervals=0, disabled=True),

        # Interval ascuns pentru LIVE Polling
        dcc.Interval(id="live-polling-interval", interval=15 * 60 * 1000, n_intervals=0, disabled=True),
    ],
    className="bg-light p-4 h-100 shadow-sm border-end",
    style={"minHeight": "100vh"}
)


def create_metric_card(title, value_id, border_color="primary", text_color="dark"):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-muted text-uppercase small mb-1"),
            html.H3("N/A", id=value_id, className=f"text-{text_color} mb-0 fw-bold"),
        ]),
        className=f"border-{border_color} shadow-sm h-100"
    )


content = html.Div(
    [
        dbc.Row([
            dbc.Col(create_metric_card("Volum Acumulat (Istoric)", "val-historic-vol", "success", "success")),
            dbc.Col(create_metric_card("Aport Curent (15m)", "val-current-vol")),
            dbc.Col(create_metric_card("Volum Anticipat (Viitor)", "val-predicted-vol", "info", "info")),
            dbc.Col(create_metric_card("Rată Maximă (mm/h)", "val-max-rain")),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(create_metric_card("Metrici (CSI/FAR/POD)", "val-metrics")),
            dbc.Col(create_metric_card("Celule Urmărite", "val-tracked")),
            dbc.Col(create_metric_card("Celule în ROI", "val-in-roi")),
        ], className="mb-4"),

        html.Div(id="final-report-div", className="mb-4"),
        html.Div(id="diagnostics-div", className="mb-4"),

        dbc.Card(
            dbc.CardBody([
                html.Img(id="map-image", style={"width": "100%", "borderRadius": "5px"})
            ]),
            className="shadow-sm border-0"
        )
    ],
    className="p-4"
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3, className="p-0"),
                dbc.Col(content, width=9),
            ],
            className="g-0",
        )
    ],
    fluid=True,
    className="g-0",
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("manual-coords-div", "style"),
    Input("location-select", "value")
)
def toggle_manual_coords(loc):
    if loc == "Manual (Introducere coordonate)":
        return {"display": "block"}
    return {"display": "none"}


@app.callback(
    Output("animation-interval", "disabled"),
    Input("btn-play", "n_clicks"),
    State("animation-interval", "disabled"),
    prevent_initial_call=True
)
def toggle_play(n_clicks, currently_disabled):
    return not currently_disabled


@app.callback(
    Output("frame-slider", "value"),
    Input("animation-interval", "n_intervals"),
    State("frame-slider", "value"),
    State("frame-slider", "max"),
    prevent_initial_call=True
)
def auto_advance_frame(n, current_frame, max_frame):
    if current_frame < max_frame:
        return current_frame + 1
    return current_frame


@app.callback(
    Output("animation-interval", "interval"),
    Input("speed-slider", "value")
)
def update_speed(val):
    return val


@app.callback(
    Output("live-polling-interval", "disabled"),
    Output("btn-play", "disabled"),
    Output("btn-reset", "disabled"),
    Output("frame-slider", "disabled"),
    Output("date-picker-range", "disabled"),
    Output("start-hour", "disabled"),
    Output("end-hour", "disabled"),
    Output("btn-download", "disabled"),
    Input("run-mode-select", "value"),
)
def toggle_live_mode(mode):
    is_live = (mode == "live")
    # Cand e LIVE, dezactivam controalele istorice si pornim polling-ul
    return not is_live, is_live, is_live, is_live, is_live, is_live, is_live, is_live


@app.callback(
    Output("frame-slider", "max", allow_duplicate=True),
    Output("frame-slider", "value", allow_duplicate=True),
    Input("live-polling-interval", "n_intervals"),
    Input("run-mode-select", "value"),
    prevent_initial_call=True
)
def poll_live_data(n_int, mode):
    if mode != "live":
        raise dash.exceptions.PreventUpdate

    # Se executa la trecerea in LIVE sau la fiecare 15 minute (Interval)
    data_service.fetch_latest()

    files = get_nc_files()
    if not files:
        raise dash.exceptions.PreventUpdate

    max_idx = len(files) - 1
    return max_idx, max_idx


@app.callback(
    Output("download-status", "children"),
    Output("frame-slider", "max", allow_duplicate=True),
    Input("btn-download", "n_clicks"),
    State("date-picker-range", "start_date"),
    State("date-picker-range", "end_date"),
    State("start-hour", "value"),
    State("end-hour", "value"),
    prevent_initial_call=True
)
def download_historic(n, start_d, end_d, start_h, end_h):
    if not start_d or not end_d:
        return "Selectează datele!", dash.no_update

    try:
        start_dt = dt.fromisoformat(start_d).replace(hour=int(start_h))
        end_dt = dt.fromisoformat(end_d).replace(hour=int(end_h))
    except ValueError:
        return "Date invalide!", dash.no_update

    if start_dt >= end_dt:
        return "Start trebuie să fie înaintea Stop!", dash.no_update

    target_files = []
    current_dt = start_dt
    while current_dt <= end_dt:
        filename = (
            f"h60_{current_dt.year}{current_dt.month:02d}{current_dt.day:02d}"
            f"_{current_dt.hour:02d}{current_dt.minute:02d}_fdk.nc.gz"
        )
        target_files.append(filename)
        current_dt += timedelta(minutes=15)

    missing_files = [f for f in target_files if not os.path.exists(os.path.join(DATA_DIR, f.replace(".gz", "")))]

    if missing_files:
        data_service.download_files(missing_files)
        msg = f"S-au descărcat {len(missing_files)} fișiere noi."
    else:
        msg = "Fișierele există deja local."

    files = get_nc_files()
    return msg, max(len(files) - 1, 0)


@app.callback(
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

    Input("frame-slider", "value"),
    Input("location-select", "value"),
    Input("manual-lat", "value"),
    Input("manual-lon", "value"),
    Input("map-zoom-slider", "value"),
    Input("roi-radius-slider", "value"),
    Input("btn-reset", "n_clicks"),
    State("run-mode-select", "value")
)
def update_dashboard(frame_idx, loc_choice, m_lat, m_lon, map_zoom, radius_km, reset_clicks, run_mode):
    # dash.ctx.triggered_id este None la apelul initial; ridica exceptie daca
    # functia e apelata in afara unui callback (ex: harness-ul de debug).
    try:
        triggered_id = dash.ctx.triggered_id
    except dash.exceptions.MissingCallbackContextException:
        triggered_id = None

    if triggered_id == "btn-reset":
        orch.reset_tracking()
        historic_state.update({"last_frame_idx": -1, "total_volume_m3": 0.0, "csi": [], "far": [], "pod": []})
        return dash.no_update, "0.00 mii m³", "0.00 mii m³", "0.00 mii m³", "0.00", "0.0 / 0.0 / 0.0", "0", "0", "Alege cadru", "", ""

    nc_files = get_nc_files()
    if not nc_files:
        return "", "0.00", "0.00", "0.00", "0.0", "N/A", "0", "0", "Fără Date", "", ""

    frame_idx = min(max(frame_idx, 0), len(nc_files) - 1)
    file_path = os.path.join(DATA_DIR, nc_files[frame_idx])

    parts = nc_files[frame_idx].split("_")
    date_str = parts[1]
    time_str = parts[2]
    current_label = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:]}"

    if loc_choice == "Manual (Introducere coordonate)":
        center_lat, center_lon = float(m_lat), float(m_lon)
    else:
        center_lat = float(PREDEFINED_LOCATIONS[loc_choice]["lat"])
        center_lon = float(PREDEFINED_LOCATIONS[loc_choice]["lon"])

    delta_lat = map_zoom / 111.0
    delta_lon = map_zoom / (111.0 * np.cos(np.radians(center_lat)))
    lat_min, lat_max = center_lat - delta_lat, center_lat + delta_lat
    lon_min, lon_max = center_lon - delta_lon, center_lon + delta_lon

    # Avans consecutiv = tracker-ul are deja starea anterioara corecta (acumulam volum + metrici).
    # Orice altceva (schimbare cadru/zoom/ROI/locatie) = reincalzim tracker-ul cu ultimele 3 cadre.
    is_consecutive = (frame_idx == historic_state["last_frame_idx"] + 1)

    if is_consecutive:
        result = orch.process_frame(file_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
        if result is None:
            return "", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", f"Eroare fisier {current_label}", "", ""
        historic_state["total_volume_m3"] += result.roi_volume_m3
        if result.global_csi is not None:
            historic_state["csi"].append(result.global_csi)
            historic_state["far"].append(result.global_far)
            historic_state["pod"].append(result.global_pod)
    else:
        orch.reset_tracking()
        for i in range(max(0, frame_idx - 3), frame_idx):
            w_path = os.path.join(DATA_DIR, nc_files[i])
            orch.process_frame(w_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
        result = orch.process_frame(file_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
        if result is None:
            return "", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", f"Eroare fisier {current_label}", "", ""
        historic_state["total_volume_m3"] = result.roi_volume_m3

    historic_state["last_frame_idx"] = frame_idx

    title = f"{current_label} UTC"
    if run_mode == "live":
        title = f"🔴 LIVE NOWCAST: {current_label} UTC"

    fig, ax, _ = StormMapPlotter.create_figure(
        result.lon_grid, result.lat_grid, result.rain_rate_masked,
        extent=(lon_min, lon_max, lat_min, lat_max),
        vmin=0.1, vmax=RAIN_VMAX,
        title=title,
        roi_center=(center_lat, center_lon),
        roi_radius_km=radius_km,
    )
    StormMapPlotter.draw_overlays(ax, result.tracked_cells, result.lon_grid, result.lat_grid)

    # --- Diagnostic ---
    diag_rows = []
    for cell in result.tracked_cells:
        vt = cell.get("volume_trend", 1.0)
        stare = "Extindere" if vt > 1.12 else "Disipare" if vt < 0.88 else "Stabil"
        err = f"{cell.get('prediction_error_pixels', 0.0):.1f} px" if cell["is_tracked"] else "Nou"

        diag_rows.append(html.Tr([
            html.Td(cell["cell_id"]),
            html.Td(f"{cell['geo_lat']:.2f}, {cell['geo_lon']:.2f}"),
            html.Td(str(cell.get("area_pixels", 0))),
            html.Td(err),
            html.Td(stare)
        ]))

    diag_table = dbc.Table([
        html.Thead(html.Tr([html.Th("ID Sistem"), html.Th("Locatie"), html.Th("Arie (px)"), html.Th("Eroare Centroid"), html.Th("Evolutie")])),
        html.Tbody(diag_rows)
    ], bordered=True, hover=True, size="sm", className="bg-white mb-0")

    diagnostics_ui = html.Div([
        html.H6("Diagnostic Corelat", className="fw-bold text-muted mb-2"),
        html.Div(diag_table, style={"maxHeight": "250px", "overflowY": "auto", "border": "1px solid #dee2e6", "borderRadius": "5px"})
    ], className="mb-4")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("ascii")
    src = f"data:image/png;base64,{img_base64}"

    val_hist_vol = f"{historic_state['total_volume_m3'] / 1000.0:.2f} mii m³"
    val_curr_vol = f"{result.roi_volume_m3 / 1000.0:.2f} mii m³"
    val_pred_vol = f"{result.predicted_roi_volume_m3 / 1000.0:.2f} mii m³"
    val_max_r = f"{result.max_rain:.2f}"
    val_metrics = (
        f"{result.global_csi:.2f} / {result.global_far:.2f} / {result.global_pod:.2f}"
        if result.global_csi is not None else "N/A (Fără ploaie)"
    )
    val_trck = f"{result.num_tracked}"
    val_in_roi = f"{sum(1 for c in result.tracked_cells if c.get('in_roi'))}"
    lbl_frame = f"Cadru: {current_label} UTC ({frame_idx+1}/{len(nc_files)})"

    final_report = ""
    # Raportul final doar in modul istoric cand a ajuns la capat
    if run_mode != "live" and frame_idx == len(nc_files) - 1:
        avg_csi = np.mean(historic_state["csi"]) if historic_state["csi"] else 0.0
        avg_far = np.mean(historic_state["far"]) if historic_state["far"] else 0.0
        avg_pod = np.mean(historic_state["pod"]) if historic_state["pod"] else 0.0

        final_report = dbc.Alert(
            [
                html.H4("🏁 Simulare Istorică Încheiată!", className="alert-heading"),
                html.P("Raport de Performanță al Algoritmului pentru intervalul selectat:"),
                html.Hr(),
                dbc.Row([
                    dbc.Col(html.Strong(f"Volum Total: {historic_state['total_volume_m3']/1000.0:.0f} m³")),
                    dbc.Col(html.Strong(f"Acuratețe (CSI): {avg_csi:.2f}")),
                    dbc.Col(html.Strong(f"Alarme False (FAR): {avg_far:.2f}")),
                    dbc.Col(html.Strong(f"Detecții (POD): {avg_pod:.2f}")),
                ])
            ],
            color="success",
        )

    return src, val_hist_vol, val_curr_vol, val_pred_vol, val_max_r, val_metrics, val_trck, val_in_roi, lbl_frame, final_report, diagnostics_ui


# ---------------------------------------------------------------------------
# Rulare Server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Pornește serverul Dash... Deschide http://127.0.0.1:8050 în browser!")
    app.run(debug=True, port=8050)
