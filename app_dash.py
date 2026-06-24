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
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from orchestrator import Orchestrator
from src.io.cloud_data_service import CloudDataService
from src.ui_helpers.plotting import StormMapPlotter

from config import (
    PREDEFINED_LOCATIONS, 
    DEFAULT_RADIUS_KM, 
    RAIN_VMAX,
    RAIN_THRESHOLD_MIN
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


def get_filtered_nc_files(time_range=None, run_mode="historic"):
    files = get_nc_files()
    if run_mode == "live" or not time_range:
        return files
        
    filtered = []
    try:
        start_dt = dt.fromisoformat(time_range["start"])
        end_dt = dt.fromisoformat(time_range["end"])
    except Exception:
        return files
        
    for f in files:
        parts = f.split("_")
        if len(parts) >= 3:
            date_str = parts[1]
            time_str = parts[2]
            try:
                f_dt = dt.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
                if start_dt <= f_dt <= end_dt:
                    filtered.append(f)
            except ValueError:
                pass
    return filtered


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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Estimarea volumului de precipitatii"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Ascundem complet orice input asociat slider-ului (feature nativ din Dash) */
            .dash-input-container, .dash-range-slider-input, .dash-range-slider-max-input { display: none !important; }
            
            /* Ascundem complet marcajele numerice de sub slider */
            .dash-slider-mark, .rc-slider-mark, .rc-slider-mark-text { display: none !important; }
            
            .rc-slider-tooltip-inner { background-color: #343a40 !important; color: #f8f9fa !important; border: 1px solid #6c757d !important; box-shadow: none !important; }
            
            /* DatePickerRange Text Inputs */
            .DateInput, .DateInput_1 { background-color: #343a40 !important; background: #343a40 !important; }
            .DateInput_input, .DateInput_input_1, input.DateInput_input { background-color: #343a40 !important; background: #343a40 !important; color: #f8f9fa !important; border: 1px solid #6c757d !important; }
            .DateRangePickerInput, .DateRangePickerInput_1 { background-color: #343a40 !important; background: #343a40 !important; border: 1px solid #6c757d !important; border-radius: 4px; overflow: hidden; }
            .DateRangePickerInput_arrow, .DateRangePickerInput_arrow_1 { background-color: #343a40 !important; background: #343a40 !important; }
            .DateRangePickerInput_arrow_svg { fill: #f8f9fa !important; }
            
            /* DatePickerRange Popup Calendar */
            .DayPicker, .DayPicker_1 { background-color: #222 !important; }
            .CalendarMonth, .CalendarMonth_1, .CalendarMonthGrid, .CalendarMonthGrid_1 { background-color: #222 !important; }
            .CalendarMonth_caption, .CalendarMonth_caption_1 { color: #fff !important; }
            .DayPickerNavigation_button__default, .DayPickerNavigation_button__default_1 { background-color: #333 !important; }
            .DayPickerNavigation_svg__default { fill: #fff !important; }
            .DayPickerKeyboardShortcuts_buttonReset { display: none !important; }
            .CalendarDay__default, .CalendarDay__default_1 { background-color: #333 !important; color: #fff !important; border: 1px solid #444 !important; }
            .CalendarDay__default:hover { background-color: #555 !important; }
            .CalendarDay__selected, .CalendarDay__selected_1 { background-color: #0d6efd !important; color: #fff !important; border: 1px solid #0d6efd !important; }
            .CalendarDay__selected_span, .CalendarDay__selected_span_1 { background-color: #0a58ca !important; color: #fff !important; }
            .CalendarDay__hovered_span, .CalendarDay__hovered_span_1 { background-color: #444 !important; color: #fff !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
sidebar = html.Div(
    [
        html.H4("Estimarea volumului de precipitatii din produse satelitare", className="text-info fw-bold mb-3", style={"fontSize": "1.2rem"}),
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
            className="mb-3 bg-dark text-light border-secondary",
        ),

        # Coordonate manuale (ascunse initial)
        html.Div(
            id="manual-coords-div",
            children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Latitudine", className="text-light"),
                        dbc.Input(id="manual-lat", type="number", value=44.33, step=0.1, className="bg-dark text-light border-secondary"),
                    ]),
                    dbc.Col([
                        dbc.Label("Longitudine", className="text-light"),
                        dbc.Input(id="manual-lon", type="number", value=23.79, step=0.1, className="bg-dark text-light border-secondary"),
                    ]),
                ], className="mb-3")
            ],
            style={"display": "none"}
        ),

        dbc.Label("Arie Vizualizare Hartă (km)", className="text-light"),
        dbc.Input(id="map-zoom-slider", type="number", step=10, debounce=True, value=500, className="mb-3 bg-dark text-light border-secondary"),

        dbc.Label("Rază Bazin/Oraș (km) - Volum", className="text-light"),
        dbc.Input(id="roi-radius-slider", type="number", step=1, debounce=True, value=30, className="mb-4 bg-dark text-light border-secondary"),

        html.Div(id="input-warnings"),

        html.H6("Ingestie Date Istorice", className="fw-bold"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Data Start", className="small text-light mb-1"),
                dbc.Input(id="start-date", type="date", value="2026-06-13", className="bg-dark text-light border-secondary", size="sm")
            ], width=6, className="pe-1"),
            dbc.Col([
                dbc.Label("Data Stop", className="small text-light mb-1"),
                dbc.Input(id="end-date", type="date", value="2026-06-14", className="bg-dark text-light border-secondary", size="sm")
            ], width=6, className="ps-1")
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Ora Start", className="small text-light"),
                dbc.Input(id="start-hour", type="number", min=0, max=23, value=22, size="sm", className="bg-dark text-light border-secondary")
            ]),
            dbc.Col([
                dbc.Label("Ora Stop", className="small text-light"),
                dbc.Input(id="end-hour", type="number", min=0, max=23, value=23, size="sm", className="bg-dark text-light border-secondary")
            ])
        ], className="mb-3"),
        dbc.Button("Validează & Descarcă", id="btn-download", color="light", outline=True, className="w-100 mb-3", size="sm", style={"fontWeight": "bold"}),
        dcc.Loading(
            id="loading-download",
            type="circle",
            color="#0dcaf0",
            children=html.Div(id="download-status", className="small text-success mb-3 fw-bold")
        ),

        html.Hr(),
        html.H6("Control Timp", className="fw-bold"),

        dbc.Label(id="frame-label", children="Cadru Selectat: N/A", className="fw-bold text-light"),
        dcc.Slider(0, max(len(get_filtered_nc_files({"start": "2026-06-13T22:00:00", "end": "2026-06-14T23:00:00"}, "historic")) - 1, 0), 1, value=0, marks={}, id="frame-slider", className="mb-3"),


        dbc.Row([
            dbc.Col(dbc.Button("Play/Pauză", id="btn-play", color="success", outline=True, className="w-100 fw-bold")),
            dbc.Col(dbc.Button("Reset", id="btn-reset", color="danger", outline=True, className="w-100 fw-bold")),
        ]),

        dcc.Interval(id="animation-interval", interval=200, n_intervals=0, disabled=True),
        dcc.Store(id="is-processing", data=False),
        dcc.Store(id="active-time-range", data={"start": "2026-06-13T22:00:00", "end": "2026-06-14T23:00:00"}),

        # Interval ascuns pentru LIVE Polling
        dcc.Interval(id="live-polling-interval", interval=15 * 60 * 1000, n_intervals=0, disabled=True),
    ],
    className="bg-dark text-light p-4 shadow-sm border-end border-secondary",
    style={"minHeight": "100vh"}
)


def create_metric_card(title, value_id, border_color="primary", text_color="light"):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-muted text-uppercase small mb-1"),
            html.H3("N/A", id=value_id, className=f"text-{text_color} mb-0 fw-bold"),
        ]),
        className=f"border-{border_color} bg-dark text-light shadow-sm h-100"
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
            className="shadow-sm border-secondary bg-dark"
        )
    ],
    className="p-4 bg-dark text-light h-100",
    style={"minHeight": "100vh"}
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
    className="g-0 bg-dark text-light",
    style={"minHeight": "100vh"}
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
    Output("is-processing", "data", allow_duplicate=True),
    Output("animation-interval", "disabled", allow_duplicate=True),
    Input("animation-interval", "n_intervals"),
    State("is-processing", "data"),
    State("frame-slider", "value"),
    State("frame-slider", "max"),
    prevent_initial_call=True
)
def auto_advance_frame(n, is_processing, current_frame, max_frame):
    if is_processing:
        # Daca serverul inca proceseaza un cadru, sarim peste acest "tick" 
        # fara sa mutam slider-ul, evitand blocarea si sarirea peste cadre.
        raise PreventUpdate

    if current_frame < max_frame:
        return current_frame + 1, True, dash.no_update
    
    # Daca a ajuns la final, opreste intervalul (pauza) ca sa poata fi citit raportul
    return current_frame, False, True


@app.callback(
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
)
def toggle_live_mode(mode):
    is_live = (mode == "live")
    # Cand e LIVE, dezactivam controalele istorice si pornim polling-ul
    # primul e live-polling-interval (disabled = not is_live)
    # restul sunt butoanele istorice (disabled = is_live)
    return not is_live, is_live, is_live, is_live, is_live, is_live, is_live, is_live, is_live


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
    Output("frame-slider", "value", allow_duplicate=True),
    Output("active-time-range", "data"),
    Input("btn-download", "n_clicks"),
    State("start-date", "value"),
    State("end-date", "value"),
    State("start-hour", "value"),
    State("end-hour", "value"),
    prevent_initial_call=True
)
def download_historic(n, start_d, end_d, start_h, end_h):
    if not start_d or not end_d:
        return "Selectează datele!", dash.no_update, dash.no_update, dash.no_update
    
    if start_h is None or end_h is None:
        return "Setați o oră validă (0-23)!", dash.no_update, dash.no_update, dash.no_update

    try:
        h_s = int(start_h)
        h_e = int(end_h)
        if not (0 <= h_s <= 23) or not (0 <= h_e <= 23):
            return "Orele trebuie să fie între 0 și 23!", dash.no_update, dash.no_update, dash.no_update
            
        start_dt = dt.fromisoformat(start_d).replace(hour=h_s)
        end_dt = dt.fromisoformat(end_d).replace(hour=h_e)
    except Exception:
        return "Format dată/oră invalid!", dash.no_update, dash.no_update, dash.no_update

    if start_dt >= end_dt:
        return "Timpul de Start trebuie să fie înaintea Stop!", dash.no_update, dash.no_update, dash.no_update

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
        msg = f"✓ S-au descărcat {len(missing_files)} fișiere noi. Gata de folosire!"
    else:
        msg = "✓ Datele există deja local. Gata de folosire!"

    time_range = {"start": start_dt.isoformat(), "end": end_dt.isoformat()}
    filtered_files = get_filtered_nc_files(time_range, run_mode="historic")
    
    return msg, max(len(filtered_files) - 1, 0), 0, time_range


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
    State("active-time-range", "data")
)
def update_dashboard(frame_idx, loc_choice, m_lat, m_lon, map_zoom, radius_km, run_mode, time_range):
    # dash.ctx.triggered_id este None la apelul initial; ridica exceptie daca
    # functia e apelata in afara unui callback (ex: harness-ul de debug).
    try:
        triggered_id = dash.ctx.triggered_id
    except dash.exceptions.MissingCallbackContextException:
        triggered_id = None

    # Protectie impotriva valorilor extreme (clamping) in caz ca se tasteaza fortat
    raw_map_zoom = map_zoom
    raw_radius = radius_km
    map_zoom = min(max(map_zoom, 100), 700) if map_zoom is not None else 500
    radius_km = min(max(radius_km, 5), 200) if radius_km is not None else 30

    warning_ui = []
    if raw_map_zoom is None:
        warning_ui.append(dbc.Alert("Valoare invalidă pentru Arie. S-a folosit valoarea implicită (500 km).", color="danger", style={"padding": "0.5rem"}, className="small mb-2"))
    elif raw_map_zoom > 700 or raw_map_zoom < 100:
        warning_ui.append(dbc.Alert(f"Aria introdusă ({raw_map_zoom} km) a fost respinsă. Valoarea maximă permisă este 700 km (minim 100 km).", color="danger", style={"padding": "0.5rem"}, className="small mb-2 fw-bold"))
        
    if raw_radius is None:
        warning_ui.append(dbc.Alert("Valoare invalidă pentru Rază. S-a folosit valoarea implicită (30 km).", color="danger", style={"padding": "0.5rem"}, className="small mb-2"))
    elif raw_radius > 200 or raw_radius < 5:
        warning_ui.append(dbc.Alert(f"Raza introdusă ({raw_radius} km) a fost respinsă. Valoarea maximă permisă este 200 km (minim 5 km).", color="danger", style={"padding": "0.5rem"}, className="small mb-2 fw-bold"))

    nc_files = get_filtered_nc_files(time_range, run_mode)
    if not nc_files:
        return "", "0.00", "0.00", "0.00", "0.0", "N/A", "0", "0", "Fără Date", "", "", False, dash.no_update, dash.no_update, dash.no_update

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
    is_same_frame = (frame_idx == historic_state["last_frame_idx"])

    if is_consecutive:
        result = orch.process_frame(file_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
        if result is None:
            raise PreventUpdate
        historic_state["total_volume_m3"] += result.roi_volume_m3
        if result.global_csi is not None:
            historic_state["csi"].append(result.global_csi)
            historic_state["far"].append(result.global_far)
            historic_state["pod"].append(result.global_pod)
    elif is_same_frame:
        # Utilizatorul probabil a dat zoom sau a schimbat raza.
        # Reprocesam cadrul pentru vizualizare, dar NU modificam istoricul de volum sau metricile globale.
        result = orch.process_frame(file_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
        if result is None:
            raise PreventUpdate
    else:
        # A avut loc un "jump" real. Daca e inainte, continuam calculul pentru toate cadrele omise.
        # Daca e inapoi, istoricul a fost deja curatat mai sus si o luam de la 0.
        start_process_idx = max(0, historic_state["last_frame_idx"] + 1)
        for i in range(start_process_idx, frame_idx):
            w_path = os.path.join(DATA_DIR, nc_files[i])
            inter_result = orch.process_frame(w_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
            if inter_result:
                historic_state["total_volume_m3"] += inter_result.roi_volume_m3
                if inter_result.global_csi is not None: historic_state["csi"].append(inter_result.global_csi)
                if inter_result.global_far is not None: historic_state["far"].append(inter_result.global_far)
                if inter_result.global_pod is not None: historic_state["pod"].append(inter_result.global_pod)

        result = orch.process_frame(file_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
        if result is None:
            return "", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", "Eroare", f"Eroare fisier {current_label}", "", "", False, dash.no_update, dash.no_update, dash.no_update
        
        historic_state["total_volume_m3"] += result.roi_volume_m3
        if result.global_csi is not None: historic_state["csi"].append(result.global_csi)
        if result.global_far is not None: historic_state["far"].append(result.global_far)
        if result.global_pod is not None: historic_state["pod"].append(result.global_pod)

    historic_state["last_frame_idx"] = frame_idx

    title = f"{current_label} UTC"
    if run_mode == "live":
        title = f"🔴 LIVE NOWCAST: {current_label} UTC"

    fig, ax, _ = StormMapPlotter.create_figure(
        result.lon_grid, result.lat_grid, result.rain_rate_masked,
        extent=(lon_min, lon_max, lat_min, lat_max),
        vmin=RAIN_THRESHOLD_MIN, vmax=RAIN_VMAX,
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
        html.Thead(html.Tr([html.Th("ID Sistem"), html.Th("Locație"), html.Th("Arie (px)"), html.Th("Eroare Centroid"), html.Th("Evoluție")])),
        html.Tbody(diag_rows)
    ], bordered=True, hover=True, size="sm", className="table-dark mb-0")

    diagnostics_ui = html.Div([
        html.H6("Diagnostic Corelat", className="fw-bold text-muted mb-2"),
        html.Div(diag_table, style={"maxHeight": "250px", "overflowY": "auto", "border": "1px solid #444", "borderRadius": "5px"})
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
                html.H4("Simulare Istorică Încheiată", className="alert-heading"),
                html.P("Raport de Performanță al Algoritmului pentru intervalul analizat:"),
                html.Hr(),
                dbc.Row([
                    dbc.Col(html.Strong(f"Volum Total Precipitat (Bazin): {historic_state['total_volume_m3']/1000.0:.0f} mii m³")),
                    dbc.Col(html.Strong(f"Acuratețe Detecție (CSI): {avg_csi:.2f}")),
                    dbc.Col(html.Strong(f"Rată Alarme False (FAR): {avg_far:.2f}")),
                    dbc.Col(html.Strong(f"Probabilitate Detecție (POD): {avg_pod:.2f}")),
                ])
            ],
            color="success",
        )

    return src, val_hist_vol, val_curr_vol, val_pred_vol, val_max_r, val_metrics, val_trck, val_in_roi, lbl_frame, final_report, diagnostics_ui, False, warning_ui, map_zoom, radius_km


@app.callback(
    Output("frame-slider", "value", allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True
)
def handle_reset(n_clicks):
    if n_clicks:
        orch.reset_tracking()
        historic_state.update({"last_frame_idx": -1, "total_volume_m3": 0.0, "csi": [], "far": [], "pod": []})
        return 0
    return dash.no_update


# ---------------------------------------------------------------------------
# Rulare Server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Pornește serverul Dash... Deschide http://127.0.0.1:8050 în browser!")
    app.run(debug=True, port=8050)
