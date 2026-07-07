"""UI structure construction (sidebar + content)."""
import uuid
from dash import dcc, html
import dash_bootstrap_components as dbc

from src.config import PREDEFINED_LOCATIONS
from src.dashboard.constants import DEFAULT_TIME_RANGE, MAP_ZOOM_DEFAULT, ROI_RADIUS_DEFAULT
from src.dashboard.frame_store import FrameStore


class DashboardLayout:
    """Builds the UI structure (sidebar + content)."""

    def __init__(self, store: FrameStore):
        self._store = store

    def build(self) -> dbc.Container:
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(self._sidebar(), width=3, className="p-0"),
                        dbc.Col(self._content(), width=9),
                    ],
                    className="g-0",
                )
            ],
            fluid=True,
            className="g-0 bg-dark text-light",
            style={"minHeight": "100vh"},
        )

    # ---- sidebar -----------------------------------------------------------
    def _sidebar(self) -> html.Div:
        return html.Div(
            [
                html.H4("Satellite Precipitation Volume Estimation",
                        className="text-info fw-bold mb-3", style={"fontSize": "1.2rem"}),
                html.Hr(),
                *self._server_config_section(),
                *self._run_mode_section(),
                *self._roi_section(),
                *self._hydrology_section(),
                *self._ingestion_section(),
                html.Hr(),
                *self._time_section(),
                *self._stores_and_intervals(),
            ],
            className="bg-dark text-light p-4 shadow-sm border-end border-secondary app-sidebar",
        )

    @staticmethod
    def _server_config_section() -> list:
        """Editable fields for server/folders/credentials/format, on top."""
        from src.io.server_settings import ServerSettings
        s = ServerSettings.load()
        inp = {"className": "mb-2 bg-dark text-light border-secondary", "size": "sm", "debounce": True}
        return [
            # Header button to toggle the configuration block.
            dbc.Button(
                "▸ Server Configuration",
                id="toggle-server-config", color="link", size="sm",
                className="fw-bold text-light text-decoration-none p-0 mb-2 shadow-none",
            ),
            dbc.Collapse(
                [
                    dbc.Label("FTP Host", className="small text-light mb-1"),
                    dbc.Input(id="srv-host", type="text", value=s.host, **inp),
                    dbc.Label("Server directory (source)", className="small text-light mb-1"),
                    dbc.Input(id="srv-remote-dir", type="text", value=s.remote_dir, **inp),
                    dbc.Label("Local directory (save)", className="small text-light mb-1"),
                    dbc.Input(id="srv-local-dir", type="text", value=s.local_dir, **inp),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("User", className="small text-light mb-1"),
                            dbc.Input(id="srv-user", type="text", value="", **inp),
                        ]),
                        dbc.Col([
                            dbc.Label("Password", className="small text-light mb-1"),
                            dbc.Input(id="srv-pass", type="password", value="", **inp),
                        ]),
                    ]),
                    dbc.Label("File format (strftime)", className="small text-light mb-1"),
                    dbc.Input(id="srv-format", type="text", value=s.file_format, **inp),
                    dbc.Label("Interval between frames (min)", className="small text-light mb-1"),
                    dbc.Input(id="time-delta", type="number", value=s.time_delta, min=1, step=1, **inp),
                    dbc.Button("Connect / Apply", id="btn-apply-config", color="primary", size="sm", className="mt-2 w-100 fw-bold"),
                ],
                id="server-config-collapse", is_open=False,
            ),
            html.Hr(),
        ]

    @staticmethod
    def _run_mode_section() -> list:
        return [
            html.H6("Run Mode", className="fw-bold"),
            dbc.RadioItems(
                id="run-mode-select",
                options=[{"label": "Historic", "value": "historic"}, {"label": "LIVE", "value": "live"}],
                value="historic", inline=True, className="mb-2",
            ),
            html.Div(id="live-status", className="small mb-3"),
        ]

    @staticmethod
    def _roi_section() -> list:
        from src.geo.reservoir_loader import ReservoirLoader
        reservoirs = ReservoirLoader.get_all_reservoirs()
        res_options = [{"label": k, "value": k} for k in sorted(reservoirs.keys())]

        return [
            html.H6("Region of Interest (ROI)", className="fw-bold"),
            dbc.RadioItems(
                id="location-type",
                options=[
                    {"label": "City (Circle)", "value": "predefined"},
                    {"label": "Reservoir (Polygon)", "value": "reservoir"}
                ],
                value="predefined",
                inline=False,
                className="mb-2 text-light"
            ),
            html.Div(
                id="predefined-loc-div",
                children=[
                    dbc.Label("Select location (Point)"),
                    dbc.Select(
                        id="location-select",
                        options=[{"label": k, "value": k} for k in PREDEFINED_LOCATIONS.keys()],
                        value=list(PREDEFINED_LOCATIONS.keys())[0],
                        className="mb-3 bg-dark text-light border-secondary",
                    )
                ]
            ),
            html.Div(
                id="reservoir-loc-div",
                style={"display": "none"},
                children=[
                    dbc.Label("Select Reservoir (Exact Polygon)"),
                    dcc.Dropdown(
                        id="reservoir-select",
                        options=res_options,
                        value=res_options[0]["value"] if res_options else None,
                        searchable=True,
                        clearable=False,
                        className="mb-3 custom-dark-dropdown",
                    )
                ]
            ),
            html.Div(
                id="manual-coords-div",
                children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Latitude", className="text-light"),
                            dbc.Input(id="manual-lat", type="number", value=44.33, step=0.1,
                                      className="bg-dark text-light border-secondary"),
                        ]),
                        dbc.Col([
                            dbc.Label("Longitude", className="text-light"),
                            dbc.Input(id="manual-lon", type="number", value=23.79, step=0.1,
                                      className="bg-dark text-light border-secondary"),
                        ]),
                    ], className="mb-3")
                ],
                style={"display": "none"},
            ),
            dbc.Label("Map View Area (km)", className="text-light"),
            dbc.Input(id="map-zoom-input", type="number", step=10, debounce=True, value=MAP_ZOOM_DEFAULT,
                      className="mb-3 bg-dark text-light border-secondary"),
            html.Div(
                id="radius-input-div",
                children=[
                    dbc.Label("Circular Radius (km) - Volume", className="text-light"),
                    dbc.Input(id="roi-radius-input", type="number", step=1, debounce=True, value=ROI_RADIUS_DEFAULT,
                              className="mb-4 bg-dark text-light border-secondary"),
                ]
            ),
            html.Div(id="input-warnings"),
        ]

    @staticmethod
    def _hydrology_section() -> list:
        from src.config import EVAP_MM_PER_DAY, RESERVOIR_OUTFLOW_M3S
        return [
            html.H6("Hydrological Balance (Outflows)", className="fw-bold"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Evaporation (mm/day)", className="small text-light"),
                    dbc.Input(id="evap-input", type="number", min=0, step=0.5, debounce=True,
                              value=EVAP_MM_PER_DAY, size="sm",
                              className="bg-dark text-light border-secondary"),
                ]),
                dbc.Col([
                    dbc.Label("Outflow (m³/s)", className="small text-light"),
                    dbc.Input(id="outflow-input", type="number", min=0, step=1, debounce=True,
                              value=RESERVOIR_OUTFLOW_M3S, size="sm",
                              className="bg-dark text-light border-secondary"),
                ]),
            ], className="mb-1"),
            html.Small("0 = ignored (short nowcast windows).", className="text-muted d-block mb-3"),
        ]

    @staticmethod
    def _ingestion_section() -> list:
        return [
            html.Div([
                html.H6("Historic Data Acquisition", className="fw-bold"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Start Date", className="small text-light"),
                        dcc.DatePickerSingle(
                            id="start-date", date=DEFAULT_TIME_RANGE["start"].split("T")[0],
                            display_format="YYYY-MM-DD", className="mb-2 d-block bg-dark"
                        )
                    ]),
                    dbc.Col([
                        dbc.Label("Stop Date", className="small text-light"),
                        dcc.DatePickerSingle(
                            id="end-date", date=DEFAULT_TIME_RANGE["end"].split("T")[0],
                            display_format="YYYY-MM-DD", className="mb-2 d-block bg-dark"
                        )
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Start Hour", className="small text-light"),
                        dbc.Input(id="start-hour", type="number", min=0, max=23, value=22, size="sm",
                                  className="bg-dark text-light border-secondary")
                    ]),
                    dbc.Col([
                        dbc.Label("Stop Hour", className="small text-light"),
                        dbc.Input(id="end-hour", type="number", min=0, max=23, value=23, size="sm",
                                  className="bg-dark text-light border-secondary")
                    ]),
                ], className="mb-3"),
                dbc.Button("Validate & Download", id="btn-download", color="light", outline=True,
                           className="w-100 mb-3", size="sm", style={"fontWeight": "bold"}),
                dcc.Loading(
                    id="loading-download", type="circle", color="#0dcaf0",
                    children=html.Div(id="download-status", className="small text-success mb-3 fw-bold"),
                ),
            ], id="historic-controls-container")
        ]

    def _time_section(self) -> list:
        initial_max = max(len(self._store.filtered(DEFAULT_TIME_RANGE, "historic")) - 1, 0)
        return [
            html.Div(
                [
                    html.H6("Time Control", className="fw-bold mb-0", style={"margin-right": "5px"}),
                    # Small spinner to the right of the title, visible while the main callback
                    # generates a new image (sentinel is an output, so Dash marks it "loading").
                    dbc.Spinner(
                        html.Div(id="img-loading-sentinel"),
                        size="sm", color="info", type="border",
                        spinner_style={"width": "1rem", "height": "1rem", "borderWidth": "0.15rem"},
                    ),
                ],
                className="d-flex align-items-center gap-2 mb-2",
            ),
            dbc.Label(id="frame-label", children="Selected Frame: N/A", className="fw-bold text-light"),
            dcc.Slider(0, initial_max, 1, value=0, marks={}, id="frame-slider", className="mb-3"),
            html.Div([
                dbc.Row([
                    dbc.Col(dbc.Button("Play/Pause", id="btn-play", color="success", outline=True, className="w-100 fw-bold")),
                    dbc.Col(dbc.Button("Reset", id="btn-reset", color="danger", outline=True, className="w-100 fw-bold")),
                ])
            ], id="playback-controls-container"),
            html.Small(id="warmup-status", className="text-info d-block mt-2", style={"minHeight": "1.2rem"}),
        ]

    @staticmethod
    def _stores_and_intervals() -> list:
        return [
            dcc.Interval(id="animation-interval", interval=200, n_intervals=0, disabled=True),
            dcc.Store(id="is-processing", data=False),
            dcc.Store(id="active-time-range", data=DEFAULT_TIME_RANGE),
            dcc.Interval(id="live-polling-interval", interval=15 * 60 * 1000, n_intervals=0, disabled=True),
            dcc.Store(id="session-id", data=str(uuid.uuid4().hex)),
            dcc.Interval(id="warmup-poll", interval=1000, n_intervals=0),
        ]

    # ---- content -----------------------------------------------------------
    def _content(self) -> html.Div:
        return html.Div(
            [
                html.Div(id="final-report-div", className="mb-4"),
                dbc.Card(
                    dbc.CardBody([
                        html.Img(id="map-image", style={"width": "100%", "borderRadius": "5px"})
                    ]),
                    className="shadow-sm border-secondary bg-dark",
                    style={"margin-bottom": "20px"},
                ),
                html.H4("Volume", className="fw-bold mb-3", style={"fontSize": "1.2rem"}),
                dbc.Row([
                    dbc.Col(self._metric_card("Accumulated (Historic)", "val-historic-vol", "info", "info")),
                    dbc.Col(self._metric_card("Current (L/m²)", "val-current-vol", "info", "info")),
                    dbc.Col(self._metric_card("Predicted Volume", "val-predicted-vol", "info", "info")),
                    dbc.Col(self._metric_card("Max. ROI (L/m²)", "val-max-rain", "danger", "danger")),
                ], className="mb-3"),
                html.H4("Cells", className="fw-bold mb-3", style={"fontSize": "1.2rem"}),
                dbc.Row([
                    dbc.Col(self._metric_card("Tracked", "val-tracked", "secondary", "light")),
                    dbc.Col(self._metric_card("In ROI", "val-in-roi", "secondary", "light")),
                ]),
                html.Div(id="diagnostics-div", className="mb-4", style={"margin-top": "20px"}),
            ],
            className="p-4 bg-dark text-light h-100",
            style={"minHeight": "100vh"},
        )

    @staticmethod
    def _metric_card(title: str, value_id: str, border_color: str = "primary", text_color: str = "light") -> dbc.Card:
        return dbc.Card(
            dbc.CardBody([
                html.H6(title, className="text-muted text-uppercase small mb-1"),
                html.H3("N/A", id=value_id, className=f"text-{text_color} mb-0 fw-bold"),
            ]),
            className=f"border-{border_color} bg-dark text-light shadow-sm h-100",
        )
