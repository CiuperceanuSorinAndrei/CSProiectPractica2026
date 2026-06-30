"""Constructia structurii UI (sidebar + continut). Stilurile dark traiesc in assets/custom.css."""
import uuid
from dash import dcc, html
import dash_bootstrap_components as dbc

from config import PREDEFINED_LOCATIONS
from src.dashboard.constants import DEFAULT_TIME_RANGE, MAP_ZOOM_DEFAULT, ROI_RADIUS_DEFAULT
from src.dashboard.frame_store import FrameStore


class DashboardLayout:
    """Construieste structura UI (sidebar + continut) din metode mici, organizate."""

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
                html.H4("Estimarea volumului de precipitații din produse satelitare",
                        className="text-info fw-bold mb-3", style={"fontSize": "1.2rem"}),
                html.Hr(),
                *self._server_config_section(),
                *self._run_mode_section(),
                *self._roi_section(),
                *self._ingestion_section(),
                html.Hr(),
                *self._time_section(),
                *self._stores_and_intervals(),
            ],
            className="bg-dark text-light p-4 shadow-sm border-end border-secondary app-sidebar",
        )

    @staticmethod
    def _server_config_section() -> list:
        """Campuri editabile pentru server/foldere/credentiale/format, deasupra tuturor.
        Pre-completate din setarile persistate (server_settings.json) sau implicitele din config."""
        from src.io.server_settings import ServerSettings
        s = ServerSettings.load()
        inp = {"className": "mb-2 bg-dark text-light border-secondary", "size": "sm", "debounce": True}
        return [
            # Buton-antet care arata/ascunde tot blocul de configurare (pagina mai putin aglomerata).
            dbc.Button(
                "▸ Configurare Server",
                id="toggle-server-config", color="link", size="sm",
                className="fw-bold text-light text-decoration-none p-0 mb-2 shadow-none",
            ),
            dbc.Collapse(
                [
                    dbc.Label("Host FTP", className="small text-light mb-1"),
                    dbc.Input(id="srv-host", type="text", value=s.host, **inp),
                    dbc.Label("Director server (sursă)", className="small text-light mb-1"),
                    dbc.Input(id="srv-remote-dir", type="text", value=s.remote_dir, **inp),
                    dbc.Label("Director local (salvare)", className="small text-light mb-1"),
                    dbc.Input(id="srv-local-dir", type="text", value=s.local_dir, **inp),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Utilizator", className="small text-light mb-1"),
                            dbc.Input(id="srv-user", type="text", value="", **inp),
                        ]),
                        dbc.Col([
                            dbc.Label("Parolă", className="small text-light mb-1"),
                            dbc.Input(id="srv-pass", type="password", value="", **inp),
                        ]),
                    ]),
                    dbc.Label("Format fișier (strftime)", className="small text-light mb-1"),
                    dbc.Input(id="srv-format", type="text", value=s.file_format, **inp),
                    dbc.Label("Interval între cadre (min)", className="small text-light mb-1"),
                    dbc.Input(id="time-delta", type="number", value=s.time_delta, min=1, step=1, **inp),
                ],
                id="server-config-collapse", is_open=False,
            ),
            html.Hr(),
        ]

    @staticmethod
    def _run_mode_section() -> list:
        return [
            html.H6("Mod de Rulare", className="fw-bold"),
            dbc.RadioItems(
                id="run-mode-select",
                options=[{"label": "Istoric", "value": "historic"}, {"label": "LIVE", "value": "live"}],
                value="historic", inline=True, className="mb-4",
            ),
        ]

    @staticmethod
    def _roi_section() -> list:
        from src.geo.reservoir_loader import ReservoirLoader
        reservoirs = ReservoirLoader.get_all_reservoirs()
        res_options = [{"label": k, "value": k} for k in sorted(reservoirs.keys())]

        return [
            html.H6("Regiune de Interes (ROI)", className="fw-bold"),
            dbc.RadioItems(
                id="location-type",
                options=[
                    {"label": "Oraș (Cerc)", "value": "predefined"},
                    {"label": "Lac Acumulare (Poligon)", "value": "reservoir"}
                ],
                value="predefined",
                inline=False,
                className="mb-2 text-light"
            ),
            html.Div(
                id="predefined-loc-div",
                children=[
                    dbc.Label("Alege locație (Punct)"),
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
                    dbc.Label("Alege Lac (Contur Exact)"),
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
                            dbc.Label("Latitudine", className="text-light"),
                            dbc.Input(id="manual-lat", type="number", value=44.33, step=0.1,
                                      className="bg-dark text-light border-secondary"),
                        ]),
                        dbc.Col([
                            dbc.Label("Longitudine", className="text-light"),
                            dbc.Input(id="manual-lon", type="number", value=23.79, step=0.1,
                                      className="bg-dark text-light border-secondary"),
                        ]),
                    ], className="mb-3")
                ],
                style={"display": "none"},
            ),
            dbc.Label("Arie Vizualizare Hartă (km)", className="text-light"),
            dbc.Input(id="map-zoom-input", type="number", step=10, debounce=True, value=MAP_ZOOM_DEFAULT,
                      className="mb-3 bg-dark text-light border-secondary"),
            html.Div(
                id="radius-input-div",
                children=[
                    dbc.Label("Rază Circulară (km) - Volum", className="text-light"),
                    dbc.Input(id="roi-radius-input", type="number", step=1, debounce=True, value=ROI_RADIUS_DEFAULT,
                              className="mb-4 bg-dark text-light border-secondary"),
                ]
            ),
            html.Div(id="input-warnings"),
        ]

    @staticmethod
    def _ingestion_section() -> list:
        return [
            html.Div([
                html.H6("Achiziție Date Istorice", className="fw-bold"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Data Start", className="small text-light"),
                        dcc.DatePickerSingle(
                            id="start-date", date=DEFAULT_TIME_RANGE["start"].split("T")[0],
                            display_format="YYYY-MM-DD", className="mb-2 d-block bg-dark"
                        )
                    ]),
                    dbc.Col([
                        dbc.Label("Data Stop", className="small text-light"),
                        dcc.DatePickerSingle(
                            id="end-date", date=DEFAULT_TIME_RANGE["end"].split("T")[0],
                            display_format="YYYY-MM-DD", className="mb-2 d-block bg-dark"
                        )
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Ora Start", className="small text-light"),
                        dbc.Input(id="start-hour", type="number", min=0, max=23, value=22, size="sm",
                                  className="bg-dark text-light border-secondary")
                    ]),
                    dbc.Col([
                        dbc.Label("Ora Stop", className="small text-light"),
                        dbc.Input(id="end-hour", type="number", min=0, max=23, value=23, size="sm",
                                  className="bg-dark text-light border-secondary")
                    ]),
                ], className="mb-3"),
                dbc.Button("Validează & Descarcă", id="btn-download", color="light", outline=True,
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
                    html.H6("Control Timp", className="fw-bold mb-0", style={"margin-right": "5px"}),
                    # Spinner mic la dreapta titlului, vizibil cat timp callback-ul principal
                    # genereaza o imagine noua (sentinel-ul e output, deci Dash il marcheaza "loading").
                    # borderWidth subtire -> inel circular curat.
                    dbc.Spinner(
                        html.Div(id="img-loading-sentinel"),
                        size="sm", color="info", type="border",
                        spinner_style={"width": "1rem", "height": "1rem", "borderWidth": "0.15rem"},
                    ),
                ],
                className="d-flex align-items-center gap-2 mb-2",
            ),
            dbc.Label(id="frame-label", children="Cadru Selectat: N/A", className="fw-bold text-light"),
            dcc.Slider(0, initial_max, 1, value=0, marks={}, id="frame-slider", className="mb-3"),
            html.Div([
                dbc.Row([
                    dbc.Col(dbc.Button("Play/Pauză", id="btn-play", color="success", outline=True, className="w-100 fw-bold")),
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
                html.H4("Volum", className="fw-bold mb-3", style={"fontSize": "1.2rem"}),
                dbc.Row([
                    dbc.Col(self._metric_card("Acumulat (Istoric)", "val-historic-vol", "info", "info")),
                    dbc.Col(self._metric_card("Aport Curent (15m)", "val-current-vol", "info", "info")),
                    dbc.Col(self._metric_card("Anticipat (Rata/15m)", "val-predicted-vol", "info", "info")),
                    dbc.Col(self._metric_card("Rată Maximă (mm/h)", "val-max-rain", "danger", "danger")),
                ], className="mb-3"),
                html.H4("Performanță Predicție (CSI/FAR/POD/FSS)", className="fw-bold mb-3", style={"fontSize": "1.2rem"}),
                dbc.Row([
                    dbc.Col(self._metric_card("Orizont +30 min", "val-metrics-30m", "success", "success")),
                    dbc.Col(self._metric_card("Orizont +1 oră", "val-metrics-1h", "success", "success")),
                    dbc.Col(self._metric_card("Orizont +2 ore", "val-metrics-2h", "success", "success")),
                    dbc.Col(self._metric_card("Scor Mediu", "val-metrics-total", "success", "success")),
                ], className="mb-4"),
                html.H4("Celule", className="fw-bold mb-3", style={"fontSize": "1.2rem"}),
                dbc.Row([
                    dbc.Col(self._metric_card("Urmărite", "val-tracked", "secondary", "light")),
                    dbc.Col(self._metric_card("În ROI", "val-in-roi", "secondary", "light")),
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
