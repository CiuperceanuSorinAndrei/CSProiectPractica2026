"""Builder pentru metricile si rapoartele HTML din Dashboard."""
from dash import html
import dash_bootstrap_components as dbc
from src.dashboard.session_manager import SessionManager
from src.geo.reservoir_fill import ReservoirFillEstimator


class ReportBuilder:

    @staticmethod
    def _avg_metric(hist, key: str, horizon: str) -> float:
        """Media valorilor strict pozitive pentru o metrica (csi/far/pod/fss) la un orizont dat."""
        vals = [m.get(horizon, 0) for m in hist.metrics_history[key] if m.get(horizon, 0) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def _with_fill_pct(value_str: str, map_mm: float, reservoir: dict | None):
        """Ataseaza sub valoarea afisata (L/m²) procentul din volumul maxim al lacului.

        Cand nu e selectat un lac (mod oras/cerc) sau ii lipseste capacitatea, intoarce doar
        textul original -> cardul ramane neschimbat in acele moduri.
        """
        pct = ReservoirFillEstimator.fill_percentage_for(map_mm, reservoir)
        if pct is None:
            return value_str
        return [
            value_str,
            html.Div(f"{pct:.2f}% din volum maxim", className="small text-muted fw-normal mt-1"),
        ]

    @staticmethod
    def format_metrics(session_id: str, result, session_manager: SessionManager, reservoir: dict | None = None):
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

        # Sub metricile de volum (L/m²) afisam procentul din volumul maxim al lacului selectat.
        # Istoricul foloseste MAP-ul acumulat; curentul/anticipatul folosesc valoarea proprie.
        hist_vol_str = ReportBuilder._with_fill_pct(hist_vol_str, hist.total_map_mm, reservoir)
        curr_vol_str = ReportBuilder._with_fill_pct(curr_vol_str, curr_vol, reservoir)
        pred_vol_str = ReportBuilder._with_fill_pct(pred_vol_str, vols.get("1h", 0.0), reservoir)

        max_rain_str = f"{result.max_rain:.1f}"

        def _fmt(horizon):
            c = ReportBuilder._avg_metric(hist, "csi", horizon)
            f = ReportBuilder._avg_metric(hist, "far", horizon)
            if c == 0.0 and f == 0.0:
                return "Așteptare..."

            p = ReportBuilder._avg_metric(hist, "pod", horizon)
            fs = ReportBuilder._avg_metric(hist, "fss", horizon)
            return (
                f"CSI: {c:.2f} | FAR: {f:.2f}\n"
                f"POD: {p:.2f} | FSS: {fs:.2f}"
            )

        m_30m = _fmt("15m")
        m_1h = _fmt("1h")
        m_2h = _fmt("2h")

        all_csis = []
        for h in ["15m", "1h", "2h"]:
            all_csis.extend([m.get(h, 0) for m in hist.metrics_history["csi"] if m.get(h, 0) > 0])
        avg_csi = sum(all_csis)/len(all_csis) if all_csis else 0.0
        m_tot = f"CSI Mediu: {avg_csi:.2f}"

        tracked = f"{result.num_tracked} Active"
        in_roi = "Nu"
        if curr_vol > 0:
            in_roi = "Da (Ploaie detectată)"

        return hist_vol_str, curr_vol_str, pred_vol_str, max_rain_str, m_30m, m_1h, m_2h, m_tot, tracked, in_roi

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
                    direction = "Staționar"
                else:
                    angle = math.degrees(math.atan2(-vy, vx)) # -vy because image Y is down
                    if angle < 0: angle += 360
                    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
                    direction = dirs[round(angle / 45) % 8]
                
                # Trend
                trend = cell.get("volume_trend", 1.0)
                if trend > 1.05:
                    trend_str = "În Creștere 📈"
                elif trend < 0.95:
                    trend_str = "În Scădere 📉"
                else:
                    trend_str = "Stabilă ➖"
                    
                # Phase
                phase = cell.get("lifecycle_phase", "MATURITY")
                phase_map = {"FORMATION": "Formare", "MATURITY": "Maturitate", "DISSIPATION": "Disipare"}
                phase_ro = phase_map.get(phase, phase)
                
                rows.append(html.Tr([
                    html.Td(short_id),
                    html.Td(f"{speed_kmh:.0f} km/h"),
                    html.Td(direction),
                    html.Td(phase_ro),
                    html.Td(trend_str),
                ]))
        if not rows:
            return html.Div(html.I("Nu există celule active în acest moment."), className="text-muted small")
        return html.Div([
            html.H6("Telemetrie Furtuni Active", className="fw-bold text-primary"),
            dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("ID Furtună"),
                        html.Th("Viteză"),
                        html.Th("Direcție"),
                        html.Th("Stadiu"),
                        html.Th("Evoluție Intensitate"),
                    ])),
                    html.Tbody(rows),
                ],
                bordered=True, hover=True, color="dark",
                className="kalman-diag mb-0",
            ),
        ])

    @staticmethod
    def _build_kinematic_rows(hist) -> list:
        """Randuri tabel cu mediile CSI/FAR/POD/FSS pe orizonturi (performanta cinematica)."""
        rows = []
        for horizon in ["15m", "1h", "2h"]:
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
        """Randuri tabel cu MAP real vs prezis acumulat per orizont (aliniat corect in timp)."""
        vol_rows = []
        horizon_steps = {"15m": 2, "1h": 5, "2h": 9}

        for horizon in ["15m", "1h", "2h"]:
            steps = horizon_steps[horizon]

            # Daca nu avem destule cadre pentru a alinia orizontul, trecem peste sau punem 0
            if len(hist.true_volumes) > steps and len(hist.pred_volumes[horizon]) > steps:
                # Volumul real este suma de la pasul 'steps' pana la final
                aligned_true = hist.true_volumes[steps:]
                # Volumul prezis este suma prezicerilor facute cu 'steps' in urma, pentru cadrele de azi
                aligned_pred = hist.pred_volumes[horizon][:-steps]

                vol_real_sum = sum(aligned_true)
                vol_pred_sum = sum(aligned_pred)
            else:
                vol_real_sum = hist.total_map_mm
                vol_pred_sum = hist.predicted_volume_accumulation.get(horizon, 0.0)

            delta_pct = ((vol_pred_sum - vol_real_sum) / vol_real_sum * 100.0) if vol_real_sum > 0 else 0.0

            vol_rows.append(html.Tr([
                html.Td(horizon), html.Td(f"{vol_real_sum:.0f}"),
                html.Td(f"{vol_pred_sum:.0f}"), html.Td(f"{delta_pct:+.1f}%")
            ]))
        return vol_rows

    @staticmethod
    def build_final_report(session_id: str, run_mode: str, frame_idx: int, total_frames: int, session_manager: SessionManager) -> html.Div | None:
        if run_mode == "live" or frame_idx < total_frames - 1:
            return None

        _, hist = session_manager.get_state(session_id)
        if hist.frames_processed == 0:
            return None

        rows = ReportBuilder._build_kinematic_rows(hist)
        vol_rows = ReportBuilder._build_volume_rows(hist)

        # Phase 6: Run the FAR Inspector at the very end of the simulation
        hist.generate_far_report()

        return html.Div([
            dbc.Alert(
                [
                    html.H4("Simulare Istorică Încheiată", className="alert-heading fw-bold"),
                    html.P("Raport agregat de performanță la finalul episodului selectat:"),
                    html.Hr(),
                    html.H6("Acuratețe Volumetrică", className="fw-bold mt-3"),
                    dbc.Table([
                        html.Thead(html.Tr([html.Th("Orizont"), html.Th("Volum Real (mii m³)"), html.Th("Volum Prezis (mii m³)"), html.Th("Eroare (Delta %)")])),
                        html.Tbody(vol_rows)
                    ], bordered=True, color="dark", hover=True, size="sm", className="mb-4"),
                    html.H6("Performanță Cinematică (Medii)", className="fw-bold"),
                    dbc.Table(
                        [
                            html.Thead(html.Tr([html.Th("Orizont"), html.Th("CSI"), html.Th("FAR"), html.Th("POD"), html.Th("FSS")])),
                            html.Tbody(rows)
                        ],
                        bordered=True, color="dark", hover=True, size="sm"
                    )
                ],
                color="success",
            )
        ])
