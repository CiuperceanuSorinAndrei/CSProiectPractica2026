"""Builder pentru metricile si rapoartele HTML din Dashboard."""
from dash import html
import dash_bootstrap_components as dbc
from src.dashboard.session_manager import SessionManager


class ReportBuilder:
    
    @staticmethod
    def format_metrics(session_id: str, result, session_manager: SessionManager):
        _, hist = session_manager.get_state(session_id)
        
        hist_vol_str = f"{hist.total_volume_m3 / 1000.0:.1f} mii"
        
        curr_vol = result.roi_volume_m3
        curr_vol_str = f"{curr_vol / 1000.0:.1f} mii"

        vols = result.predicted_volumes_horizons
        pred_vol_str = (
            f"30m: {vols.get('30m', 0)/1000.0:.0f} | "
            f"1h: {vols.get('1h', 0)/1000.0:.0f} | "
            f"2h: {vols.get('2h', 0)/1000.0:.0f} mii"
        )
        
        max_rain_str = f"{result.max_rain:.1f}"

        csi = result.global_csi
        far = result.global_far
        pod = result.global_pod
        fss = result.global_fss

        def _fmt(horizon):
            csis = [m.get(horizon, 0) for m in hist.metrics_history["csi"] if m.get(horizon, 0) > 0]
            fars = [m.get(horizon, 0) for m in hist.metrics_history["far"] if m.get(horizon, 0) > 0]
            pods = [m.get(horizon, 0) for m in hist.metrics_history["pod"] if m.get(horizon, 0) > 0]
            fsss = [m.get(horizon, 0) for m in hist.metrics_history["fss"] if m.get(horizon, 0) > 0]
            
            if not csis and not fars:
                return "Așteptare..."
                
            c = sum(csis)/len(csis) if csis else 0.0
            f = sum(fars)/len(fars) if fars else 0.0
            p = sum(pods)/len(pods) if pods else 0.0
            fs = sum(fsss)/len(fsss) if fsss else 0.0
            
            return (
                f"CSI: {c:.2f} | FAR: {f:.2f}\n"
                f"POD: {p:.2f} | FSS: {fs:.2f}"
            )

        m_30m = _fmt("30m")
        m_1h = _fmt("1h")
        m_2h = _fmt("2h")

        all_csis = []
        for h in ["30m", "1h", "2h"]:
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
        diagnostics = []
        for cell in tracked_cells:
            if cell.get("is_tracked", False):
                err = cell.get("prediction_error_pixels", 0.0)
                v_err = cell.get("size_error_percent", 0.0)
                short_id = str(cell.get('cell_id', '???'))[:4]
                diagnostics.append(
                    html.Li(f"Celulă {short_id}: Eroare deplasare = {err:.1f} px | "
                            f"Viteză ({cell.get('v_x',0):.1f}, {cell.get('v_y',0):.1f}) | "
                            f"Accelerație ({cell.get('a_x',0):.2f}, {cell.get('a_y',0):.2f}) | "
                            f"Deviație Volum = {v_err:.1f}%")
                )
        if not diagnostics:
            return html.Div(html.I("Nu există celule urmărite în acest cadru."), className="text-muted small")
        return html.Div([
            html.H6("Diagnoză Filtru Kalman (Constant Acceleration)", className="fw-bold"),
            html.Ul(diagnostics, className="small text-warning", style={"listStyleType": "square"}),
        ])

    @staticmethod
    def build_final_report(session_id: str, run_mode: str, frame_idx: int, total_frames: int, session_manager: SessionManager) -> html.Div | None:
        if run_mode == "live" or frame_idx < total_frames - 1:
            return None

        _, hist = session_manager.get_state(session_id)
        if hist.frames_processed == 0:
            return None

        # Calculam tabelul de performanta
        rows = []
        for horizon in ["30m", "1h", "2h"]:
            csis = [m.get(horizon, 0) for m in hist.metrics_history["csi"] if m.get(horizon, 0) > 0]
            fars = [m.get(horizon, 0) for m in hist.metrics_history["far"] if m.get(horizon, 0) > 0]
            pods = [m.get(horizon, 0) for m in hist.metrics_history["pod"] if m.get(horizon, 0) > 0]
            fsss = [m.get(horizon, 0) for m in hist.metrics_history["fss"] if m.get(horizon, 0) > 0]
            
            c = sum(csis)/len(csis) if csis else 0.0
            f = sum(fars)/len(fars) if fars else 0.0
            p = sum(pods)/len(pods) if pods else 0.0
            fs = sum(fsss)/len(fsss) if fsss else 0.0
            
            rows.append(html.Tr([
                html.Td(horizon), html.Td(f"{c:.2f}"), html.Td(f"{f:.2f}"), 
                html.Td(f"{p:.2f}"), html.Td(f"{fs:.2f}")
            ]))

        # Calculam integrarea volumetrica totala per orizont aliniata corect in timp
        vol_rows = []
        
        horizon_steps = {"30m": 2, "1h": 4, "2h": 8}
        
        for horizon in ["30m", "1h", "2h"]:
            steps = horizon_steps[horizon]
            
            # Daca nu avem destule cadre pentru a alinia orizontul, trecem peste sau punem 0
            if len(hist.true_volumes) > steps and len(hist.pred_volumes[horizon]) > steps:
                # Volumul real este suma de la pasul 'steps' pana la final
                aligned_true = hist.true_volumes[steps:]
                # Volumul prezis este suma prezicerilor facute cu 'steps' in urma, pentru cadrele de azi
                aligned_pred = hist.pred_volumes[horizon][:-steps]
                
                vol_real_sum = sum(aligned_true) / 1000.0
                vol_pred_sum = sum(aligned_pred) / 1000.0
            else:
                vol_real_sum = hist.total_volume_m3 / 1000.0
                vol_pred_sum = hist.predicted_volume_accumulation.get(horizon, 0.0) / 1000.0
            
            delta_pct = ((vol_pred_sum - vol_real_sum) / vol_real_sum * 100.0) if vol_real_sum > 0 else 0.0
            
            vol_rows.append(html.Tr([
                html.Td(horizon), html.Td(f"{vol_real_sum:.0f}"), 
                html.Td(f"{vol_pred_sum:.0f}"), html.Td(f"{delta_pct:+.1f}%")
            ]))

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
