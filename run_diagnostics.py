import os
import glob
import numpy as np

from src.core.storm_tracker import StormTracker
from src.core.matcher import Matcher
from src.core.frame_processor import FrameProcessor
from src.io.frame_preprocessor import compute_geometry, preprocess
from src.diagnostics.autopsier import Autopsier

def run_evaluation_pipeline(data_folder: str, forecast_horizon_steps: int = 2):
    """
    Rulează pipeline-ul complet pe episoade și generează diagnosticul.
    forecast_horizon_steps: 2 înseamnă 30 minute (dacă pasul e 15m).
    """
    files = sorted(glob.glob(os.path.join(data_folder, "*.nc")))
    if not files:
        print("Nu s-au găsit fișiere .nc în", data_folder)
        return
        
    tracker = StormTracker()
    matcher = Matcher()
    
    # Construim bbox pentru preprocesor (întreaga țară/regiune)
    # Acestea sunt coordonate arbitrare, vor acoperi Romania in general, sau poti restrictiona daca e nevoie.
    lon_min, lon_max = 19.0, 30.0
    lat_min, lat_max = 43.0, 49.0
    center_lat, center_lon = 46.0, 25.0
    radius_km = 800.0
    bbox = (lon_min, lon_max, lat_min, lat_max)
    
    # 1. Calculam geometria o singura data
    print("Calculam geometria...")
    geom = compute_geometry(files[0], bbox, (center_lat, center_lon), radius_km)
    if geom is None:
        print("Eroare la calcularea geometriei.")
        return
        
    print(f"Incepem evaluarea pentru {len(files)} cadre. Orizont de predictie: +{forecast_horizon_steps * 15} min")
    
    predictions_history = {} # frame_idx -> predicted cells
    
    for idx, filepath in enumerate(files):
        print(f"\n--- Procesam T0: {os.path.basename(filepath)} ({idx+1}/{len(files)}) ---")
        
        # 2. Citim matricea reala de ploaie folosind parser-ul H-SAF din proiect
        prep = preprocess(filepath, geom, bbox)
        if prep is None:
            print("Eroare la citirea/preprocesarea fisierului.")
            continue
            
        actual_cells_T0 = prep.filtered_cells
        
        # 3. Rulam Tracker-ul pentru a obtine predictiile de la T0 inspre viitor
        predictions_queue = []
        result = FrameProcessor.process(prep, geom, tracker, predictions_queue)
        
        # Salvam predictiile facute ACUM (T0) pentru momentul viitor (T0 + forecast_horizon_steps)
        # Atentie: Tracker-ul tocmai s-a actualizat cu noile celule reale.
        # Va trebui sa apelam functia sa de extrapolare interna sau sa luam direct din predictions_queue daca le populeaza.
        # Când Extrapolate ruleaza în sistemul complet (pe matrici pixel), celulele urmărite își obțin volumele viitoare.
        # Pentru diagnoză vectoriala, clonăm starea celulelor si aplicăm direct miscarea cinematică din Kalman.
        # Deci noi putem doar clona starea celulelor si aplica miscarea kinematica
        predicted_cells_future = []
        for c in tracker._previous_cells:
            if not c.is_tracked:
                continue
            import copy
            future_c = copy.deepcopy(c)
            # Aplicam 1 salt
            step = forecast_horizon_steps
            gamma = 0.8
            term_a = (2*step - (1+gamma)*(1-gamma**step)/(1-gamma)) / (2*(1-gamma))
            future_c.centroid_x += future_c.v_x * step + future_c.a_x * term_a
            future_c.centroid_y += future_c.v_y * step + future_c.a_y * term_a
            
            from src.core.algorithms_config import config as algo_config
            phase = getattr(future_c, 'lifecycle_phase', 'MATURITY')
            if getattr(algo_config, 'ENABLE_THERMODYNAMIC_DECAY', True):
                curve = algo_config.DECAY_CURVES.get(phase, algo_config.DECAY_CURVES["MATURITY"])
                lookup_step = min(max(0, step), len(curve) - 1)
                max_growth = curve[lookup_step]
            else:
                max_growth = 1.0
            
            future_c.predicted_area_kalman *= max_growth
            predicted_cells_future.append(future_c)
            
        predictions_history[idx + forecast_horizon_steps] = predicted_cells_future
        
        # --- VERIFICARE DIAGNOSTIC (Autopsier) ---
        if idx in predictions_history:
            predicted_cells_past = predictions_history[idx]
            
            # Match între ce a prezis trecutul și ce e real acum la T0
            # Matcher.match_cells returneaza {index_in_current_cells : index_in_previous_cells}
            matches = Matcher.match_cells(
                current_cells=actual_cells_T0,
                previous_cells=predicted_cells_past,
                kalman_bank=tracker._kinematic_updater._kalman_bank,
                max_dist_pixels=30
            )
            
            iou_matches = {}
            for c_idx, p_idx in matches.items():
                iou_matches[predicted_cells_past[p_idx].id] = actual_cells_T0[c_idx].id
                
            report = Autopsier.evaluate(
                step=forecast_horizon_steps,
                predicted_cells=predicted_cells_past,
                actual_cells=actual_cells_T0,
                iou_matches=iou_matches
            )
            
            # Afisare Autopsie
            print(f">>> REZULTAT AUTOPSIE (Predictia pentru T0, lansata acum {forecast_horizon_steps * 15} minute):")
            print(f"    - Furtuni Fantoma (False Alarms): {len(report.false_alarms)}")
            print(f"    - Furtuni Ratate (Missed): {len(report.missed_cells)}")
            
            if report.worst_cells:
                print("    - TOP 3 CELE MAI GRAVE ERORI (Root Cause Analysis):")
                for diag in report.worst_cells[:3]:
                    print(f"      * Celula {diag.cell_id}: Eroare Volum {diag.vol_error_pct:.1f}%")
                    print(f"        Cauza Principala: {diag.cluster.name}")
                    if diag.cluster.name != "NONE":
                        print(f"        Vinovati: Advectie({diag.attribution.advection_pct}%), Decay({diag.attribution.decay_pct}%), Birth({diag.attribution.birth_pct}%)")

if __name__ == "__main__":
    # Citim datele gata pregatite din /data/raw
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
    run_evaluation_pipeline(DATA_DIR, forecast_horizon_steps=2)
