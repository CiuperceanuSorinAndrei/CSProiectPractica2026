import os
import sys
from datetime import datetime, timezone

# Asiguram ca directorul src poate fi gasit
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dashboard.frame_store import FrameStore
from src.dashboard.session_manager import SessionManager
from src.dashboard.constants import DATA_DIR

def run_simulation(location_name: str, bbox: tuple[float, float, float, float], center: tuple[float, float], start_str: str, end_str: str):
    store = FrameStore(DATA_DIR, "h60_%Y%m%d_%H%M_fdk.nc.gz")
    
    time_range = {"start": start_str, "end": end_str}
    files = store.filtered(time_range, run_mode="historic")
    
    print(f"Running simulation for {location_name} from {start_str} to {end_str} with {len(files)} files.")
    
    if not files:
        print("No files found!")
        return
        
    session_manager = SessionManager()
    session_id = f"test_{location_name}"
    
    # Rulam primul cadru pentru initializare
    res = session_manager.process_to_frame(
        session_id=session_id,
        frame_idx=0,
        nc_files=files,
        bbox=bbox,
        center=center,
        radius_km=30.0,
        run_mode="historic",
        time_range=time_range,
        store=store
    )
    
    # Rulam toate cadrele folosind _accumulate_range (acelasi flux ca la salt)
    orch, hist = session_manager.get_state(session_id)
    
    for i in range(1, len(files)):
        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} frames...")
        res = session_manager.process_to_frame(
            session_id=session_id,
            frame_idx=i,
            nc_files=files,
            bbox=bbox,
            center=center,
            radius_km=30.0,
            run_mode="historic",
            time_range=time_range,
            store=store
        )
        
    print(f"--- Simulation Complete for {location_name} ---")
    
    real_vol = hist.total_map_mm
    print("\nAcumulare Precipitatii Bazin")
    print("Orizont | Realizat (L/m²) | Prezis (L/m²) | Bias (%)")
    for horizon in ["15m", "1h", "2h"]:
        pred_vol = hist.predicted_volume_accumulation.get(horizon, 0.0)
        bias = ((pred_vol - real_vol) / real_vol * 100.0) if real_vol > 0 else 0.0
        print(f"{horizon:7} | {real_vol:15.2f} | {pred_vol:13.2f} | {bias:+.1f}%")
        
    metrics = hist.get_reliability_metrics()
    print("\nIncredere Avertizari Bazin (Acuratete Volum Cumulat)")
    for t in [1.0, 5.0]:
        print(f"Acumulare > {t} L/m2:")
        print("Orizont | POD | FAR | CMAPE")
        for horizon in ["15m", "1h", "2h"]:
            m = metrics[t][horizon]
            print(f"{horizon:7} | {m['pod']:3.0f}% | {m['far']:3.0f}% | ±{m['cmae']:.1f}%")
    print("\n")

if __name__ == "__main__":
    locations = [
        {"name": "Craiova", "bbox": (23.750, 23.850, 44.250, 44.350), "center": (44.310, 23.800)},
        {"name": "Portile de Fier", "bbox": (22.500, 22.600, 44.600, 44.700), "center": (44.670, 22.530)},
        {"name": "Vidraru", "bbox": (24.580, 24.700, 45.360, 45.450), "center": (45.360, 24.580)}
    ]
    # Define timeframes
    timeframes = {
        "Short": ("2026-06-13T22:00:00", "2026-06-14T23:00:00"),
        "Medium": ("2026-06-10T01:00:00", "2026-06-14T23:00:00")
    }
    
    for tf_name, (start_time, end_time) in timeframes.items():
        print(f"\n========================================")
        print(f"RUNNING TIMEFRAME: {tf_name}")
        print(f"========================================\n")
        for loc in locations:
            run_simulation(loc["name"], loc["bbox"], loc["center"], start_time, end_time)
