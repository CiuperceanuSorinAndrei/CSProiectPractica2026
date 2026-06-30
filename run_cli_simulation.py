import os
import glob
from src.dashboard.session_manager import SessionManager
from src.core.frame_processor import FrameProcessor

def main():
    print("Initializing DataService and discovering files...")
    # Get all .nc files
    data_dir = os.path.join("data", "raw")
    nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    # Limit to 150 files for a deeper diagnostic run
    nc_files = nc_files[:150]
    if not nc_files:
        print("No files found!")
        return

    # Use a dummy store that just returns the path directly
    class DummyStore:
        def path(self, f):
            return f
            
    store = DummyStore()
    sm = SessionManager()
    session_id = "cli_diag"
    
    bbox = (19.0, 30.0, 43.0, 49.0)
    center = (46.0, 25.0)
    radius_km = 800.0
    
    print(f"Running simulation on {len(nc_files)} frames...")
    
    for i, file_path in enumerate(nc_files):
        print(f"Processing frame {i+1}/{len(nc_files)}: {os.path.basename(file_path)}")
        try:
            sm.process_to_frame(
                session_id=session_id,
                frame_idx=i,
                nc_files=nc_files,
                bbox=bbox,
                center=center,
                radius_km=radius_km,
                run_mode="historic",
                time_range=None,
                store=store
            )
        except Exception as e:
            print(f"Error at frame {i}: {e}")

    _, hist = sm.get_state(session_id)
    
    print("\n" + "="*50)
    print(" FINAL PERFORMANCE REPORT ")
    print("="*50)
    
    vol_real = hist.total_volume_m3 / 1000.0
    
    for horizon in ["30m", "1h", "2h"]:
        vol_pred = hist.predicted_volume_accumulation.get(horizon, 0.0) / 1000.0
        delta = ((vol_pred - vol_real) / vol_real * 100) if vol_real > 0 else 0.0
        print(f"Horizon {horizon}: Real Vol {vol_real:.2f}, Pred Vol {vol_pred:.2f}, Delta {delta:+.3f}%")

    print("\n" + "-"*50)
    print(" PIXEL-BASED METRICS (CSI / FAR / POD) ")
    print("-"*50)
    
    from collections import defaultdict
    avg_csi = defaultdict(list)
    avg_far = defaultdict(list)
    avg_pod = defaultdict(list)
    
    for h in ["30m", "1h", "2h"]:
        csis = [m.get(h, 0) for m in hist.metrics_history["csi"] if m.get(h, 0) > 0]
        fars = [m.get(h, 0) for m in hist.metrics_history["far"] if m.get(h, 0) > 0]
        pods = [m.get(h, 0) for m in hist.metrics_history["pod"] if m.get(h, 0) > 0]
        c = sum(csis)/len(csis) if csis else 0.0
        f = sum(fars)/len(fars) if fars else 0.0
        p = sum(pods)/len(pods) if pods else 0.0
        print(f"Horizon {h}: CSI {c:.2f} | FAR {f:.2f} | POD {p:.2f}")

    print("\nRunning FAR Inspector...")
    hist.generate_far_report()

if __name__ == "__main__":
    main()
