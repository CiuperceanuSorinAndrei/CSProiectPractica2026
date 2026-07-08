import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.core.constants import HORIZON_NAMES
from src.dashboard.constants import DATA_DIR
from src.dashboard.frame_store import FrameStore
from src.dashboard.session_manager import SessionManager
from src.geo.reservoir_loader import ReservoirLoader

# Configuration
TARGET_RESERVOIRS = ("Vidraru", "Portile De Fier I", "Izvorul Muntelui", "Gura Apelor", "Tarnita", "Somesu Cald")

def build_target_locations() -> list[dict]:
    # 1. Base locations
    locs = [{"name": "Craiova", "bbox": (23.750, 23.850, 44.250, 44.350), "center": (44.310, 23.800), "radius_km": 30.0, "polygon": None}]
    reservoirs = ReservoirLoader.get_all_reservoirs()
    
    missing = [n for n in TARGET_RESERVOIRS if n not in reservoirs]
    if missing: raise RuntimeError(f"Missing reservoirs: {', '.join(missing)}")
        
    for name in TARGET_RESERVOIRS:
        res = reservoirs[name]
        min_lon, min_lat, max_lon, max_lat = res["bounds"]
        locs.append({"name": name, "bbox": (min_lon, max_lon, min_lat, max_lat), "center": res["center"], "radius_km": res["radius_km"], "polygon": res["polygon"]})
    return locs

def run_simulation(loc: str, bbox: tuple, center: tuple, start: str, end: str, poly=None, r_km: float=30.0, quiet: bool=False):
    # 2. Setup session
    store = FrameStore(DATA_DIR, "h60_%Y%m%d_%H%M_fdk.nc.gz")
    tr = {"start": start, "end": end}
    files = store.filtered(tr, run_mode="historic")

    if not files:
        if not quiet: print("No files found!")
        return None

    sm = SessionManager()
    sid = f"test_{loc}_{start}_{end}"

    # 3. Process frames
    for i in range(len(files)):
        sm.process_to_frame(sid, i, files, bbox, center, r_km, poly, "historic", tr, store)

    orch, hist = sm.get_state(sid)
    actual, predicted = {}, {}

    for h in HORIZON_NAMES:
        rv, pv = hist.volume_sums(h)
        actual[h], predicted[h] = rv, pv

    if not quiet:
        print(f"\n--- {loc} ({start} to {end}) ---")
        for h in HORIZON_NAMES:
            bias = ((predicted[h] - actual[h]) / actual[h] * 100) if actual[h] > 0 else 0
            print(f"{h:7} | Real: {actual[h]:10.2f} | Pred: {predicted[h]:10.2f} | {bias:+.1f}%")
            
    orch.stop_warmup()
    return {"actual": actual, "predicted": predicted}

def run_rolling_validation(locations: list[dict], start_day: int=1, end_day: int=26):
    # 4. Rolling validation
    by_loc = {l["name"]: {h: {"actual": 0.0, "predicted": 0.0} for h in HORIZON_NAMES} for l in locations}
    totals = {h: {"actual": 0.0, "predicted": 0.0} for h in HORIZON_NAMES}
    cases = 0

    for day in range(start_day, end_day + 1):
        s, e = f"2026-06-{day:02d}T00:00:00", f"2026-06-{day:02d}T23:59:59"
        print(f"Rolling {s} -> {e}")
        
        for l in locations:
            res = run_simulation(l["name"], l["bbox"], l["center"], s, e, l.get("polygon"), l.get("radius_km", 30.0), True)
            if not res: continue
            
            cases += 1
            for h in HORIZON_NAMES:
                totals[h]["actual"] += res["actual"][h]
                totals[h]["predicted"] += res["predicted"][h]
                by_loc[l["name"]][h]["actual"] += res["actual"][h]
                by_loc[l["name"]][h]["predicted"] += res["predicted"][h]

    print("\nAggregate validation:")
    for h in HORIZON_NAMES:
        a, p = totals[h]["actual"], totals[h]["predicted"]
        bias = ((p - a) / a * 100) if a > 0 else 0
        print(f"{cases:5d} cases | {h:7} | Real: {a:10.2f} | Pred: {p:10.2f} | {bias:+6.1f}% | {'PASS' if -15 < bias < 15 else 'FAIL'}")

def main():
    # 5. Initialization
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling-validation", action="store_true")
    args = parser.parse_args()

    locs = build_target_locations()

    if args.rolling_validation:
        run_rolling_validation(locs)
        sys.exit(0)

    timeframes = {"Short": ("2026-06-13T22:00:00", "2026-06-14T23:00:00"), "Medium": ("2026-06-10T01:00:00", "2026-06-14T23:00:00")}

    for tf_name, (start, end) in timeframes.items():
        print(f"\n=== {tf_name.upper()} ===")
        for l in locs:
            run_simulation(l["name"], l["bbox"], l["center"], start, end, l.get("polygon"), l.get("radius_km", 30.0))

if __name__ == "__main__":
    main()
