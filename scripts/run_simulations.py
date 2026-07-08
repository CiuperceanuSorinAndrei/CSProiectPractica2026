import argparse
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.constants import HORIZON_NAMES
from src.dashboard.constants import DATA_DIR
from src.dashboard.frame_store import FrameStore
from src.dashboard.session_manager import SessionManager
from src.geo.reservoir_loader import ReservoirLoader

TARGET_RESERVOIRS = ("Vidraru", "Portile De Fier I", "Izvorul Muntelui", "Gura Apelor", "Tarnita", "Somesu Cald")
TARGET_PERIODS = {
    "Period 1": ("2026-06-01T00:00:00", "2026-06-12T16:00:00", False),
    "Period 2": ("2026-06-12T16:00:00", "2026-06-24T08:00:00", False),
    "Period 3": ("2026-06-24T08:00:00", "2026-07-06T00:00:00", True),
    "Full Period": ("2026-06-01T00:00:00", "2026-07-06T00:00:00", True),
}

def build_target_locations() -> list[dict]:
    locs = []
    pad = 0.05
    
    reservoirs = ReservoirLoader.get_all_reservoirs()
    missing = [name for name in TARGET_RESERVOIRS if name not in reservoirs]
    if missing:
        raise RuntimeError(f"Missing reservoirs: {', '.join(missing)}")
    
    for name in TARGET_RESERVOIRS:
        res = reservoirs[name]
        min_lon, min_lat, max_lon, max_lat = res["bounds"]
        locs.append({
            "name": name,
            "bbox": (min_lon - pad, max_lon + pad, min_lat - pad, max_lat + pad),
            "center": res["center"],
            "polygon": res.get("polygon"),
            "radius_km": res.get("radius_km", 15.0)
        })

    locs.insert(0, {
        "name": "Craiova",
        "bbox": (23.750 - pad, 23.850 + pad, 44.250 - pad, 44.350 + pad),
        "center": (44.310, 23.800),
        "radius_km": 30.0,
        "polygon": None,
    })

    return locs

def run_simulation(
    loc: str,
    bbox: tuple,
    center: tuple,
    start: str,
    end: str,
    poly=None,
    r_km: float=30.0,
    quiet: bool=False,
    include_end: bool=True,
):
    # 2. Setup session
    store = FrameStore(DATA_DIR, "h60_%Y%m%d_%H%M_fdk.nc")
    tr = {"start": start, "end": end}
    files = store.filtered(tr, run_mode="historic")
    if not include_end:
        files = [filename for filename in files if store.datetime(filename).isoformat() < end]

    if not files:
        if not quiet: print("No files found!")
        return None

    sm = SessionManager()
    sid = f"test_{loc}_{start}_{end}"

    # 3. Process frames
    for i in range(len(files)):
        sm.process_to_frame(sid, i, files, bbox, center, r_km, "historic", tr, store, polygon=poly)

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

def bias_pct(actual: float, predicted: float) -> float:
    if actual > 0.0:
        return (predicted - actual) / actual * 100.0
    return 0.0 if abs(predicted) < 1e-9 else float("inf")

def row_passes(actual: float, predicted: float) -> bool:
    return abs(predicted) < 1e-9 if actual <= 0.0 else -15.0 <= bias_pct(actual, predicted) <= 15.0

def run_target_validation(locations: list[dict], verbose: bool=True) -> list[dict]:
    rows = []
    for period_name, (start, end, include_end) in TARGET_PERIODS.items():
        if verbose:
            print(f"\n=== {period_name.upper()} ===", flush=True)
        for loc in locations:
            res = run_simulation(
                loc["name"], loc["bbox"], loc["center"], start, end,
                loc.get("polygon"), loc.get("radius_km", 30.0),
                quiet=True, include_end=include_end,
            )
            if not res:
                continue
            for horizon in HORIZON_NAMES:
                actual = res["actual"][horizon]
                predicted = res["predicted"][horizon]
                bias = bias_pct(actual, predicted)
                passed = row_passes(actual, predicted)
                rows.append({
                    "period": period_name,
                    "location": loc["name"],
                    "horizon": horizon,
                    "actual_mm": actual,
                    "predicted_mm": predicted,
                    "bias_pct": bias,
                    "pass": passed,
                })
                if verbose:
                    bias_text = "inf" if bias == float("inf") else f"{bias:+.1f}%"
                    print(
                        f"{loc['name']:18} | {horizon:3} | Real: {actual:10.2f} | "
                        f"Pred: {predicted:10.2f} | {bias_text:>7} | {'PASS' if passed else 'FAIL'}",
                        flush=True,
                    )
    return rows

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
    parser.add_argument("--target-validation", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--assert-targets", action="store_true")
    args = parser.parse_args()

    locs = build_target_locations()

    if args.rolling_validation:
        run_rolling_validation(locs)
        sys.exit(0)
    if args.target_validation:
        rows = run_target_validation(locs, verbose=not args.json)
        if args.json:
            print(json.dumps(rows, indent=2, allow_nan=True))
        if args.assert_targets and not all(row["pass"] for row in rows):
            sys.exit(1)
        sys.exit(0)

    for tf_name, (start, end, include_end) in TARGET_PERIODS.items():
        print(f"\n=== {tf_name.upper()} ===")
        for l in locs:
            run_simulation(
                l["name"], l["bbox"], l["center"], start, end,
                l.get("polygon"), l.get("radius_km", 30.0),
                include_end=include_end,
            )

if __name__ == "__main__":
    main()
