import argparse
import os
import sys

# Asiguram ca directorul src poate fi gasit
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.constants import HORIZON_NAMES
from src.dashboard.constants import DATA_DIR
from src.dashboard.frame_store import FrameStore
from src.dashboard.session_manager import SessionManager
from src.geo.reservoir_loader import ReservoirLoader


TARGET_RESERVOIRS = (
    "Vidraru",
    "Portile De Fier I",
    "Izvorul Muntelui",
    "Gura Apelor",
    "Tarnita",
    "Somesu Cald",
)


def build_target_locations() -> list[dict]:
    locations = [
        {
            "name": "Craiova",
            "bbox": (23.750, 23.850, 44.250, 44.350),
            "center": (44.310, 23.800),
            "radius_km": 30.0,
            "polygon": None,
        }
    ]
    reservoirs = ReservoirLoader.get_all_reservoirs()
    missing = [name for name in TARGET_RESERVOIRS if name not in reservoirs]
    if missing:
        raise RuntimeError(f"Missing local reservoir polygons: {', '.join(missing)}")
    for name in TARGET_RESERVOIRS:
        res = reservoirs[name]
        min_lon, min_lat, max_lon, max_lat = res["bounds"]
        locations.append({
            "name": name,
            "bbox": (min_lon, max_lon, min_lat, max_lat),
            "center": res["center"],
            "radius_km": res["radius_km"],
            "polygon": res["polygon"],
        })
    return locations


def run_simulation(
    location_name: str,
    bbox: tuple[float, float, float, float],
    center: tuple[float, float],
    start_str: str,
    end_str: str,
    polygon=None,
    radius_km: float = 30.0,
    quiet: bool = False,
):
    store = FrameStore(DATA_DIR, "h60_%Y%m%d_%H%M_fdk.nc.gz")

    time_range = {"start": start_str, "end": end_str}
    files = store.filtered(time_range, run_mode="historic")

    if not quiet:
        print(f"Running simulation for {location_name} from {start_str} to {end_str} with {len(files)} files.")

    if not files:
        if not quiet:
            print("No files found!")
        return None

    session_manager = SessionManager()
    session_id = f"test_{location_name}_{start_str}_{end_str}"

    # Rulam primul cadru pentru initializare
    session_manager.process_to_frame(
        session_id=session_id,
        frame_idx=0,
        nc_files=files,
        bbox=bbox,
        center=center,
        radius_km=radius_km,
        polygon=polygon,
        run_mode="historic",
        time_range=time_range,
        store=store,
    )

    # Rulam toate cadrele folosind _accumulate_range (acelasi flux ca la salt)
    orch, hist = session_manager.get_state(session_id)

    for i in range(1, len(files)):
        if not quiet and i % 100 == 0:
            print(f"Processed {i}/{len(files)} frames...")
        session_manager.process_to_frame(
            session_id=session_id,
            frame_idx=i,
            nc_files=files,
            bbox=bbox,
            center=center,
            radius_km=radius_km,
            polygon=polygon,
            run_mode="historic",
            time_range=time_range,
            store=store,
        )

    if not quiet:
        print(f"--- Simulation Complete for {location_name} ---")

    actual_by_horizon = {}
    predicted_by_horizon = {}
    if not quiet:
        print("\nAcumulare Precipitatii Bazin")
        print("Orizont | Realizat (L/m2) | Prezis (L/m2) | Bias (%)")
    for horizon in HORIZON_NAMES:
        real_vol, pred_vol = hist.volume_sums(horizon)
        actual_by_horizon[horizon] = real_vol
        predicted_by_horizon[horizon] = pred_vol
        bias = ((pred_vol - real_vol) / real_vol * 100.0) if real_vol > 0 else 0.0
        if not quiet:
            print(f"{horizon:7} | {real_vol:15.2f} | {pred_vol:13.2f} | {bias:+.1f}%")

    metrics = hist.get_reliability_metrics()
    if not quiet:
        print("\nIncredere Avertizari Bazin (Acuratete Volum Cumulat)")
        for t in [1.0, 5.0]:
            print(f"Acumulare > {t} L/m2:")
            print("Orizont | POD | FAR | CMAPE")
            for horizon in HORIZON_NAMES:
                m = metrics[t][horizon]
                print(f"{horizon:7} | {m['pod']:3.0f}% | {m['far']:3.0f}% | +/-{m['cmae']:.1f}%")
        print("\n")
    orch.stop_warmup()
    return {
        "actual": actual_by_horizon,
        "predicted": predicted_by_horizon,
    }


def run_rolling_validation(locations: list[dict], start_day: int = 1, end_day: int = 26) -> None:
    by_location = {
        loc["name"]: {horizon: {"actual": 0.0, "predicted": 0.0} for horizon in HORIZON_NAMES}
        for loc in locations
    }
    totals = {
        horizon: {"actual": 0.0, "predicted": 0.0}
        for horizon in HORIZON_NAMES
    }
    cases = 0

    for day in range(start_day, end_day + 1):
        start = f"2026-06-{day:02d}T00:00:00"
        end = f"2026-06-{day:02d}T23:59:59"
        print(f"Rolling window {start} -> {end}")
        for loc in locations:
            result = run_simulation(
                loc["name"], loc["bbox"], loc["center"], start, end,
                polygon=loc.get("polygon"), radius_km=loc.get("radius_km", 30.0), quiet=True
            )
            if result is None:
                continue
            cases += 1
            for horizon in HORIZON_NAMES:
                totals[horizon]["actual"] += result["actual"][horizon]
                totals[horizon]["predicted"] += result["predicted"][horizon]
                by_location[loc["name"]][horizon]["actual"] += result["actual"][horizon]
                by_location[loc["name"]][horizon]["predicted"] += result["predicted"][horizon]

    print("\nLocation rolling validation, June 1-26")
    print("Location           | Horizon | Realized MAP | Predicted MAP | Bias (%) | Gate")
    for loc in locations:
        for horizon in HORIZON_NAMES:
            actual = by_location[loc["name"]][horizon]["actual"]
            predicted = by_location[loc["name"]][horizon]["predicted"]
            bias = ((predicted - actual) / actual * 100.0) if actual > 0 else 0.0
            gate = "PASS" if -15.0 < bias < 15.0 else "FAIL"
            print(f"{loc['name'][:18]:18} | {horizon:7} | {actual:12.2f} | {predicted:13.2f} | {bias:+7.1f}% | {gate}")

    print("\nAggregate rolling validation, June 1-26")
    print("Cases   | Horizon | Realized MAP | Predicted MAP | Bias (%) | Gate")
    for horizon in HORIZON_NAMES:
        actual = totals[horizon]["actual"]
        predicted = totals[horizon]["predicted"]
        bias = ((predicted - actual) / actual * 100.0) if actual > 0 else 0.0
        gate = "PASS" if -15.0 < bias < 15.0 else "FAIL"
        print(f"{cases:7d} | {horizon:7} | {actual:12.2f} | {predicted:13.2f} | {bias:+7.1f}% | {gate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling-validation", action="store_true")
    args = parser.parse_args()

    locations = build_target_locations()

    if args.rolling_validation:
        run_rolling_validation(locations)
        raise SystemExit(0)

    # Define timeframes
    timeframes = {
        "Short": ("2026-06-13T22:00:00", "2026-06-14T23:00:00"),
        "Medium": ("2026-06-10T01:00:00", "2026-06-14T23:00:00"),
    }

    for tf_name, (start_time, end_time) in timeframes.items():
        print("\n========================================")
        print(f"RUNNING TIMEFRAME: {tf_name}")
        print("========================================\n")
        for loc in locations:
            run_simulation(
                loc["name"], loc["bbox"], loc["center"], start_time, end_time,
                polygon=loc.get("polygon"), radius_km=loc.get("radius_km", 30.0)
            )
