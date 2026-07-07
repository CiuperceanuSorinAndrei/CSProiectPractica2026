import argparse
import sys
import os
from datetime import datetime

# Setup paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
from src.geo.reservoir_loader import ReservoirLoader
from src.geo import sentinel2_level as s2
from src.geo.meteo_service import MeteoService

def main():
    parser = argparse.ArgumentParser(description="Verify hydrological routing model by comparing two satellite anchors.")
    parser.add_argument("--reservoir", type=str, required=True, help="Name of the reservoir (e.g., 'Vidraru')")
    parser.add_argument("--start", type=str, required=True, help="Start Date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End Date YYYY-MM-DD")
    args = parser.parse_args()

    print(f"=== Hydrological Model Verification ===")
    print(f"Reservoir: {args.reservoir}")
    print(f"Period: {args.start} to {args.end}")
    
    # 1. Load Reservoir Data
    print("\n[1] Loading reservoir data...")
    reservoirs = ReservoirLoader.get_all_reservoirs()
    
    res_data = None
    # Case-insensitive search
    for k, v in reservoirs.items():
        if k.lower() == args.reservoir.lower():
            res_data = v
            break
            
    if not res_data:
        print(f"Error: Reservoir '{args.reservoir}' not found in database.")
        print(f"Available reservoirs: {', '.join(list(reservoirs.keys())[:5])}...")
        sys.exit(1)
        
    curve = res_data.get("stage_storage")
    v_nnr = res_data.get("max_volume_m3") or 0.0
    catch_km2 = res_data.get("catchment_km2") or 0.0
    area_m2 = res_data.get("surface_area_m2") or 0.0
    
    if not curve or v_nnr <= 0:
        print("Error: Selected reservoir is missing stage-storage curve or volume data.")
        sys.exit(1)
        
    print(f"  Catchment Area: {catch_km2:.1f} km2")
    print(f"  NNR Volume: {v_nnr/1e6:.1f} mil m3")
    
    # 2. Authenticate Sentinel Hub
    if not config.SH_ID or not config.SH_SECRET:
        print("\nError: Missing SH_ID or SH_SECRET in .env file.")
        sys.exit(1)
        
    print("\n[2] Authenticating with Copernicus Sentinel Hub...")
    try:
        token = s2.get_token(config.SH_ID, config.SH_SECRET)
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    from datetime import timedelta
    
    # Helper to find valid observation
    def get_anchor(target_date_str, window_days_before=5, window_days_after=5):
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        d_from = (target_dt - timedelta(days=window_days_before)).strftime("%Y-%m-%d")
        d_to = (target_dt + timedelta(days=window_days_after)).strftime("%Y-%m-%d")
        
        actual_date = s2.best_scene_date(token, res_data["polygon"], d_from, d_to)
        if not actual_date:
            return None, None, None, None
            
        area, valid, wse = s2.measure_level(token, res_data["polygon"], curve, actual_date, actual_date, timeout=60)
        v = max(curve.volume_for_wse(wse), 0.0)
        return actual_date, wse, v, valid

    # 3. Anchor 1: Start Date
    print(f"\n[3] Measuring Start Level (Target: {args.start})...")
    actual_start, wse_1, v1, valid_1 = get_anchor(args.start, 5, 0) # Search up to 5 days before
    if not actual_start:
        print("  Error: No valid cloud-free satellite image found near start date.")
        sys.exit(1)
    print(f"  Found valid observation on {actual_start}! Level: {wse_1:.2f} m, Volume: {v1/1e6:.2f} mil m3")

    # 5. Anchor 2: End Date
    print(f"\n[4] Measuring End Level (Target: {args.end})...")
    actual_end, wse_2, v2, valid_2 = get_anchor(args.end, 0, 5) # Search up to 5 days after
    if not actual_end:
        print("  Error: No valid cloud-free satellite image found near end date.")
        sys.exit(1)
    print(f"  Found valid observation on {actual_end}! Level: {wse_2:.2f} m, Volume: {v2/1e6:.2f} mil m3")

    # 4. Routing via Open-Meteo
    print(f"\n[5] Querying Open-Meteo for Meteorological Gap ({actual_start} to {actual_end})...")
    try:
        start_dt = datetime.strptime(actual_start, "%Y-%m-%d")
        end_dt = datetime.strptime(actual_end, "%Y-%m-%d")
        days = (end_dt - start_dt).days
        
        if days <= 0:
            print("Error: End date must be strictly after start date (after adjusting for available satellite images).")
            sys.exit(1)
            
        gap_data = MeteoService.fetch_historical_gap(
            res_data["center"][0], res_data["center"][1], start_dt, end_dt)
            
        precip_mm = gap_data["precipitation_mm"]
        evap_mm = gap_data["evaporation_mm"]
        
        print(f"  Precipitation over {days} days: {precip_mm:.1f} mm")
        print(f"  Evaporation over {days} days:   {evap_mm:.1f} mm")
        
        # Calculate inflows and outflows
        if catch_km2 > 0:
            inflow_m3 = config.RUNOFF_COEFFICIENT * (precip_mm * 0.001) * (catch_km2 * 1e6)
        else:
            inflow_m3 = (precip_mm * 0.001) * area_m2
            
        base_outflow_m3s = (catch_km2 * 0.005) if catch_km2 else 10.0
        outflow_m3 = base_outflow_m3s * (days * 24 * 3600)
        evap_m3 = (evap_mm * 0.001) * area_m2
        
        print(f"  -> Inflow:  +{inflow_m3/1e6:.2f} mil m3")
        print(f"  -> Outflow: -{outflow_m3/1e6:.2f} mil m3 (Base flow assumption)")
        print(f"  -> Evap:    -{evap_m3/1e6:.2f} mil m3")
        
        v_simulated = max(v1 + inflow_m3 - outflow_m3 - evap_m3, 0.0)
        level_simulated = curve.level_for_volume(v_simulated)
        
        print(f"\n  [SIMULATED RESULT AT {actual_end}]")
        print(f"  Volume: {v_simulated/1e6:.2f} mil m3")
        print(f"  Level:  {level_simulated:.2f} m")
        
    except Exception as e:
        print(f"  Failed meteorological routing: {e}")
        sys.exit(1)
        
    # 6. Comparison
    print("\n=======================================================")
    print("VERIFICATION REPORT")
    print("=======================================================")
    print(f"  Theoretical Simulated Volume: {v_simulated/1e6:.2f} mil m3")
    print(f"  Actual Satellite Volume:      {v2/1e6:.2f} mil m3")
    print("")
    
    vol_err = v_simulated - v2
    vol_err_pct = (vol_err / v2 * 100) if v2 > 0 else 0
    level_err = level_simulated - wse_2
    
    print(f"  Volume Error: {vol_err/1e6:+.2f} mil m3 ({vol_err_pct:+.1f}%)")
    print(f"  Level Error:  {level_err:+.2f} m")
    print("=======================================================")

if __name__ == "__main__":
    main()
