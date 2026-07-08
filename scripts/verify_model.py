import argparse
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
from src.geo.reservoir_loader import ReservoirLoader
from src.geo import sentinel2_level as s2
from src.geo.meteo_service import MeteoService

def main():
    # 1. Initialization
    parser = argparse.ArgumentParser(description="Verify hydrological routing model.")
    parser.add_argument("--reservoir", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Verification: {args.reservoir} ({args.start} to {args.end}) ===")
    
    reservoirs = ReservoirLoader.get_all_reservoirs()
    res_data = next((v for k, v in reservoirs.items() if k.lower() == args.reservoir.lower()), None)
            
    if not res_data: sys.exit(f"Error: Reservoir '{args.reservoir}' not found.")
        
    curve = res_data.get("stage_storage")
    v_nnr = res_data.get("max_volume_m3") or 0.0
    catch_km2 = res_data.get("catchment_km2") or 0.0
    area_m2 = res_data.get("surface_area_m2") or 0.0
    
    if not curve or v_nnr <= 0: sys.exit("Error: Missing stage-storage curve or volume data.")
    if not config.SH_ID or not config.SH_SECRET: sys.exit("Error: Missing SH_ID or SH_SECRET in .env.")
        
    token = s2.get_token(config.SH_ID, config.SH_SECRET)
    
    # 2. Sentinel fetch
    def get_anchor(target_date_str, before=5, after=5):
        dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        d_from = (dt - timedelta(days=before)).strftime("%Y-%m-%d")
        d_to = (dt + timedelta(days=after)).strftime("%Y-%m-%d")
        
        act = s2.best_scene_date(token, res_data["polygon"], d_from, d_to)
        if not act: return None, None, None
            
        area, _, wse = s2.measure_level(token, res_data["polygon"], curve, act, act, timeout=60)
        return act, wse, max(curve.volume_for_wse(wse), 0.0)

    act_start, wse_1, v1 = get_anchor(args.start, 5, 0)
    if not act_start: sys.exit("Error: No start anchor.")
    print(f"Start Anchor: {act_start} | WSE: {wse_1:.2f} m | Vol: {v1/1e6:.2f} mil m3")

    act_end, wse_2, v2 = get_anchor(args.end, 0, 5)
    if not act_end: sys.exit("Error: No end anchor.")
    print(f"End Anchor: {act_end} | WSE: {wse_2:.2f} m | Vol: {v2/1e6:.2f} mil m3")

    # 3. Simulate gap
    s_dt, e_dt = datetime.strptime(act_start, "%Y-%m-%d"), datetime.strptime(act_end, "%Y-%m-%d")
    days = (e_dt - s_dt).days
    if days <= 0: sys.exit("Error: End date must be after start date.")
        
    gap = MeteoService.fetch_historical_gap(res_data["center"][0], res_data["center"][1], s_dt, e_dt)
    precip_m, evap_m = gap["precipitation_mm"] * 0.001, gap["evaporation_mm"] * 0.001
    
    inflow_m3 = (config.RUNOFF_COEFFICIENT * precip_m * catch_km2 * 1e6) if catch_km2 > 0 else (precip_m * area_m2)
    outflow_m3 = ((catch_km2 * 0.005) if catch_km2 else 10.0) * days * 24 * 3600
    evap_m3 = evap_m * area_m2
    
    v_sim = max(v1 + inflow_m3 - outflow_m3 - evap_m3, 0.0)
    l_sim = curve.level_for_volume(v_sim)
    
    # 4. Results
    print("\n--- RESULTS ---")
    print(f"Simulated Vol: {v_sim/1e6:.2f} mil m3 | Sat Vol: {v2/1e6:.2f} mil m3")
    print(f"Volume Error:  {(v_sim - v2)/1e6:+.2f} mil m3 ({((v_sim - v2) / v2 * 100) if v2 > 0 else 0:+.1f}%)")
    print(f"Level Error:   {l_sim - wse_2:+.2f} m")

if __name__ == "__main__":
    main()
