
  # Advection-Led Bias Improvement Plan

  ## Summary

  Keep the forecast as a real nowcast: move the rain field forward using tracked storm motion, then calibrate only the systematic volume error. Do not
  replace 15m/1h with “same as now” persistence.

  ## Key Changes

  - Keep Evaluator.calculate_volumes() and FrameHistory.volume_sums() alignment unchanged.
  - Improve AdvectionEngine.extrapolate() with a cheap advection ensemble:
      - Current ROI-weighted velocity forecast.
      - Mass-weighted storm velocity forecast.
      - Conservative damped forecast used only when tracking confidence is poor.

  - Blend those forecasts by existing confidence signals: centroid error, size error, ROI proximity, and number of tracked cells.
  - Keep online per-horizon bias correction, but make it more robust:
      - Store recent matured actual / predicted ratios per horizon.
      - Use median log-ratio, already cheap and resistant to outliers.
      - Add asymmetric protection for dry false positives: allow faster downward correction than upward correction.
      - Do not widen upward correction aggressively.

  - Add a very small dry guard only when recent actual rain is near zero and predicted volume is also low-confidence; this suppresses drizzle tails, not real
    incoming storms.

  ## Validation

  - Treat full IS as the tuning set:
      - Run all 7 locations from 2026-06-01T00:00:00 through the agreed IS end date, likely 2026-06-20 or 2026-06-23.
      - Report bias for 15m, 1h, 2h per location.

  - Freeze parameters after IS passes or reaches best defensible result.
  - Then run OOS separately:
  - Acceptance target:
      - Primary: -15% < Volumetric bias < +15% for all 7 locations and 3 horizons on full IS.
      - Secondary: no major OOS regression versus current baseline.

  ## Tests

  - Existing unit tests must stay green.
  - Add tests for:
      - Advection ensemble still returns non-NaN maps with correct shape.
      - Low tracking confidence increases damping but does not replace forecast with raw persistence.
      - Dry guard only triggers for low-recent-rain and low-confidence forecasts.
      - Per-horizon bias correction compares matured forecasts to matching cumulative windows.

  - Run:
      - .\.venv\Scripts\python.exe -m pytest -q
      - Full IS validation command.
      - OOS validation command.

  ## Assumptions

  - No external data.
  - No heavy methods like neural nets, optical-flow grids, or parameter searches over large combinations.
  - The forecast must remain advection-led; persistence is allowed only as a confidence fallback, not as the main 15m/1h prediction.