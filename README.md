# Satellite-based Precipitation Volume Estimation System

An advanced data pipeline and predictive nowcasting system for tracking and forecasting precipitation runoff and reservoir volumes across Romanian hydroelectric reservoirs.

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Architecture & Data Pipeline](#2-architecture--data-pipeline)
3. [Data Sources & APIs](#3-data-sources--apis)
4. [Storm Cell Detection](#4-storm-cell-detection)
5. [Storm Tracking](#5-storm-tracking)
6. [Nowcasting Algorithm (Advection + Thermodynamics)](#6-nowcasting-algorithm-advection--thermodynamics)
7. [Volume Integration & Reservoir Fill Estimation](#7-volume-integration--reservoir-fill-estimation)
8. [Running Modes: Live vs Historic (Backtest)](#8-running-modes-live-vs-historic-backtest)
9. [Constant Calibration (IS) & Validation (OOS)](#9-constant-calibration-is--validation-oos)
10. [Accuracy Results](#10-accuracy-results)
11. [Setup & Installation](#11-setup--installation)
12. [Limitations & Future Improvements](#12-limitations--future-improvements)

---

## 1. Executive Summary

**Project Name**: Satellite-based Precipitation Volume Estimation.

**Final Goal**: *Estimate precipitation accumulation in reservoir basins. These estimates can be used by HidroElectrica for better energy production management.*

**Immediate Goal**: *Identify data acquisition methods and analytical techniques to determine whether a precipitation front will intersect an area of interest, and calculate the estimated runoff volume.*

### What does this system actually do?

Imagine driving a car. Looking in the rearview mirror shows you where you've been (historical radar precipitation), while looking through the windshield shows what's ahead. This system does exactly that, but for rainfall and reservoirs:

1. **Sees** — Downloads space-borne "photographs" every 15 minutes from meteorological satellites (H60, SWOT, Sentinel-2). These contain a precipitation intensity map covering all of Romania.
2. **Detects** — Identifies "patches" of rain (storm cells) in the map, computing their rain-weighted centroid, area, volume and exact shape.
3. **Tracks** — Compares the current frame with previous ones to determine each cell's direction and speed using 4D Kalman filters (the same technology used for missile and aircraft tracking).
4. **Predicts** — Extrapolates cell movement into the future at three horizons: 15 minutes, 1 hour and 2 hours. Simulates storm intensification or dissipation using thermodynamic Reaction-Diffusion equations.
5. **Calculates** — Overlays the prediction onto the exact catchment polygon of a dam and converts millimeters of rain into cubic meters of water that will actually reach the reservoir.
6. **Learns** — Continuously compares past predictions with reality and automatically adjusts its coefficients to become increasingly accurate.

These insights are critical: if a reservoir risks sudden overflow from a precipitation front, turbines can be started preemptively to generate electricity and safely release water.

---

## 2. Architecture & Data Pipeline

The system runs a continuous 4-stage pipeline:

![Pipeline Architecture](pipeline_graph.png)

**Stage 1 — Data Acquisition**: Precipitation maps (H60) via FTP, physical lake levels (SWOT) via NASA Earthdata, water surface area (Sentinel-2) via Copernicus CDSE, Digital Elevation Models (DEM GLO-30) from S3, and evapotranspiration data (Open-Meteo API).

**Stage 2 — Nowcasting Engine**: Storm cells are detected (`StormCellDetector`), tracked frame-to-frame (`StormTracker` with 4D Kalman + Hungarian Matching), and extrapolated forward (`AdvectionEngine` with kinematic advection and Reaction-Diffusion thermodynamics).

**Stage 3 — Volume Integration**: Predicted rainfall is overlaid onto catchment polygons. Inflow is calculated, surface evaporation and base dam outflow are subtracted. The result is net estimated volume (m³).

**Stage 4 — Calibration & Output**: Predictions are corrected in real-time via In-Sample calibration (logarithmic median window). Results are displayed in the Dash Dashboard and validated Out-of-Sample via `run_simulations.py`.

---

## 3. Data Sources & APIs

The system integrates **6 external data sources**:

### 3.1. HSAF H60 (EUMETSAT / MeteoAM)
- **Protocol**: FTP/FTPS with automatic retry (3 attempts, 2s backoff).
- **Server**: `ftphsaf.meteoam.it`, folder: `h60/h60_cur_mon_data`.
- **Format**: `.nc.gz` (compressed NetCDF). Contains `rr` variable (rain rate, mm/h).
- **Temporal resolution**: One frame every 15 minutes.

### 3.2. NASA SWOT (Surface Water and Ocean Topography)
- **API**: NASA CMR (`https://cmr.earthdata.nasa.gov/search/granules.json`).
- **Auth**: NASA Earthdata Login (`EDL_USER`, `EDL_PASS`).
- **Data**: Shapefiles with Water Surface Elevation (`wse`) for Romanian lakes and rivers.

### 3.3. Copernicus Sentinel-2 API (Sentinel Hub)
- **Process URL**: `https://sh.dataspace.copernicus.eu/api/v1/process`
- **Auth**: `SH_ID`, `SH_SECRET` (OAuth2).
- **Algorithm**: NDWI = (B03 - B08) / (B03 + B08). Pixel is water if NDWI > 0, excluding cloud classes SCL ∈ {8, 9, 10}.

### 3.4. Copernicus GLO-30 DEM (S3 Public)
- **URL**: `https://copernicus-dem-30m.s3.amazonaws.com`
- **Resolution**: 30m/pixel. Used for Stage-Storage curves and catchment delineation (Priority-Flood + D8).

### 3.5. Open-Meteo Archive API (Evapotranspiration)
- **URL**: `https://archive-api.open-meteo.com/v1/archive`
- **Parameters**: `daily=precipitation_sum,et0_fao_evapotranspiration`
- **Usage**: Computes surface evaporation for gap-filling between satellite observations.

### 3.6. NASA Earthdata CMR
- **URL**: `https://cmr.earthdata.nasa.gov/search/granules.json`
- **Usage**: Searching and paginating SWOT granules for Romania.

---

## 4. Storm Cell Detection

`StormCellDetector` uses **dual-threshold connected-component labeling**:

1. **Pass 1 (Large cells)**: Binary opening of `rain_rate >= threshold` with 3×3 structuring element, then `scipy.ndimage.label()`, then `skimage.measure.regionprops()`. Filter by area.
2. **Pass 2 (Small cells)**: Same pipeline with lower threshold. Discard any overlapping with already-detected cells.

### Rain-Weighted Centroid
The centroid is **NOT geometric** — it is weighted by precipitation intensity:

$$y_{centroid} = \frac{\sum y_i \cdot rain_i}{\sum rain_i}, \quad x_{centroid} = \frac{\sum x_i \cdot rain_i}{\sum rain_i}$$

**Production constants**: `threshold = 1.0` mm/h, `min_size = 2` pixels.

---

## 5. Storm Tracking

`StormTracker` combines:

### 5.1. 4D Kalman Filter (Constant Velocity Model)
- **State**: `[x, y, vx, vy]` (4D). **Observation**: `[x, y]` (2D).
- Uses `filterpy.kalman.KalmanFilter` with process noise var=0.1, measurement noise R=diag(10,10).
- Eigenvalue clamping in [1e-8, 50.0] for stability. Joseph form covariance update.

### 5.2. KD-Tree + Hungarian Matching
1. **KD-Tree pre-filtering**: `scipy.spatial.cKDTree` within 100px radius.
2. **Hybrid cost**: `dist_norm + 0.5*area_penalty + 0.5*volume_penalty + 1.5*IoU_penalty`.
3. **Hungarian**: `scipy.optimize.linear_sum_assignment()` per connected component.

### 5.3. Cell Lifecycle
Area trend via geometric mean of last 3 area ratios. Phase determined by Reaction-Diffusion `lifecycle()` function.

---

## 6. Nowcasting Algorithm (Advection + Thermodynamics)

### 6.1. Velocity Calculation
Weighted median across tracked cells: `weight = mass × proximity × direction_weight`.

### 6.2. Kinematic Advection
`scipy.ndimage.shift(rain_rate, shift=(y, x), order=1, cval=0.0)` — sub-pixel bilinear translation.

### 6.3. Three-Component Blending
- **`shifted_raw`**: Full ROI velocity. **`mass_shifted`**: Global mass-weighted velocity. **`damped_shifted`**: Half-velocity × 0.90.
- Blended based on tracking confidence: higher confidence → more ROI weight, lower → more damped.

### 6.4. Thermodynamic Reaction-Diffusion
$$R = \begin{cases} \min(0.9 + 0.1 \frac{E}{E+1} + 0.2 \cdot \frac{dE}{E}, \ 1.05) & \text{growth} \\ \max((0.9 + 0.1 \frac{E}{E+1}) \cdot e^{-0.2 |dE/E|}, \ 0.95) & \text{decay} \end{cases}$$

Cumulative multiplier: `thermo_multiplier *= R` at each step.

### 6.5. Dry Guard
If tracking confidence < 0.35 AND recent actual < 0.03mm AND prediction < 0.20mm → multiply by 0.35.

---

## 7. Volume Integration & Reservoir Fill Estimation

### MAP Calculation
$$MAP = \frac{\sum rain_i \times pixel\_area_i \times frac_i \times 0.25}{area_{total}}$$

### Hydrological Balance
$$V_{t+1} = V_t + Q_{in} - Q_{out} - E_{evap}$$

- **Inflow**: `RUNOFF_COEFFICIENT (0.35) × depth_m × catchment_km2 × 1e6`
- **Outflow**: `catchment_km2 × 0.005 × duration_h × 3600` (5 L/s/km²)
- **Evaporation**: Open-Meteo ET0 or monthly fallback table (Jan=0.5 ... Jul=5.0 mm/day)

### Stage-Storage Curve
Built from DEM integration (flood-fill from waterline, 0-25m, 0.5m step) or prismatic model fallback. All queries via `np.interp()`.

---

## 8. Running Modes: Live vs Historic (Backtest)

### 8.1. Live Mode
- **Data**: FTP polling every 15 minutes. Downloads latest + 3 previous frames for warm-up.
- **Filtering**: Only today's files (`datetime.utcnow().date()`).
- **Latency Compensation**: Dynamic horizon shifting based on frame age:
  ```
  delay_minutes = max(0, (now - frame_time) / 60)
  step_15m = ceil((delay + 15) / 15)
  step_1h  = ceil((delay + 60) / 15)
  step_2h  = ceil((delay + 120) / 15)
  ```
- **UI**: Date picker disabled, slider auto-advances.

### 8.2. Historic (Backtest) Mode
- **Data**: Bulk download for selected date range.
- **Static horizons**: `{15m: 2, 1h: 5, 2h: 9}` steps (with H-SAF latency padding).
- **UI**: All controls enabled. Play button animates at 200ms/frame.

### 8.3. Session Manager
- Per-user `Orchestrator` + `FrameHistory`. Dataset change → full reset + replay.
- Session expiry: 3600 seconds of inactivity.

---

## 9. Constant Calibration (IS) & Validation (OOS)

All hardcoded constants were calibrated **In-Sample** on historical periods and validated **Out-of-Sample** on unseen data:

| Constant | Value | Meaning |
|----------|-------|---------|
| `_BIAS_MIN` | 0.45 | Lower bias correction bound |
| `_BIAS_MAX` | 1.60 | Upper bias correction bound |
| `_BIAS_ALPHA_UP` | 0.35 | Upward EMA learning rate |
| `_BIAS_ALPHA_DOWN` | 0.55 | Downward EMA learning rate |
| `_STATIC_HORIZON_CALIBRATION` | {15m: 1.0, 1h: 1.0, 2h: 1.065} | Static per-horizon correction |
| `RUNOFF_COEFFICIENT` | 0.35 | Hydrological runoff coefficient |
| `HORIZON_STEPS` | {15m: 2, 1h: 5, 2h: 9} | Steps with H-SAF latency padding |
| Kalman Q variance | 0.1 | Process noise |
| Kalman R | diag(10, 10) | Measurement noise |

### Dynamic Bias Correction
Asymmetric EMA with cumulative actual/predicted ratio:
$$bias_{new} = (1-\alpha) \times bias_{current} + \alpha \times \frac{\sum actual}{\sum predicted}$$

α = 0.55 (over-prediction correction), α = 0.35 (under-prediction), α = 0.80 (2h horizon).

---

## 10. Accuracy Results

### OOS Validation — Period 1 (2026-06-24 → 2026-07-06)

| Location | 15m | 1h | 2h |
|----------|-----|----|----|
| Craiova | -6.2% | -14.6% | -21.6% |
| Vidraru | +2.4% | +3.3% | +7.5% |
| Portile De Fier I | -0.5% | -1.0% | -2.7% |
| Izvorul Muntelui | -5.5% | -10.5% | -3.6% |
| Gura Apelor | +11.9% | +21.2% | +35.1% |
| Tarnita | +3.5% | +10.1% | +19.2% |
| Somesu Cald | +2.5% | +13.4% | +20.9% |

### OOS Validation — Full Period (2026-06-01 → 2026-07-06)

| Location | 15m | 1h | 2h |
|----------|-----|----|----|
| Craiova | -3.5% | -7.7% | -8.1% |
| Vidraru | -0.8% | -2.1% | -0.3% |
| Portile De Fier I | -0.7% | -2.2% | -3.7% |
| Izvorul Muntelui | -6.1% | -8.4% | -2.7% |
| Gura Apelor | -0.3% | +1.1% | +10.4% |
| Tarnita | -3.7% | -4.2% | 0% |
| Somesu Cald | -4.3% | -2.7% | +1.7% |

### Full Period Aggregate

| Horizon | Mean Absolute Bias |
|---------|-------------------|
| **15m** | **2.8%** |
| **1h** | **4.1%** |
| **2h** | **3.8%** |

---

## 11. Setup & Installation

### Prerequisites
- Python 3.10+
- Valid credentials for Sentinel Hub and NASA Earthdata Login.

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration (`.env`)
```env
EDL_USER=your_earthdata_username
EDL_PASS=your_earthdata_password
SH_ID=your_sentinel_hub_client_id
SH_SECRET=your_sentinel_hub_client_secret
```

### Running
```bash
# Dashboard
python src/main.py

# Target validation
python scripts/run_simulations.py --target-validation

# Rolling validation
python scripts/run_simulations.py --rolling-validation
```

---

## 12. Limitations & Future Improvements

### Current Limitations
1. **Linear advection**: `KinematicAdvector` only applies uniform translations. Cannot model rotations, shear, or local divergence.
2. **Fixed runoff coefficient**: `RUNOFF_COEFFICIENT = 0.35` is constant regardless of soil type, antecedent moisture, or slope.
3. **Temporal resolution**: H60 provides one frame per 15 minutes. Fast convective storms can evolve significantly between frames.
4. **SWOT/S2 data quality dependence**: Initial lake level accuracy depends on cloud cover and satellite orbit frequency.

### Future Improvements
1. **Optical Flow / Deep Learning**: Replace kinematic advection with ConvLSTM, TrajGRU, or MetNet architectures.
2. **DEM-based Hydrological Routing**: Use the already-computed D8 drainage network to route water through actual riverbeds.
3. **Soil Moisture Integration (SMAP)**: Pre-calibrate runoff coefficient based on soil saturation.
4. **GPM Data Assimilation**: Integrate Global Precipitation Measurement as a redundant source.
5. **Ensemble Prediction**: Run multiple scenarios with perturbed velocities and thermodynamics for confidence intervals.

---

## Dependencies

| Package | Min Version | Purpose |
|---------|-------------|---------|
| `netCDF4` | ≥1.6.0 | Reading H60 .nc files |
| `numpy` | ≥1.22.0 | Matrix operations |
| `scipy` | ≥1.8.0 | `ndimage.shift`, `ndimage.label`, `cKDTree`, `linear_sum_assignment` |
| `shapely` | ≥2.0.0 | Geometry operations, `STRtree` |
| `filterpy` | ≥1.4.5 | 4D Kalman filters |
| `scikit-image` | ≥0.19.0 | `regionprops` for morphological properties |
| `dash` | ≥2.14.0 | Web Dashboard framework |
| `pyproj` | ≥3.4.0 | Coordinate transformations |
| `pyshp` | ≥3.0.0 | SWOT shapefile reading |
| `requests` | ≥2.28.0 | HTTP calls (Sentinel Hub, Open-Meteo) |
