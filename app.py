import streamlit as st
import os
import time
import numpy as np
import datetime
from datetime import timedelta, datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import CRS, Transformer

from src.io.netcdf_reader import NetCdfReader
from src.processing.dataset_cropper import DatasetCropper
from src.services.cloud_data_service import CloudDataService
from src.core.storm_cell_detector import StormCellDetector

from config import PREDEFINED_LOCATIONS, DEFAULT_RADIUS_KM, DEFAULT_ANIMATION_SPEED, RAIN_VMAX, RAIN_THRESHOLD_MIN

# Configurare pagina
st.set_page_config(page_title="H-SAF Nowcasting Dashboard", layout="wide", initial_sidebar_state="expanded")

# Stiluri CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Titlu
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.markdown('<div class="main-header">H-SAF Nowcasting Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analiza Precipitatii in Timp Real</div>', unsafe_allow_html=True)

# Creare director date
DATA_DIR = os.path.join("data", "raw")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("---")
    
    st.subheader("Ingestie Date Istorice")
    with st.expander("Setari Download", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Data Start", datetime.date(2026, 6, 13))
            start_hour = st.number_input("Ora Start (UTC)", min_value=0, max_value=23, value=22)
        with col2:
            end_date = st.date_input("Data Sfarsit", datetime.date(2026, 6, 14))
            end_hour = st.number_input("Ora Sfarsit (UTC)", min_value=0, max_value=23, value=23)
        
        if st.button("Descarca Perioada", use_container_width=True):
            start_dt = dt.combine(start_date, datetime.time(hour=int(start_hour)))
            end_dt = dt.combine(end_date, datetime.time(hour=int(end_hour)))
            
            if start_dt >= end_dt:
                st.error("Data start trebuie sa fie inaintea datei sfarsit")
            else:
                target_files = []
                current_dt = start_dt
                while current_dt <= end_dt:
                    filename = f"h60_{current_dt.year}{current_dt.month:02d}{current_dt.day:02d}_{current_dt.hour:02d}{current_dt.minute:02d}_fdk.nc.gz"
                    target_files.append(filename)
                    current_dt += timedelta(minutes=15)
                    
                st.info(f"Fisiere de verificat: {len(target_files)}")
                
                with st.spinner("Descarca secventa temporala..."):
                    service = CloudDataService(time_frames=[0, 15, 30, 45])
                    service.download_files(target_files)
                st.success("Descarcarea completă")
    
    st.markdown("---")
    
    st.subheader("Regiune de Interes (ROI)")
    location_choice = st.selectbox("Alege Locatie", list(PREDEFINED_LOCATIONS.keys()), label_visibility="collapsed")
    
    if location_choice == "Manual (Introducere coordonate)":
        center_lat = st.number_input("Latitudine Centru", value=44.33, min_value=40.0, max_value=52.0, step=0.1)
        center_lon = st.number_input("Longitudine Centru", value=23.79, min_value=15.0, max_value=35.0, step=0.1)
    else:
        center_lat = PREDEFINED_LOCATIONS[location_choice]["lat"]
        center_lon = PREDEFINED_LOCATIONS[location_choice]["lon"]
        st.info(f"Coordonate: {center_lat}°N, {center_lon}°E")
    
    radius_km = st.slider("Raza ROI (km)", min_value=50, max_value=500, value=DEFAULT_RADIUS_KM, step=25)
    
    st.markdown("---")
    
    st.subheader("Control Timp")
    nc_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.nc')])
    
    if len(nc_files) > 0:
        animation_speed = st.slider("Viteza cadre (sec)", 0.1, 2.0, DEFAULT_ANIMATION_SPEED, step=0.1)
        play_mode = st.toggle("Ruleaza automat", value=False)
    else:
        st.warning("Nu exista fisiere in baza de date")
        play_mode = False

# ===== CONTINUT PRINCIPAL =====

if len(nc_files) == 0:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.info("Nu exista fisiere. Configureaza perioada si descarca datele.")
else:
    # Calcul limitele ROI
    delta_lat = radius_km / 111.0
    delta_lon = radius_km / (111.0 * np.cos(np.radians(center_lat)))
    
    lon_min, lon_max = center_lon - delta_lon, center_lon + delta_lon
    lat_min, lat_max = center_lat - delta_lat, center_lat + delta_lat
    
    # Format etichete timp
    file_labels = []
    for f in nc_files:
        try:
            parts = f.split("_")
            d, o = parts[1], parts[2]
            file_labels.append(f"{d[6:]}-{d[4:6]} {o[:2]}:{o[2:]}")
        except:
            file_labels.append(f)
            
    if 'frame_idx' not in st.session_state:
        st.session_state.frame_idx = 0
    
    # Incrementare index la animatie
    if play_mode:
        if st.session_state.frame_idx < len(nc_files) - 1:
            st.session_state.frame_idx += 1
        else:
            st.session_state.frame_idx = 0
    
    # Selector cadru
    selected_idx = st.select_slider(
        "Selecteaza Cadrul",
        options=list(range(len(nc_files))),
        value=st.session_state.frame_idx,
        label_visibility="collapsed"
    )
    
    if not play_mode:
        st.session_state.frame_idx = selected_idx
    
    # Incarca date
    selected_file_name = nc_files[st.session_state.frame_idx]
    cale_completa = os.path.join(DATA_DIR, selected_file_name)
    ds = NetCdfReader(cale_completa).load_data()
    
    # Metrice principale
    st.markdown("---")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Locatie", location_choice[:20], "")
    with stat_col2:
        st.metric("Timp Cadru", file_labels[st.session_state.frame_idx], "UTC")
    with stat_col3:
        st.metric("Raza ROI", f"{radius_km} km", "")
    with stat_col4:
        st.metric("Cadre Disponibile", len(nc_files), "")
    
    st.markdown("---")
    
    if ds is None:
        st.error(f"Eroare: Fisierul {selected_file_name} nu a putut fi citit")
    else:
        ds_cropped = DatasetCropper(lon_min, lon_max, lat_min, lat_max).crop(ds)
        
        if ds_cropped is not None:
            # Extrage parametri proiectie
            proj_info = ds_cropped['geostationary_projection'].attrs
            h = proj_info['perspective_point_height']
            
            x_vals = ds_cropped['nx'].values * h if np.max(np.abs(ds_cropped['nx'].values)) < 1.0 else ds_cropped['nx'].values
            y_vals = ds_cropped['ny'].values * h if np.max(np.abs(ds_cropped['ny'].values)) < 1.0 else ds_cropped['ny'].values
            
            # Prelucrare date precipitatii
            rain_rate = ds_cropped['rr'].values
            rain_rate = np.nan_to_num(rain_rate, nan=0.0)
            rain_rate[rain_rate < 0] = 0.0
            rain_rate_masked = np.ma.masked_where(rain_rate < 0.1, rain_rate)
            
            # Transformare coordonate geostaționare la lat/lon
            x_grid, y_grid = np.meshgrid(x_vals, y_vals)
            proj4_str = (
                f"+proj=geos +h={h} +lon_0={proj_info['longitude_of_projection_origin']} "
                f"+sweep={proj_info['sweep_angle_axis']} +a={proj_info['semi_major_axis']} "
                f"+b={proj_info['semi_minor_axis']} +units=m"
            )
            transformer = Transformer.from_crs(CRS.from_proj4(proj4_str), CRS.from_epsg(4326), always_xy=True)
            lon_grid, lat_grid = transformer.transform(x_grid, y_grid)
            
            # Calcul statistici ploaie
            rain_rate_flat = rain_rate.flatten()
            rain_rate_valid = rain_rate_flat[rain_rate_flat > 0]
            max_rain = np.max(rain_rate)
            mean_rain = np.mean(rain_rate_valid) if len(rain_rate_valid) > 0 else 0
            area_ploaie = np.sum(rain_rate > 0.1) * (111.0 ** 2) / (1000 ** 2)
            
            # Detectare celule furtuna
            storm_cells = StormCellDetector(threshold=RAIN_THRESHOLD_MIN, min_size=5).extract_cells(rain_rate)

            # Filtrare centroizi valizi in ROI
            valid_cells = []
            for cell in storm_cells:
                y_idx = int(cell["centroid_y"])
                x_idx = int(cell["centroid_x"])

                if not (0 <= y_idx < lat_grid.shape[0] and 0 <= x_idx < lon_grid.shape[1]):
                    continue

                cell_lon = lon_grid[y_idx, x_idx]
                cell_lat = lat_grid[y_idx, x_idx]

                if not (np.isfinite(cell_lon) and np.isfinite(cell_lat)):
                    continue

                if not (lon_min <= cell_lon <= lon_max and lat_min <= cell_lat <= lat_max):
                    continue

                valid_cells.append((cell, cell_lon, cell_lat))
            
            # Desenare harta cu precipitatii
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            
            ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.5, edgecolor='black')
            ax.add_feature(cfeature.COASTLINE, linestyle='-', linewidth=1)
            ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3)
            
            im = ax.pcolormesh(
                lon_grid, lat_grid, rain_rate_masked,
                transform=ccrs.PlateCarree(),
                cmap='Blues',
                vmin=0.1, vmax=RAIN_VMAX,
                shading='auto',
                alpha=0.85
            )
            
            plt.colorbar(im, ax=ax, label='Intensitate ploaie (mm/h)', orientation='vertical', pad=0.02, shrink=0.8)
            ax.plot(center_lon, center_lat, 'r*', markersize=15, transform=ccrs.PlateCarree(), label='Centru ROI', zorder=5)
            
            # Marcare celule detectate
            for cell, cell_lon, cell_lat in valid_cells:
                ax.plot(cell_lon, cell_lat, 'kx', markersize=8, mew=2.5, transform=ccrs.PlateCarree(), zorder=4)
                ax.text(cell_lon + 0.03, cell_lat + 0.03, f"{cell['max_intensity']:.1f}", 
                        color='darkred', fontsize=8, fontweight='bold', transform=ccrs.PlateCarree(),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax.legend(loc='upper left', fontsize=10)
            ax.set_title(f'Harta Precipitatii - {file_labels[st.session_state.frame_idx]} UTC', fontsize=12, fontweight='bold')
            
            st.pyplot(fig, use_container_width=True)
            
            # Afisare statistici
            st.markdown("---")
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.metric("Ploaie Maxima", f"{max_rain:.2f} mm/h", "")
            with info_col2:
                st.metric("Ploaie Medie", f"{mean_rain:.2f} mm/h", "")
            with info_col3:
                st.metric("Celule Detectate", len(valid_cells), "")
            
            st.markdown("---")
            
            # Detalii analiza
            with st.expander("Detalii Analiza"):
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **Matrice Analizata:**
                    - Dimensiuni: {rain_rate.shape[0]} x {rain_rate.shape[1]} pixeli
                    - Arie cu ploaie: {area_ploaie:.1f} km²
                    - Pixeli activi: {np.sum(rain_rate > 0.1)} / {rain_rate.size}
                    """)
                with col2:
                    st.info(f"""
                    **Coordonate ROI:**
                    - Lat: [{lat_min:.2f}, {lat_max:.2f}] grade
                    - Lon: [{lon_min:.2f}, {lon_max:.2f}] grade
                    - Raza: {radius_km} km
                    """)
            
            # Tabel celule cu detalii
            if len(valid_cells) > 0:
                with st.expander("Celule Detectate - Detalii"):
                    cell_data = []
                    for i, (cell, cell_lon, cell_lat) in enumerate(valid_cells, 1):
                        cell_data.append({
                            "ID": i,
                            "Lat": f"{cell_lat:.4f}",
                            "Lon": f"{cell_lon:.4f}",
                            "Intensitate (mm/h)": f"{cell['max_intensity']:.2f}",
                            "Arie (px)": cell["area_pixels"]
                        })
                    
                    if cell_data:
                        st.dataframe(cell_data, use_container_width=True, hide_index=True)
    
    # Animatie - incrementare cadru si reincarca
    if play_mode:
        time.sleep(animation_speed)
        st.rerun()