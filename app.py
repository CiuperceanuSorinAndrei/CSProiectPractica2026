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
from src.io.reader import load_hsaf_data
from src.processing.cropper import crop_dataset_to_bbox
from src.io.ftp_client import fetch_hsaf_files

# Configurare pagina Streamlit
st.set_page_config(page_title="H-SAF Nowcasting Dashboard", layout="wide")
st.title("Sistem Nowcasting H-SAF - Analiză Secvențială")

# Asigurare existenta folder date
DATA_DIR = os.path.join("data", "raw")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ==========================================
# 1. SIDEBAR: INGESTIE DATE ISTORICE
# ==========================================
st.sidebar.header("1. Ingestie Date Istorice")

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Data Start", datetime.date(2026, 6, 13))
    start_hour = st.number_input("Ora Start (UTC)", min_value=0, max_value=23, value=22)
with col2:
    end_date = st.date_input("Data Sfarsit", datetime.date(2026, 6, 14))
    end_hour = st.number_input("Ora Sfarsit (UTC)", min_value=0, max_value=23, value=23)

if st.sidebar.button("Descarcă Perioada Continuă"):
    start_dt = dt.combine(start_date, datetime.time(hour=int(start_hour)))
    end_dt = dt.combine(end_date, datetime.time(hour=int(end_hour)))
    
    if start_dt >= end_dt:
        st.sidebar.error("Momentul de start trebuie sa fie inaintea celui de sfarsit.")
    else:
        target_files = []
        current_dt = start_dt
        while current_dt <= end_dt:
            filename = f"h60_{current_dt.year}{current_dt.month:02d}{current_dt.day:02d}_{current_dt.hour:02d}{current_dt.minute:02d}_fdk.nc.gz"
            target_files.append(filename)
            current_dt += timedelta(minutes=15)
            
        st.sidebar.write(f"Fisiere de verificat: {len(target_files)}")
        
        with st.spinner("Se descarca secventa temporala..."):
            fetch_hsaf_files(target_files)
        st.sidebar.success("Secventa descarcata cu succes!")

# Scanare folder local pentru a gasi fisierele .nc dezarhivate
nc_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.nc')])

# ==========================================
# 2. SIDEBAR: CONFIGURARE REGIUNE DE INTERES (ROI)
# ==========================================
st.sidebar.header("2. Setări Regiune (ROI)")
center_lat = st.sidebar.number_input("Latitudine Centru", value=44.33, min_value=40.0, max_value=52.0, step=0.1)
center_lon = st.sidebar.number_input("Longitudine Centru", value=23.79, min_value=15.0, max_value=35.0, step=0.1)
radius_km = st.sidebar.slider("Raza ariei (km)", min_value=50, max_value=500, value=200, step=25)

# Conversie aproximativa din Kilometri in Grade Geografice pentru latitudinea Romaniei
delta_lat = radius_km / 111.0
delta_lon = radius_km / (111.0 * np.cos(np.radians(center_lat)))

lon_min, lon_max = center_lon - delta_lon, center_lon + delta_lon
lat_min, lat_max = center_lat - delta_lat, center_lat + delta_lat

# ==========================================
# 3. CONTROL TIMP SI AFISARE GRAFIC
# ==========================================
if len(nc_files) == 0:
    st.info("Nu exista fisiere in data/raw/. Configureaza perioada in stanga si porneste descarcarea.")
else:
    st.sidebar.header("3. Control Timp")
    
    # Construim etichete lizibile pentru slider (ex: "13-06 22:15")
    file_labels = []
    for f in nc_files:
        try:
            parts = f.split("_")
            d, o = parts[1], parts[2]
            file_labels.append(f"{d[6:]}-{d[4:6]} {o[:2]}:{o[2:]}")
        except:
            file_labels.append(f)
            
    # Initializam indexul cadrului curent in session_state daca nu exista
    if 'frame_idx' not in st.session_state:
        st.session_state.frame_idx = 0

    # Comutator pentru animatie automata si viteza
    play_mode = st.sidebar.toggle("Rulează automat ", value=False)
    animation_speed = st.sidebar.slider("Viteză cadre ", 0.1, 2.0, 0.4, step=0.1)

    if play_mode:
        if st.session_state.frame_idx < len(nc_files) - 1:
            st.session_state.frame_idx += 1
        else:
            st.session_state.frame_idx = 0  # Loop: o ia de la inceput

    selected_idx = st.sidebar.slider(
        "Selectează Cadrul", 
        min_value=0, 
        max_value=len(nc_files) - 1, 
        value=st.session_state.frame_idx,
        key="time_slider_control"
    )
    
    if not play_mode:
        st.session_state.frame_idx = selected_idx

    selected_file_name = nc_files[st.session_state.frame_idx]
    st.subheader(f"Timp Cadru Curent: {file_labels[st.session_state.frame_idx]} UTC")

    # Incarcare si procesare date fisier curent
    cale_completa = os.path.join(DATA_DIR, selected_file_name)
    ds = load_hsaf_data(cale_completa)
    
    if ds is not None:
        # Decupare dinamica pe baza ROI calculat in km
        ds_cropped = crop_dataset_to_bbox(ds, lon_min, lon_max, lat_min, lat_max)
        
        if ds_cropped is not None:
            proj_info = ds_cropped['geostationary_projection'].attrs
            h = proj_info['perspective_point_height']
            
            x_vals = ds_cropped['nx'].values * h if np.max(np.abs(ds_cropped['nx'].values)) < 1.0 else ds_cropped['nx'].values
            y_vals = ds_cropped['ny'].values * h if np.max(np.abs(ds_cropped['ny'].values)) < 1.0 else ds_cropped['ny'].values
            
            rain_rate = ds_cropped['rr'].values
            rain_rate = np.nan_to_num(rain_rate, nan=0.0)
            rain_rate[rain_rate < 0] = 0.0
            
            # Reproiectare grila pixeli in Lat/Lon pentru harta
            x_grid, y_grid = np.meshgrid(x_vals, y_vals)
            proj4_str = (
                f"+proj=geos +h={h} +lon_0={proj_info['longitude_of_projection_origin']} "
                f"+sweep={proj_info['sweep_angle_axis']} +a={proj_info['semi_major_axis']} "
                f"+b={proj_info['semi_minor_axis']} +units=m"
            )
            transformer = Transformer.from_crs(CRS.from_proj4(proj4_str), CRS.from_epsg(4326), always_xy=True)
            lon_grid, lat_grid = transformer.transform(x_grid, y_grid)
            
            # Generare figura Matplotlib cu proiectie Cartopy
            fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            
            # Adaugare elemente de harta (Granite state, Tarmuri)
            ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.5, edgecolor='black')
            ax.add_feature(cfeature.COASTLINE, linestyle='-', linewidth=1)
            
            # Randare date precipitatii (pcolormesh)
            im = ax.pcolormesh(
                lon_grid, lat_grid, rain_rate,
                transform=ccrs.PlateCarree(),
                cmap='Blues',
                vmin=0.1, vmax=12.0,
                shading='auto',
                alpha=0.7
            )
            
            plt.colorbar(im, ax=ax, label='Intensitate ploaie (mm/h)', orientation='horizontal', pad=0.05, shrink=0.7)
            
            # Desenare punct pentru coordonatele centrale selectate
            ax.plot(center_lon, center_lat, 'ro', markersize=8, transform=ccrs.PlateCarree(), label='Centru ROI')
            ax.legend()
            
            # Trimitere plot catre aplicatia Streamlit
            st.pyplot(fig)
            st.write(f"Rezoluție sub-matrice curentă: {rain_rate.shape} pixeli (Linii x Coloane)")

    # Daca Timelapse-ul este pornit, asteptam intervalul setat si fortam re-randarea aplicatiei
    if play_mode:
        time.sleep(animation_speed)
        st.rerun()