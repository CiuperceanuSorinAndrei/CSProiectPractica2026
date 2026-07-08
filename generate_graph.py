import base64
import urllib.request
import json

# Expanded pipeline with Open-Meteo, DEM, and all actual components
mermaid_code = """
graph TD
    subgraph "1. Achizitia Datelor"
        H["HSAF H60 FTP<br/>(ftphsaf.meteoam.it)<br/>Precipitatii 15 min"] --> |".nc.gz files"| FTP["FTP Client<br/>(FTPS + Retry)"]
        FTP --> DEC["GZ Decompressor"]
        DEC --> PP["Frame Preprocessor<br/>(NetCDF -> Matrice)"]
        
        S1["NASA SWOT CMR<br/>(Earthdata Login)<br/>WSE Lacuri/Rauri"] --> |Shapefiles| RL["Reservoir Loader<br/>(Stereo 70 -> WGS84)"]
        
        S2["Sentinel-2 API<br/>(Copernicus CDSE)<br/>NDWI Water Masks"] --> |"Suprafata Apa"| RL
        
        DEM["Copernicus GLO-30<br/>(S3 Public)<br/>DEM 30m"] --> |"Elevatie"| SS["Stage-Storage<br/>Curve Builder"]
        SS --> RL
        
        OM["Open-Meteo API<br/>(archive-api)<br/>ET0 FAO Evaporatie"] --> |"Evapotranspiratie"| RLS["Reservoir Level<br/>Service"]
        RL --> RLS
    end
    
    subgraph "2. Motorul de Nowcasting"
        PP --> |"Matrice Ploaie"| DET["Storm Cell Detector<br/>(Connected Components<br/>+ Regionprops)"]
        DET --> |"Celule Detectate"| ST["Storm Tracker<br/>(Kalman 4D + Hungarian<br/>+ KD-Tree Matching)"]
        ST --> |"Centroizi, Viteze,<br/>Volum, Trend"| AE["Advection Engine<br/>(Lagrangian)"]
        AE --> |"Shift sub-pixel"| KA["Kinematic Advector<br/>(scipy.ndimage.shift)"]
        AE <--> |"Multiplicator<br/>Termodinamic"| RD["Reaction-Diffusion<br/>(Crestere / Disipare)"]
        AE --> |"Harti Prezise<br/>(15m, 1h, 2h)"| EV["Evaluator MAP<br/>(Mean Areal Precip)"]
    end
    
    subgraph "3. Integrare Volum"
        EV --> |"mm/h pe ROI"| SM["Session Manager"]
        RLS --> |"Poligoane Bazin,<br/>Nivel Curent"| SM
        SM --> |"Precipitatie * Coef Scurgere<br/>(C = 0.35)"| RF["Reservoir Fill<br/>Estimator"]
        RF --> |"Debit Intrare - Debit Iesire<br/>- Evaporare"| Vol["Volume Prezise<br/>(m3 si %)"]
    end
    
    subgraph "4. Calibrare si Validare"
        Vol --> IS["Calibrare In-Sample<br/>(Fereastra Mediana Log<br/>+ EMA Asimetric)"]
        IS --> |"Bias Correction<br/>per Orizont"| AE
        Vol --> DB["Dashboard Dash<br/>(Live + Historic)"]
        Vol --> Val["Validare OOS<br/>(run_simulations.py)"]
    end
"""

state = {"code": mermaid_code, "mermaid": {"theme": "default"}}
b64 = base64.urlsafe_b64encode(json.dumps(state).encode('utf-8')).decode('utf-8')
url = f"https://mermaid.ink/img/{b64}?width=1400&height=1000"

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=30) as response, open("pipeline_graph.png", 'wb') as out_file:
        out_file.write(response.read())
    print("Graph downloaded successfully!")
except Exception as e:
    print(f"Error downloading graph: {e}")
