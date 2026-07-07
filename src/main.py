"""Entry point for the Nowcasting Dashboard."""
import sys
import os
from pathlib import Path

# Adaugă folderul părinte în sys.path pentru ca `src.` să fie recunoscut
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.geo.data_bootstrap import ensure_reservoir_data
from src.dashboard import NowcastingDashboard

# Verifica datele volumetrice (curbe DEM, nivele SWOT/Sentinel-2) si descarca ce lipseste.
ensure_reservoir_data()

dashboard = NowcastingDashboard()
app = dashboard.app
server = app.server

if __name__ == "__main__":
    dashboard.run(debug=True, port=8050)
