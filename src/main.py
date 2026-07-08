import sys
from pathlib import Path

# 1. Path Resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.geo.data_bootstrap import ensure_reservoir_data
from src.dashboard import NowcastingDashboard

# 2. Bootstrap Data
ensure_reservoir_data()

# 3. Application Initialization
dashboard = NowcastingDashboard()
app = dashboard.app
server = app.server

if __name__ == "__main__":
    # 4. Entry Point
    dashboard.run(debug=True, port=8050)
