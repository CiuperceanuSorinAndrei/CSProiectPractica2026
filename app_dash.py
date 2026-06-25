"""H-SAF Nowcasting Dashboard — punct de intrare.

Structura UI/aplicatie traieste in pachetul `src.dashboard`. Aici doar instantiem
si pornim aplicatia (si expunem `app`/`server` pentru WSGI / harness-ul de debug).
"""
from src.dashboard import NowcastingDashboard

dashboard = NowcastingDashboard()
app = dashboard.app
server = app.server

if __name__ == "__main__":
    dashboard.run(debug=True, port=8050)
