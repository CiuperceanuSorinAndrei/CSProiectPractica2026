from __future__ import annotations

import os
from datetime import datetime as dt


class FrameStore:
    """Listare, filtrare pe interval si etichetare a cadrelor .nc locale."""

    def __init__(self, data_dir: str):
        self._dir = data_dir

    def list(self) -> list[str]:
        return sorted(f for f in os.listdir(self._dir) if f.endswith(".nc"))

    def filtered(self, time_range: dict | None, run_mode: str = "historic") -> list[str]:
        """Cadrele din interval; in modul LIVE returneaza doar cadrele din ziua curenta."""
        files = self.list()
        if run_mode == "live":
            today = dt.utcnow().date()
            return [
                f for f in files
                if (f_dt := self._file_datetime(f)) is not None and f_dt.date() == today
            ]
        
        if not time_range:
            return files

        try:
            start_dt = dt.fromisoformat(time_range["start"])
            end_dt = dt.fromisoformat(time_range["end"])
        except Exception:
            return files

        return [
            f for f in files
            if (f_dt := self._file_datetime(f)) is not None and start_dt <= f_dt <= end_dt
        ]

    def path(self, filename: str) -> str:
        return os.path.join(self._dir, filename)

    @staticmethod
    def _file_datetime(filename: str):
        parts = filename.split("_")
        if len(parts) < 3:
            return None
        try:
            return dt.strptime(f"{parts[1]}{parts[2]}", "%Y%m%d%H%M")
        except ValueError:
            return None

    @staticmethod
    def label(filename: str) -> str:
        """Eticheta umana 'YYYY-MM-DD HH:MM' din numele fisierului H60."""
        parts = filename.split("_")
        if len(parts) < 3:
            return filename
        d, o = parts[1], parts[2]
        return f"{d[:4]}-{d[4:6]}-{d[6:]} {o[:2]}:{o[2:]}"
