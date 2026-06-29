from __future__ import annotations

import os
from datetime import datetime as dt


class FrameStore:
    """Listare, filtrare pe interval si etichetare a cadrelor locale, dupa sablonul de nume.

    Sablonul (strftime, ex. 'h60_%Y%m%d_%H%M_fdk.nc.gz') e configurabil din UI. Fisierele
    locale sunt decomprimate, deci parsam dupa sablon FARA sufixul .gz.
    """

    def __init__(self, data_dir: str, file_format: str):
        self._dir = data_dir
        self.set_format(file_format)

    def set_format(self, file_format: str) -> None:
        self._fmt = file_format
        self._local_fmt = file_format[:-3] if file_format.endswith(".gz") else file_format

    def list(self) -> list[str]:
        if not os.path.isdir(self._dir):
            return []
        return sorted(f for f in os.listdir(self._dir) if self._file_datetime(f) is not None)

    def filtered(self, time_range: dict | None, run_mode: str = "historic") -> list[str]:
        """Cadrele din interval; in modul LIVE returneaza doar cadrele din ziua curenta."""
        files = self.list()
        if run_mode == "live":
            today = dt.utcnow().date()
            return [f for f in files if (d := self._file_datetime(f)) is not None and d.date() == today]

        if not time_range:
            return files

        try:
            start_dt = dt.fromisoformat(time_range["start"])
            end_dt = dt.fromisoformat(time_range["end"])
        except Exception:
            return files

        return [f for f in files if (d := self._file_datetime(f)) is not None and start_dt <= d <= end_dt]

    def path(self, filename: str) -> str:
        return os.path.join(self._dir, filename)

    def _file_datetime(self, filename: str):
        try:
            return dt.strptime(filename, self._local_fmt)
        except (ValueError, TypeError):
            return None

    def label(self, filename: str) -> str:
        """Eticheta umana 'YYYY-MM-DD HH:MM' din numele fisierului."""
        d = self._file_datetime(filename)
        return d.strftime("%Y-%m-%d %H:%M") if d is not None else filename
