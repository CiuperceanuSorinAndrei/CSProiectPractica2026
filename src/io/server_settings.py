"""Setari de server/fisiere editabile din UI, persistate pe disc (FARA credentiale).

Campurile host / director remote / director local / format fisier se salveaza in
server_settings.json. Utilizatorul si parola NU se persista niciodata - vin din .env la
pornire si pot fi suprascrise in UI doar pentru sesiunea curenta.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from config import (
    FTP_HOST, FTP_BASE_FOLDER, DATA_RAW_DIR, FTP_FILE_FORMAT,
    FTP_USER, FTP_PASS, BASE_DIR,
)

_SETTINGS_PATH = os.path.join(BASE_DIR, "server_settings.json")
# Doar aceste campuri se scriu pe disc (credentialele raman in afara fisierului).
_PERSIST_KEYS = ("host", "remote_dir", "local_dir", "file_format", "time_delta")


@dataclass
class ServerSettings:
    host: str
    remote_dir: str          # folderul de pe serverul FTP (sursa)
    local_dir: str           # folderul local unde se salveaza fisierele
    file_format: str         # sablon strftime pentru numele fisierelor
    time_delta: int
    username: str
    password: str

    @classmethod
    def load(cls) -> "ServerSettings":
        """Setarile persistate (sau implicitele din config). Credentialele vin din .env."""
        data: dict = {}
        if os.path.exists(_SETTINGS_PATH):
            try:
                with open(_SETTINGS_PATH, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        return cls(
            host=data.get("host") or FTP_HOST,
            remote_dir=data.get("remote_dir") or FTP_BASE_FOLDER,
            local_dir=data.get("local_dir") or DATA_RAW_DIR,
            file_format=data.get("file_format") or FTP_FILE_FORMAT,
            time_delta=data.get("time_delta") or 15,
            username=FTP_USER or "",
            password=FTP_PASS or "",
        )

    @classmethod
    def from_inputs(cls, host, remote_dir, local_dir, file_format, time_delta, username, password) -> "ServerSettings":
        """Construieste din campurile UI, cu fallback la implicitele din config pentru cele goale.
        Credentialele lasate goale revin la .env (nu se persista niciodata pe disc); time_delta e in minute."""
        return cls(
            host=(host or "").strip() or FTP_HOST,
            remote_dir=(remote_dir or "").strip() or FTP_BASE_FOLDER,
            local_dir=(local_dir or "").strip() or DATA_RAW_DIR,
            file_format=(file_format or "").strip() or FTP_FILE_FORMAT,
            time_delta=int(time_delta) if time_delta else 15,
            username=(username or "").strip() or (FTP_USER or ""),
            password=password or (FTP_PASS or ""),
        )

    def save(self) -> None:
        """Persista pe disc DOAR campurile non-credentiale (host/directoare/format/interval)."""
        try:
            os.makedirs(self.local_dir, exist_ok=True)
            with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump({k: getattr(self, k) for k in _PERSIST_KEYS}, f, indent=2)
        except Exception as e:
            print(f"Nu am putut salva setarile de server: {e}")
