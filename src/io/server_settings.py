"""Server settings / file settings editable from the UI, persisted to disk (WITHOUT credentials).

Fields like host / remote directory / local directory / file format are saved in
server_settings.json. The username and password are NEVER persisted - they can only be 
provided in the UI for the current session.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from src.config import (
    FTP_HOST, FTP_BASE_FOLDER, DATA_RAW_DIR, FTP_FILE_FORMAT,
    BASE_DIR,
)

_SETTINGS_PATH = os.path.join(BASE_DIR, "data", "server_settings.json")
# Only these fields are written to disk (credentials stay out of the file).
_PERSIST_KEYS = ("host", "remote_dir", "local_dir", "file_format", "time_delta")


@dataclass
class ServerSettings:
    host: str
    remote_dir: str          # folder on the FTP server (source)
    local_dir: str           # local folder where files are saved
    file_format: str         # strftime pattern for filenames
    time_delta: int
    username: str
    password: str

    @classmethod
    def load(cls) -> "ServerSettings":
        """Persisted settings (or config defaults)."""
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
            username="",
            password="",
        )

    @classmethod
    def from_inputs(cls, host, remote_dir, local_dir, file_format, time_delta, username, password) -> "ServerSettings":
        """Builds from UI fields, with fallback to config defaults for empty ones.
        time_delta is in minutes."""
        return cls(
            host=(host or "").strip() or FTP_HOST,
            remote_dir=(remote_dir or "").strip() or FTP_BASE_FOLDER,
            local_dir=(local_dir or "").strip() or DATA_RAW_DIR,
            file_format=(file_format or "").strip() or FTP_FILE_FORMAT,
            time_delta=int(time_delta) if time_delta else 15,
            username=(username or "").strip(),
            password=password or "",
        )

    def save(self) -> None:
        """Persists ONLY non-credential fields (host/directories/format/interval) to disk."""
        try:
            os.makedirs(self.local_dir, exist_ok=True)
            with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump({k: getattr(self, k) for k in _PERSIST_KEYS}, f, indent=2)
        except Exception as e:
            print(f"Could not save server settings: {e}")
