from __future__ import annotations

import json
import os
from dataclasses import dataclass

from src.config import FTP_HOST, FTP_BASE_FOLDER, DATA_RAW_DIR, FTP_FILE_FORMAT, BASE_DIR

_SETTINGS_PATH = os.path.join(BASE_DIR, "data", "server_settings.json")
_PERSIST_KEYS = ("host", "remote_dir", "local_dir", "file_format", "time_delta")

@dataclass
class ServerSettings:
    # 1. State Definition
    host: str
    remote_dir: str
    local_dir: str
    file_format: str
    time_delta: int
    username: str
    password: str

    @classmethod
    def load(cls) -> "ServerSettings":
        # 2. Disk Load Strategy
        data = {}
        if os.path.exists(_SETTINGS_PATH):
            try:
                with open(_SETTINGS_PATH, encoding="utf-8") as f: data = json.load(f)
            except Exception: pass
            
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
        # 3. Validation Logic
        try: interval = int(time_delta)
        except (TypeError, ValueError): interval = 15
        
        return cls(
            host=(host or "").strip() or FTP_HOST,
            remote_dir=(remote_dir or "").strip() or FTP_BASE_FOLDER,
            local_dir=(local_dir or "").strip() or DATA_RAW_DIR,
            file_format=(file_format or "").strip() or FTP_FILE_FORMAT,
            time_delta=interval if interval > 0 else 15,
            username=(username or "").strip(),
            password=password or "",
        )

    def save(self) -> None:
        # 4. Disk Save Strategy
        try:
            os.makedirs(self.local_dir, exist_ok=True)
            with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump({k: getattr(self, k) for k in _PERSIST_KEYS}, f, indent=2)
        except Exception as e:
            print(f"Could not save server settings: {e}")
