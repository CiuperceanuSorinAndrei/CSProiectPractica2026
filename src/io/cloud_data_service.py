import os
import datetime
from datetime import timedelta

from src.config import FTP_TIMEOUT, FTP_MAX_RETRIES
from src.io.ftp_client import FtpClient
from src.io.server_settings import ServerSettings

class CloudDataService:
    # 1. Initialization
    def __init__(self, settings: ServerSettings) -> None:
        self._settings = settings
        self._build_client()

    def reconfigure(self, settings: ServerSettings) -> None:
        self._settings = settings
        self._build_client()

    def _build_client(self) -> None:
        s = self._settings
        self._ftp_client = FtpClient(
            s.host, s.username, s.password, s.remote_dir, s.local_dir,
            timeout=FTP_TIMEOUT, max_retries=FTP_MAX_RETRIES,
            allow_plaintext_fallback=True
        )

    # 2. File Name Resolution
    def _filename(self, when: datetime.datetime) -> str:
        return when.strftime(self._settings.file_format)

    @staticmethod
    def _local_name(remote_name: str) -> str:
        return remote_name[:-3] if remote_name.endswith(".gz") else remote_name

    def _target_filenames(self, start_dt: datetime.datetime, end_dt: datetime.datetime) -> list[str]:
        files = []
        current = start_dt
        step = timedelta(minutes=self._settings.time_delta)
        while current <= end_dt:
            files.append(self._filename(current))
            current += step
        return files

    # 3. Explicit Download Flow
    def download_range(self, start_dt: datetime.datetime, end_dt: datetime.datetime) -> tuple[int, int]:
        targets = self._target_filenames(start_dt, end_dt)
        local_dir = self._settings.local_dir
        missing = [f for f in targets if not os.path.exists(os.path.join(local_dir, self._local_name(f))) or os.path.getsize(os.path.join(local_dir, self._local_name(f))) < 1024]
        
        if missing:
            self.download_files(missing)
            downloaded = sum(1 for f in missing if os.path.exists(os.path.join(local_dir, self._local_name(f))) and os.path.getsize(os.path.join(local_dir, self._local_name(f))) >= 1024)
            return len(missing), downloaded
            
        return 0, 0

    def download_files(self, target_files: list[str]) -> list[str]:
        self._ftp_client.connect()
        try:
            return self._ftp_client.fetch_files(target_files)
        finally:
            self._ftp_client.disconnect()

    # 4. LIVE Polling Flow
    def fetch_latest(self) -> str | None:
        self._ftp_client.connect()
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            delta = self._settings.time_delta
            minute = (now.minute // (delta if 0 < delta <= 60 else 60)) * (delta if 0 < delta <= 60 else 60)
            search_time = now.replace(minute=minute, second=0, microsecond=0)

            latest_file = next((self._filename(search_time - timedelta(minutes=delta * step)) 
                                for step in range(5) if self._ftp_client.file_size(self._filename(search_time - timedelta(minutes=delta * step))) is not None), None)

            if not latest_file: return None
            
            search_time = now.replace(minute=minute, second=0, microsecond=0) - timedelta(minutes=delta * next(step for step in range(5) if self._filename(search_time - timedelta(minutes=delta * step)) == latest_file))

            final_nc = os.path.join(self._settings.local_dir, self._local_name(latest_file))
            if os.path.exists(final_nc) and os.path.getsize(final_nc) > 1024:
                self._download_flow_history(search_time)
                return final_nc

            if self._ftp_client.fetch_file(latest_file): self._download_flow_history(search_time)
            return final_nc
        except Exception as e:
            print(f"LIVE error: {e}")
            return None
        finally:
            self._ftp_client.disconnect()

    def _download_flow_history(self, search_time: datetime.datetime, frames: int = 3) -> None:
        delta = self._settings.time_delta
        for prev_step in range(1, frames + 1):
            f_name = self._filename(search_time - timedelta(minutes=delta * prev_step))
            f_final = os.path.join(self._settings.local_dir, self._local_name(f_name))
            if (not os.path.exists(f_final) or os.path.getsize(f_final) < 1024) and self._ftp_client.file_size(f_name):
                self._ftp_client.fetch_file(f_name)
