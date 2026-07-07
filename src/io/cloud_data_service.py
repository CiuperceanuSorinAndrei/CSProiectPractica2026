import os
import datetime
from datetime import timedelta

from src.config import FTP_TIMEOUT, FTP_MAX_RETRIES
from src.io.ftp_client import FtpClient
from src.io.server_settings import ServerSettings


class CloudDataService:
    """H-SAF data service: historical (interval) and LIVE (latest frame) downloads.

    All configuration (host, folders, credentials, filename format) comes from
    ServerSettings and can be changed at runtime from the UI via reconfigure().
    Frames are every 15 minutes; FtpClient automatically skips files already present.
    """

    def __init__(self, settings: ServerSettings) -> None:
        self._settings = settings
        self._build_client()

    def reconfigure(self, settings: ServerSettings) -> None:
        """Apply new settings (host/folders/credentials/format) from the UI."""
        self._settings = settings
        self._build_client()

    def _build_client(self) -> None:
        s = self._settings
        self._ftp_client = FtpClient(
            s.host, s.username, s.password, s.remote_dir, s.local_dir,
            timeout=FTP_TIMEOUT, max_retries=FTP_MAX_RETRIES,
        )

    # Remote filename of a frame, following the configured strftime pattern.
    def _filename(self, when: datetime.datetime) -> str:
        return when.strftime(self._settings.file_format)

    # Local filename (decompressed)
    @staticmethod
    def _local_name(remote_name: str) -> str:
        return remote_name[:-3] if remote_name.endswith(".gz") else remote_name

    # Download all frames (every 15 min) in the [start_dt, end_dt] interval.
    # Returns the number of newly downloaded files (those already local are skipped).
    def download_range(self, start_dt: datetime.datetime, end_dt: datetime.datetime) -> tuple[int, int]:
        targets = self._target_filenames(start_dt, end_dt)
        local_dir = self._settings.local_dir
        missing = []
        for f in targets:
            local_path = os.path.join(local_dir, self._local_name(f))
            if not os.path.exists(local_path) or os.path.getsize(local_path) < 1024:
                missing.append(f)
        if missing:
            self.download_files(missing)
            
            # Recalculate how many were actually downloaded successfully
            actually_downloaded = 0
            for f in missing:
                local_path = os.path.join(local_dir, self._local_name(f))
                if os.path.exists(local_path) and os.path.getsize(local_path) >= 1024:
                    actually_downloaded += 1
            return len(missing), actually_downloaded
            
        return 0, 0

    # Downloads an explicit list of files; returns the local paths
    def download_files(self, target_files: list[str]) -> list[str]:
        self._ftp_client.connect()
        paths = self._ftp_client.fetch_files(target_files)
        self._ftp_client.disconnect()
        return paths

    # LIVE Mode: finds and downloads the most recent available frame on the FTP, plus
    # the last few historical frames. Returns the local path.
    def fetch_latest(self) -> str | None:
        self._ftp_client.connect()
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            delta = self._settings.time_delta
            gran = delta if 0 < delta <= 60 else 60  # round to source granularity
            minute = (now.minute // gran) * gran
            search_time = now.replace(minute=minute, second=0, microsecond=0)

            latest_file = None
            for step in range(5):  # a few frames back (source sometimes has lag)
                test_time = search_time - timedelta(minutes=delta * step)
                filename = self._filename(test_time)
                if self._ftp_client.file_size(filename) is not None:
                    latest_file = filename
                    search_time = test_time
                    break

            if not latest_file:
                return None

            final_nc_path = os.path.join(self._settings.local_dir, self._local_name(latest_file))

            if os.path.exists(final_nc_path) and os.path.getsize(final_nc_path) > 1024:
                self._download_flow_history(search_time)
                return final_nc_path

            path = self._ftp_client.fetch_file(latest_file)
            if path:
                self._download_flow_history(search_time)
            return path
        except Exception as e:
            print(f"LIVE error: {e}")
            return None
        finally:
            self._ftp_client.disconnect()

    # Downloads the last N previous frames 
    def _download_flow_history(self, search_time: datetime.datetime, frames: int = 3) -> None:
        delta = self._settings.time_delta
        for prev_step in range(1, frames + 1):
            prev_time = search_time - timedelta(minutes=delta * prev_step)
            prev_filename = self._filename(prev_time)
            prev_final = os.path.join(self._settings.local_dir, self._local_name(prev_filename))
            if (not os.path.exists(prev_final) or os.path.getsize(prev_final) < 1024) and self._ftp_client.file_size(prev_filename):
                self._ftp_client.fetch_file(prev_filename)

    # Names of frames (every 15 min) covering the [start_dt, end_dt] interval
    def _target_filenames(self, start_dt: datetime.datetime, end_dt: datetime.datetime) -> list[str]:
        files = []
        current = start_dt
        step = timedelta(minutes=self._settings.time_delta)  # time_delta is in MINUTES
        while current <= end_dt:
            files.append(self._filename(current))
            current += step
        return files
