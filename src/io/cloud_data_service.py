import os
import datetime
from datetime import timedelta

from config import FTP_TIMEOUT, FTP_MAX_RETRIES
from src.io.ftp_client import FtpClient
from src.io.server_settings import ServerSettings


class CloudDataService:
    """Serviciu de date H-SAF: descarcare istorica (interval) si LIVE (ultimul cadru).

    Toata configuratia (host, foldere, credentiale, format nume fisier) vine din
    ServerSettings si poate fi schimbata la runtime din UI via reconfigure().
    Cadrele sunt la fiecare 15 minute; FtpClient sare automat peste fisierele deja prezente.
    """

    def __init__(self, settings: ServerSettings) -> None:
        self._settings = settings
        self._build_client()

    def reconfigure(self, settings: ServerSettings) -> None:
        """Aplica setari noi (host/foldere/credentiale/format) venite din UI."""
        self._settings = settings
        self._build_client()

    def _build_client(self) -> None:
        s = self._settings
        self._ftp_client = FtpClient(
            s.host, s.username, s.password, s.remote_dir, s.local_dir,
            timeout=FTP_TIMEOUT, max_retries=FTP_MAX_RETRIES,
        )

    # Numele remote al unui cadru, dupa sablonul strftime configurat.
    def _filename(self, when: datetime.datetime) -> str:
        return when.strftime(self._settings.file_format)

    # Numele local (decomprimat): fara .gz daca formatul e gzipat.
    @staticmethod
    def _local_name(remote_name: str) -> str:
        return remote_name[:-3] if remote_name.endswith(".gz") else remote_name

    # Descarca toate cadrele (la 15 min) din intervalul [start_dt, end_dt].
    # Returneaza numarul de fisiere noi descarcate (cele deja locale sunt sarite).
    def download_range(self, start_dt: datetime.datetime, end_dt: datetime.datetime) -> int:
        targets = self._target_filenames(start_dt, end_dt)
        local_dir = self._settings.local_dir
        missing = [f for f in targets if not os.path.exists(os.path.join(local_dir, self._local_name(f)))]
        if missing:
            self.download_files(missing)
        return len(missing)

    # Descarca o lista explicita de fisiere; returneaza caile locale
    def download_files(self, target_files: list[str]) -> list[str]:
        self._ftp_client.connect()
        paths = self._ftp_client.fetch_files(target_files)
        self._ftp_client.disconnect()
        return paths

    # Mod LIVE: gaseste si descarca cel mai recent cadru disponibil pe FTP, plus ultimele
    # cateva cadre istorice (necesare pentru Optical Flow). Returneaza calea locala.
    def fetch_latest(self) -> str | None:
        self._ftp_client.connect()
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            delta = self._settings.time_delta
            gran = delta if 0 < delta <= 60 else 60  # rotunjire la granularitatea sursei
            minute = (now.minute // gran) * gran
            search_time = now.replace(minute=minute, second=0, microsecond=0)

            latest_file = None
            for step in range(5):  # cateva cadre in urma (sursa are uneori lag)
                test_time = search_time - timedelta(minutes=delta * step)
                filename = self._filename(test_time)
                if self._ftp_client.file_size(filename) is not None:
                    latest_file = filename
                    search_time = test_time
                    break

            if not latest_file:
                return None

            final_nc_path = os.path.join(self._settings.local_dir, self._local_name(latest_file))

            if os.path.exists(final_nc_path):
                self._download_flow_history(search_time)
                return final_nc_path

            path = self._ftp_client.fetch_file(latest_file)
            if path:
                self._download_flow_history(search_time)
            return path
        except Exception as e:
            print(f"Eroare LIVE: {e}")
            return None
        finally:
            self._ftp_client.disconnect()

    # Descarca ultimele N cadre anterioare (istoric pentru Optical Flow in modul LIVE)
    def _download_flow_history(self, search_time: datetime.datetime, frames: int = 3) -> None:
        delta = self._settings.time_delta
        for prev_step in range(1, frames + 1):
            prev_time = search_time - timedelta(minutes=delta * prev_step)
            prev_filename = self._filename(prev_time)
            prev_final = os.path.join(self._settings.local_dir, self._local_name(prev_filename))
            if not os.path.exists(prev_final) and self._ftp_client.file_size(prev_filename):
                self._ftp_client.fetch_file(prev_filename)

    # Numele cadrelor (la 15 min) care acopera intervalul [start_dt, end_dt]
    def _target_filenames(self, start_dt: datetime.datetime, end_dt: datetime.datetime) -> list[str]:
        files = []
        current = start_dt
        step = timedelta(minutes=self._settings.time_delta)  # time_delta e in MINUTE
        while current <= end_dt:
            files.append(self._filename(current))
            current += step
        return files
