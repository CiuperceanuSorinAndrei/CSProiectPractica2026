import os
import datetime
from datetime import timedelta

from config import (
    FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_FOLDER, DATA_RAW_DIR,
    FTP_TIMEOUT, FTP_MAX_RETRIES,
)
from src.io.ftp_client import FtpClient


class CloudDataService:
    """Serviciu de date H-SAF: descarcare istorica (interval) si LIVE (ultimul cadru).

    Cadrele H60 sunt la fiecare 15 minute. FtpClient sare automat peste fisierele
    deja prezente local, deci re-descarcarea unui interval este ieftina.
    """

    _ftp_client: FtpClient = FtpClient(
        FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_FOLDER,
        timeout=FTP_TIMEOUT, max_retries=FTP_MAX_RETRIES,
    )

    # Descarca toate cadrele (la 15 min) din intervalul [start_dt, end_dt].
    # Returneaza numarul de fisiere noi descarcate (cele deja locale sunt sarite).
    def download_range(self, start_dt: datetime.datetime, end_dt: datetime.datetime) -> int:
        targets = self._target_filenames(start_dt, end_dt)
        missing = [f for f in targets if not os.path.exists(os.path.join(DATA_RAW_DIR, f[:-3]))]
        if missing:
            self.download_files(missing)
        return len(missing)

    # Descarca o lista explicita de fisiere; returneaza caile locale (.nc)
    def download_files(self, target_files: list[str]) -> list[str]:
        self._ftp_client.connect()
        paths = self._ftp_client.fetch_files(target_files)
        self._ftp_client.disconnect()
        return paths

    # Mod LIVE: gaseste si descarca cel mai recent cadru H-SAF disponibil pe FTP, plus
    # ultimele cateva cadre istorice (necesare pentru Optical Flow). Returneaza calea locala.
    def fetch_latest(self) -> str | None:
        self._ftp_client.connect()
        try:
            # Pornim de la ora curenta UTC, rotunjita in jos la cel mai apropiat 15 minute
            now = datetime.datetime.now(datetime.timezone.utc)
            minute_rounded = (now.minute // 15) * 15
            search_time = now.replace(minute=minute_rounded, second=0, microsecond=0)

            # Cautam inapoi pana la 12 cadre (3 ore) pentru ultima procesare disponibila
            for _ in range(12):
                filename = self._h60_filename(search_time)
                final_nc_path = os.path.join(DATA_RAW_DIR, filename[:-3])

                # Daca exista deja local, suntem la zi
                if os.path.exists(final_nc_path):
                    return final_nc_path

                # SIZE este mai rapid decat nlst()/RETR pentru a verifica existenta
                if self._ftp_client.file_size(filename):
                    path = self._ftp_client.fetch_file(filename)
                    if path:
                        self._download_flow_history(search_time)
                    return path

                search_time -= timedelta(minutes=15)

            return None
        finally:
            self._ftp_client.disconnect()

    # Descarca ultimele N cadre anterioare (istoric pentru Optical Flow in modul LIVE)
    def _download_flow_history(self, search_time: datetime.datetime, frames: int = 3) -> None:
        for prev_step in range(1, frames + 1):
            prev_time = search_time - timedelta(minutes=15 * prev_step)
            prev_filename = self._h60_filename(prev_time)
            prev_final = os.path.join(DATA_RAW_DIR, prev_filename[:-3])
            if not os.path.exists(prev_final) and self._ftp_client.file_size(prev_filename):
                self._ftp_client.fetch_file(prev_filename)

    # Numele cadrelor H60 (la 15 min) care acopera intervalul [start_dt, end_dt]
    @staticmethod
    def _target_filenames(start_dt: datetime.datetime, end_dt: datetime.datetime) -> list[str]:
        files = []
        current = start_dt
        while current <= end_dt:
            files.append(CloudDataService._h60_filename(current))
            current += timedelta(minutes=15)
        return files

    @staticmethod
    def _h60_filename(dt: datetime.datetime) -> str:
        return CloudDataService._h60_filename_parts(dt.year, dt.month, dt.day, dt.hour, dt.minute)

    @staticmethod
    def _h60_filename_parts(year: int, month: int, day: int, hour: int, minute: int) -> str:
        # Formatul standard EUMETSAT H60: h60_YYYYMMDD_HHMM_fdk.nc.gz
        return f"h60_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}_fdk.nc.gz"
