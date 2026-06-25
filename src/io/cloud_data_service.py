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
            # Obtinem lista de pe server
            files = self._ftp_client._current_ftp.nlst()
            # Filtram si sortam alfabetic. Ultimul fisier este cel mai recent.
            h60_files = sorted([f for f in files if f.startswith('h60_') and f.endswith('_fdk.nc.gz')])
            
            if not h60_files:
                return None
                
            latest_file = h60_files[-1]
            final_nc_path = os.path.join(DATA_RAW_DIR, latest_file[:-3])
            
            # Verificam data / time din filename pentru flow history
            # Format: h60_YYYYMMDD_HHMM_fdk.nc.gz
            parts = latest_file.split('_')
            if len(parts) >= 3:
                date_str = parts[1]
                time_str = parts[2]
                search_time = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
            else:
                search_time = datetime.datetime.now(datetime.timezone.utc)

            if os.path.exists(final_nc_path):
                # E deja local, vedem daca e fixat corect.
                # Aici returnam calea direct, insemnand ca FTP-ul nu are ceva nou inca
                return final_nc_path

            # Altfel il descarcam
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
