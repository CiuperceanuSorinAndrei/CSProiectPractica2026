import os
import datetime
from datetime import timedelta

import xarray as xr

from config import (
    FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_FOLDER, DATA_RAW_DIR,
    FTP_TIMEOUT, FTP_MAX_RETRIES,
)
from src.io.netcdf_reader import NetCdfReader
from src.io.ftp_client import FtpClient


class CloudDataService:
    _time_frames: list[int] = None
    _ftp_client: FtpClient = FtpClient(
        FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_FOLDER,
        timeout=FTP_TIMEOUT, max_retries=FTP_MAX_RETRIES,
    )

    def __init__(self, time_frames: list[int]):
        self._time_frames = time_frames

    # Descarca toate cadrele dintr-o perioada (o singura zi) si le incarca in memorie
    def download_historical_period(self, year: int, month: int, day: int, start_hour: int, end_hour: int) -> list[xr.Dataset]:
        print(f"\n[BATCH] Pornire descarcari pentru data: {day:02d}/{month:02d}/{year}")
        target_files = self._get_target_files(year, month, day, start_hour, end_hour)
        return self.download_files(target_files)

    # Descarca o lista explicita de fisiere si le incarca in memorie
    def download_files(self, target_files: list[str]) -> list[xr.Dataset]:
        self._ftp_client.connect()
        file_paths = self._ftp_client.fetch_files(target_files)
        self._ftp_client.disconnect()

        return self._load_data(file_paths)

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

    def _get_target_files(self, year: int, month: int, day: int, start_hour: int, end_hour: int) -> list[str]:
        file_paths = []
        for hour in range(start_hour, end_hour + 1):
            for minute in self._time_frames:
                if hour == end_hour and minute > 0:
                    break
                file_paths.append(self._h60_filename_parts(year, month, day, hour, minute))

        return file_paths

    @staticmethod
    def _h60_filename(dt: datetime.datetime) -> str:
        return CloudDataService._h60_filename_parts(dt.year, dt.month, dt.day, dt.hour, dt.minute)

    @staticmethod
    def _h60_filename_parts(year: int, month: int, day: int, hour: int, minute: int) -> str:
        # Formatul standard EUMETSAT H60: h60_YYYYMMDD_HHMM_fdk.nc.gz
        return f"h60_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}_fdk.nc.gz"

    @staticmethod
    def _load_data(file_paths: list[str]) -> list[xr.Dataset]:
        datasets = []
        netcdf_reader = NetCdfReader()

        for file_path in file_paths:
            netcdf_reader.set_file_path(file_path)
            datasets.append(netcdf_reader.load_data())

        return datasets
