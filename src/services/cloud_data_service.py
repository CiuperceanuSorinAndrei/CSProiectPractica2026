from config import FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_FOLDER
from src.io.netcdf_reader import NetCdfReader
from src.io.ftp_client import FtpClient
import xarray as xr

class CloudDataService:
    _time_frames: list[int] = None
    _ftp_client: FtpClient = FtpClient(FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_FOLDER)

    def __init__(self, time_frames: list[int]):
        self._time_frames = time_frames

    # Returneaza o lista cu numele fisierelor locale
    def download_historical_period(self, year: int, month: int, day: int, start_hour: int, end_hour: int) -> list[xr.Dataset]:
        print(f"\n[BATCH] Pornire descărcări pentru data: {day:02d}/{month:02d}/{year}")
        target_files = self._get_target_files(year, month, day, start_hour, end_hour)

        self._ftp_client.connect()
        file_paths = self._ftp_client.fetch_files(target_files)
        self._ftp_client.disconnect()

        datasets = self._load_data(file_paths)
        return datasets

    def _get_target_files(self, year: int, month: int, day: int, start_hour: int, end_hour: int) -> list[str]:
        file_paths = []
        for hour in range(start_hour, end_hour + 1):
            for minute in self._time_frames:
                if hour == end_hour and minute > 0:
                    break

                # Formatul standard EUMETSAT H60: h60_YYYYMMDD_HHMM_fdk.nc.gz
                filename = f"h60_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}_fdk.nc.gz"
                file_paths.append(filename)

        return file_paths

    @staticmethod
    def _load_data(file_paths: list[str]) -> list[xr.Dataset]:
        datasets = []
        netcdf_reader = NetCdfReader()

        for file_path in file_paths:
            netcdf_reader.set_file_path(file_path)
            datasets.append(netcdf_reader.load_data())

        return datasets