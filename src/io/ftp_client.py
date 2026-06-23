import os
import ftplib
from config import FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_FOLDER, DATA_RAW_DIR
from src.processing.gz_decompressor import GZDecompressor


class FtpClient:
    _host: str = None
    _username: str = None
    _password: str = None
    _base_dir: str = None
    _current_ftp: ftplib.FTP = None

    def __init__(self, host: str, username: str, password: str, base_dir: str):
        self._host = host
        self._username = username
        self._password = password
        self._base_dir = base_dir

    def connect(self):
        print(f"Conectare la FTP {self._host}...")
        self._current_ftp = ftplib.FTP(self._host)
        self._current_ftp.login(user=self._username, passwd=self._password)
        self._current_ftp.cwd(self._base_dir)
        print(f"Conectare reusita")

    def disconnect(self):
        print(f"Deconectare de la FTP {self._host}...")
        self._current_ftp.quit()
        self._current_ftp = None
        print(f"Deconectare reusita")

    # Returneaza fisierul local unde este descarcat
    def fetch_file(self, file_name: str) -> str:
        if file_name.endswith('.gz'):
            unzipped_filename = file_name[:-3]
            remote_name = file_name
        else:
            unzipped_filename = file_name
            remote_name = file_name + '.gz'

        final_nc_path = os.path.join(DATA_RAW_DIR, unzipped_filename)
        if os.path.exists(final_nc_path):
            print(f"[SKIP] Găsit local: {unzipped_filename}")
            return final_nc_path

        gz_local_path = final_nc_path + ".gz"
        print(f"Descarcă: {remote_name} ... ", end="", flush=True)
        try:
            with open(gz_local_path, 'wb') as local_file:
                self._current_ftp.retrbinary(f"RETR {remote_name}", local_file.write)
            print("OK -> [GZIP] Extract ... ", end="", flush=True)

            GZDecompressor.decompress_file(gz_local_path, final_nc_path)
            os.remove(gz_local_path)
            print("DONE")
            return final_nc_path
        except Exception as file_err:
            print(f"EȘUAT ({file_err})")
            if os.path.exists(gz_local_path): os.remove(gz_local_path)

    # Returneaza fisierele locale unde sunt descarcate
    def fetch_files(self, file_names: list[str]) -> list[str]:
        local_paths = []
        for file_name in file_names:
            local_paths.append(self.fetch_file(file_name))

        return local_paths

# --- Testing ---
if __name__ == "__main__":
    client = FtpClient(FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_FOLDER)
    try:
        client.connect()
        client.disconnect()
    except Exception as err:
        print(err)