import os
import time
import ftplib
from src.config import FTP_HOST, FTP_BASE_FOLDER, DATA_RAW_DIR
from src.io.gz_decompressor import decompress_file


class FtpClient:
    _host: str = None
    _username: str = None
    _password: str = None
    _base_dir: str = None
    _local_dir: str = None
    _timeout: int = None
    _max_retries: int = None
    _retry_backoff: float = None
    _current_ftp: ftplib.FTP = None

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        base_dir: str,
        local_dir: str = DATA_RAW_DIR,
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        self._host = host
        self._username = username
        self._password = password
        self._base_dir = base_dir
        self._local_dir = local_dir
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    def connect(self):
        print(f"Connecting to FTP {self._host}...")
        try:
            self._current_ftp = ftplib.FTP_TLS(self._host, timeout=self._timeout)
            self._current_ftp.login(user=self._username, passwd=self._password)
            self._current_ftp.prot_p()  # Secures the data connection
            print("Secure connection (FTPS) successful")
        except ftplib.error_perm as e:
            if "Login incorrect" in str(e) or "Authentication failed" in str(e):
                print(f"Authentication failed: {e}")
                raise  # Let the caller know authentication failed directly!
            # Otherwise maybe it's some other permission issue, fallback
            print(f"FTPS connection failed ({e}). Attempting fallback to plaintext FTP...")
            self._current_ftp = ftplib.FTP(self._host, timeout=self._timeout)
            self._current_ftp.login(user=self._username, passwd=self._password)
            print("Plaintext connection (FTP) successful")
        except (ftplib.error_temp, EOFError, OSError) as e:
            print(f"FTPS connection failed ({e}). Attempting fallback to plaintext FTP...")
            self._current_ftp = ftplib.FTP(self._host, timeout=self._timeout)
            self._current_ftp.login(user=self._username, passwd=self._password)
            print("Plaintext connection (FTP) successful")
        self._current_ftp.cwd(self._base_dir)

    def disconnect(self):
        print(f"Disconnecting from FTP {self._host}...")
        self._current_ftp.quit()
        self._current_ftp = None
        print("Disconnection successful")

    # Size of a remote file in bytes (None if it doesn't exist). Used in LIVE mode.
    def file_size(self, file_name: str) -> int | None:
        try:
            return self._current_ftp.size(file_name)
        except Exception:
            return None

    # Returns the local path where the file is downloaded (or None on failure)
    def fetch_file(self, file_name: str) -> str | None:
        if file_name.endswith('.gz'):
            unzipped_filename = file_name[:-3]
            remote_name = file_name
        else:
            unzipped_filename = file_name
            remote_name = file_name + '.gz'

        os.makedirs(self._local_dir, exist_ok=True)
        final_nc_path = os.path.join(self._local_dir, unzipped_filename)
        if os.path.exists(final_nc_path) and os.path.getsize(final_nc_path) > 1024:
            print(f"[SKIP] Found locally: {unzipped_filename}")
            return final_nc_path

        gz_local_path = final_nc_path + ".gz"
        for attempt in range(1, self._max_retries + 1):
            try:
                with open(gz_local_path, 'wb') as local_file:
                    self._current_ftp.retrbinary(f"RETR {remote_name}", local_file.write)
                decompress_file(gz_local_path, final_nc_path)
                print(f"[OK] {remote_name}")
                return final_nc_path
            except (ftplib.error_temp, OSError) as exc:
                print(f"[RETRY {attempt}/{self._max_retries}] {remote_name}: {exc}")
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff * attempt)
            except Exception as file_err:
                print(f"[FAILED] {remote_name}: {file_err}")
                return None
            finally:
                if os.path.exists(gz_local_path):
                    try:
                        os.remove(gz_local_path)
                    except OSError:
                        pass

        return None

    # Returns the local files where they are downloaded
    def fetch_files(self, file_names: list[str]) -> list[str]:
        local_paths = []
        for file_name in file_names:
            local_paths.append(self.fetch_file(file_name))

        return local_paths


# --- Testing ---
if __name__ == "__main__":
    client = FtpClient(FTP_HOST, "username", "password", FTP_BASE_FOLDER)
    try:
        client.connect()
        client.disconnect()
    except Exception as err:
        print(err)
