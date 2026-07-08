import os
import time
import ftplib
import logging

from src.config import FTP_HOST, FTP_BASE_FOLDER, DATA_RAW_DIR
from src.io.gz_decompressor import decompress_file

logger = logging.getLogger(__name__)

class FtpClient:
    # 1. Initialization
    def __init__(self, host: str, username: str, password: str, base_dir: str, local_dir: str = DATA_RAW_DIR, timeout: int = 30, max_retries: int = 3, retry_backoff: float = 2.0, allow_plaintext_fallback: bool = False):
        self._host, self._username, self._password = host, username, password
        self._base_dir, self._local_dir = base_dir, local_dir
        self._timeout, self._max_retries, self._retry_backoff = timeout, max_retries, retry_backoff
        self._allow_plaintext_fallback = allow_plaintext_fallback
        self._current_ftp = None

    # 2. Connection Management
    def connect(self):
        logger.info(f"Connecting to FTP {self._host}...")
        try:
            self._current_ftp = ftplib.FTP_TLS(self._host, timeout=self._timeout)
            self._current_ftp.login(user=self._username, passwd=self._password)
            self._current_ftp.prot_p()
        except ftplib.error_perm as e:
            if "Login incorrect" in str(e) or "Authentication failed" in str(e): raise
            if not self._allow_plaintext_fallback: raise
            self._fallback_connect()
        except (ftplib.error_temp, EOFError, OSError):
            if not self._allow_plaintext_fallback: raise
            self._fallback_connect()
        self._current_ftp.cwd(self._base_dir)

    def _fallback_connect(self):
        try: self._current_ftp.close()
        except Exception: pass
        self._current_ftp = ftplib.FTP(self._host, timeout=self._timeout)
        self._current_ftp.login(user=self._username, passwd=self._password)

    def disconnect(self):
        self._current_ftp.quit()
        self._current_ftp = None

    # 3. File Operations
    def file_size(self, file_name: str) -> int | None:
        try: return self._current_ftp.size(file_name)
        except Exception: return None

    def fetch_file(self, file_name: str) -> str | None:
        unzipped = file_name[:-3] if file_name.endswith('.gz') else file_name
        remote = file_name if file_name.endswith('.gz') else file_name + '.gz'

        os.makedirs(self._local_dir, exist_ok=True)
        final_path = os.path.join(self._local_dir, unzipped)
        if os.path.exists(final_path) and os.path.getsize(final_path) > 1024: return final_path

        gz_path = final_path + ".gz"
        for attempt in range(1, self._max_retries + 1):
            try:
                with open(gz_path, 'wb') as lf: self._current_ftp.retrbinary(f"RETR {remote}", lf.write)
                decompress_file(gz_path, final_path)
                return final_path
            except (ftplib.error_temp, OSError):
                if attempt < self._max_retries: time.sleep(self._retry_backoff * attempt)
            except Exception: return None
            finally:
                if os.path.exists(gz_path):
                    try: os.remove(gz_path)
                    except OSError: pass
        return None

    def fetch_files(self, file_names: list[str]) -> list[str]:
        return [self.fetch_file(fn) for fn in file_names]
