import gzip
import os
import shutil


def decompress_file(gz_path: str, out_path: str) -> None:
    tmp_path = out_path + ".tmp"
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(tmp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # Rename is atomic on POSIX, on Windows it overwrites if needed or we use replace
        os.replace(tmp_path, out_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise Exception(f"Unzip error for {os.path.basename(gz_path)}: {e}") from e
