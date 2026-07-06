import gzip
import os
import shutil


def decompress_file(gz_path: str, out_path: str) -> None:
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    except Exception as e:
        raise Exception(f"Eroare la unzip {os.path.basename(gz_path)}: {e}") from e
