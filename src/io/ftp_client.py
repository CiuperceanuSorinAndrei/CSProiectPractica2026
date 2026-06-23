import os
import ftplib
import gzip
import shutil
from datetime import datetime, timedelta
from config import FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_FOLDER, DATA_RAW_DIR

def decompress_gz_file(gz_path: str, out_path: str) -> bool:
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        print(f"Eroare la unzip {os.path.basename(gz_path)}: {e}")
        return False

def fetch_hsaf_files(file_names: list) -> list:
    local_paths = []
    files_to_download = []

    # Caching 
    for filename in file_names:
        if filename.endswith('.gz'):
            unzipped_filename = filename[:-3]
            remote_name = filename
        else:
            unzipped_filename = filename
            remote_name = filename + '.gz'
            
        final_nc_path = os.path.join(DATA_RAW_DIR, unzipped_filename)
        
        if os.path.exists(final_nc_path):
            print(f"[SKIP] Găsit local: {unzipped_filename}")
            local_paths.append(final_nc_path)
        else:
            files_to_download.append((remote_name, final_nc_path))

    # Pipeline-ul de descărcare
    if files_to_download:
        print(f"Conectare la FTP {FTP_HOST}...")
        try:
            ftp = ftplib.FTP(FTP_HOST)
            ftp.login(user=FTP_USER, passwd=FTP_PASS)
            ftp.cwd(FTP_BASE_FOLDER)
            
            for remote_filename, final_nc_path in files_to_download:
                gz_local_path = final_nc_path + ".gz"
                print(f"Descarcă: {remote_filename} ... ", end="", flush=True)
                
                try:
                    with open(gz_local_path, 'wb') as local_file:
                        ftp.retrbinary(f"RETR {remote_filename}", local_file.write)
                    print("OK -> [GZIP] Extract ... ", end="", flush=True)
                    
                    if decompress_gz_file(gz_local_path, final_nc_path):
                        print("DONE")
                        local_paths.append(final_nc_path)
                    os.remove(gz_local_path) 
                except Exception as file_err:
                    print(f"EȘUAT ({file_err})")
                    if os.path.exists(gz_local_path): os.remove(gz_local_path)
                    
            ftp.quit()
        except Exception as e:
            print(f"Eroare generală FTP: {e}")
            
    return local_paths

def download_historical_period(year: int, month: int, day: int, start_hour: int, end_hour: int) -> list:
    print(f"\n[BATCH] Pornire descărcări pentru data: {day:02d}/{month:02d}/{year}")
    target_files = []
    minutes = [0, 15, 30, 45]
    
    for hour in range(start_hour, end_hour + 1):
        for minute in minutes:
            if hour == end_hour and minute > 0:
                break
                
            # Formatul standard EUMETSAT H60: h60_YYYYMMDD_HHMM_fdk.nc.gz
            filename = f"h60_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}_fdk.nc.gz"
            target_files.append(filename)
            
    return fetch_hsaf_files(target_files)

# --- Testing ---
if __name__ == "__main__":
    cai_salvate = download_historical_period(year=2026, month=6, day=13, start_hour=14, end_hour=15)
    print(f"\nDownloader complet! ({len(cai_salvate)} fișiere).")