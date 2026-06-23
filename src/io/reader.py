import xarray as xr
import os

def load_hsaf_data(file_path: str) -> xr.Dataset:
    """Incarca un fisier NetCDF (.nc) in memorie."""
    if not os.path.exists(file_path):
        print(f"Eroare: Fisierul nu exista la: {file_path}")
        return None
        
    try:
        return xr.open_dataset(file_path, engine='netcdf4')
    except Exception as e:
        print(f"Eroare la citirea NetCDF: {e}")
        return None

if __name__ == "__main__":
    nume_fisier = "h60_20260613_1400_fdk.nc" 
    cale_test = os.path.join("data", "raw", nume_fisier)
    
    ds = load_hsaf_data(cale_test)
    if ds is not None:
        print("Fisier citit cu succes.")
        print("Dimensiuni:", ds.dims)
        print("Coordonate:", list(ds.coords))
        print("Variabile de date:", list(ds.data_vars))