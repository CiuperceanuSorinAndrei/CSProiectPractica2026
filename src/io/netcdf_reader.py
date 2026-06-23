import xarray as xr
import os

class NetCdfReader:
    _file_path: str = None

    def __init__(self, file_path: str | None = None):
        self.set_file_path(file_path)

    def set_file_path(self, file_path: str | None = None):
        if file_path is not None and not os.path.exists(file_path):
            print(f"Eroare: Fisierul nu exista la: {file_path}")
            return
        self._file_path = file_path

    def load_data(self) -> xr.Dataset:
        """Incarca un fisier NetCDF (.nc) in memorie."""
        try:
            return xr.open_dataset(self._file_path, engine='netcdf4')
        except Exception as e:
            print(f"Eroare la citirea NetCDF: {e}")
            return None

if __name__ == "__main__":
    nume_fisier = "h60_20260613_1400_fdk.nc" 
    cale_test = os.path.join("data", "raw", nume_fisier)

    reader = NetCdfReader(cale_test)

    ds = reader.load_data()
    if ds is not None:
        print("Fisier citit cu succes.")
        print("Dimensiuni:", ds.dims)
        print("Coordonate:", list(ds.coords))
        print("Variabile de date:", list(ds.data_vars))