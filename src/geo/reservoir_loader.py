import os
import threading
import shapefile
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

class ReservoirLoader:
    _cache = None
    _lock = threading.Lock()

    @staticmethod
    def get_all_reservoirs(shapefile_path="data/geo/reservoirs/LacuriAcumulare.shp") -> dict:
        """Parseaza shapefile-ul cu lacurile de acumulare si extrage geometriile.
        
        Returneaza un dictionar { "Nume Lac": { "polygon": shapely.Geometry, "center": (lat, lon), "radius_km": float, "bounds": (min_lon, min_lat, max_lon, max_lat) } }
        """
        if ReservoirLoader._cache is not None:
            return ReservoirLoader._cache
            
        with ReservoirLoader._lock:
            if ReservoirLoader._cache is not None:
                return ReservoirLoader._cache

            if not os.path.exists(shapefile_path):
                return {}

        reservoirs = {}
        try:
            sf = shapefile.Reader(shapefile_path)
            records = sf.records()
            shapes = sf.shapes()
            
            # Proiectia shapefile-ului este Stereo 70 (EPSG:31700), noi vrem Lat/Lon (EPSG:4326)
            proj_transformer = Transformer.from_crs("EPSG:31700", "EPSG:4326", always_xy=True)

            for rec, shp in zip(records, shapes):
                entry = ReservoirLoader._build_reservoir_entry(rec, shp, proj_transformer, reservoirs)
                if entry is not None:
                    name, data = entry
                    reservoirs[name] = data

            sf.close()
            ReservoirLoader._cache = reservoirs
        except Exception as e:
            print(f"Eroare la incarcarea lacurilor de acumulare: {e}")

        return ReservoirLoader._cache if ReservoirLoader._cache else {}

    @staticmethod
    def _build_reservoir_entry(rec, shp, transformer, existing: dict):
        """Construieste o intrare (nume, date) dintr-un record+shape; None daca lipseste numele sau e gol.

        Reproiecteaza geometria din Stereo 70 (EPSG:31700) in WGS84 si dezambiguizeaza numele
        duplicate consultand `existing` (lacurile deja adaugate).
        """
        denumire = rec.denumire
        if not denumire or str(denumire).strip() == "":
            return None

        name = str(denumire).strip().title()

        # Evitam duplicatele (daca sunt mai multe poligoane cu acelasi nume)
        original_name = name
        idx = 2
        while name in existing:
            name = f"{original_name} {idx}"
            idx += 1

        geom_stereo = shape(shp)
        if geom_stereo.is_empty:
            return None

        # Transformam poligonul din metri (Stereo 70) in grade (WGS84)
        geom = transform(transformer.transform, geom_stereo)

        bounds = geom.bounds  # (min_lon, min_lat, max_lon, max_lat)

        # Calculam o raza acoperitoare aproximativa (latimea/inaltimea in km)
        delta_lon = bounds[2] - bounds[0]
        delta_lat = bounds[3] - bounds[1]
        radius_km = max(delta_lon * 80.0, delta_lat * 111.0) / 2.0 + 5.0  # +5km padding

        data = {
            "name": name,
            "polygon": geom,
            "bounds": bounds,
            "center": (geom.centroid.y, geom.centroid.x),  # (lat, lon)
            "radius_km": radius_km
        }
        return name, data
