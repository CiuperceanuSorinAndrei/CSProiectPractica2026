import os
import shapefile
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

class ReservoirLoader:
    _cache = None

    @staticmethod
    def get_all_reservoirs(shapefile_path="data/geo/reservoirs/LacuriAcumulare.shp") -> dict:
        """Parseaza shapefile-ul cu lacurile de acumulare si extrage geometriile.
        
        Returneaza un dictionar { "Nume Lac": { "polygon": shapely.Geometry, "center": (lat, lon), "radius_km": float, "bounds": (min_lon, min_lat, max_lon, max_lat) } }
        """
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
                denumire = rec.denumire
                if not denumire or str(denumire).strip() == "":
                    continue
                    
                name = str(denumire).strip().title()
                
                # Evitam duplicatele (daca sunt mai multe poligoane cu acelasi nume)
                original_name = name
                idx = 2
                while name in reservoirs:
                    name = f"{original_name} {idx}"
                    idx += 1

                geom_stereo = shape(shp)
                if geom_stereo.is_empty:
                    continue
                    
                # Transformam poligonul din metri (Stereo 70) in grade (WGS84)
                geom = transform(proj_transformer.transform, geom_stereo)

                bounds = geom.bounds  # (min_lon, min_lat, max_lon, max_lat)

                # Calculam o raza acoperitoare aproximativa (latimea/inaltimea in km)
                delta_lon = bounds[2] - bounds[0]
                delta_lat = bounds[3] - bounds[1]
                radius_km = max(delta_lon * 80.0, delta_lat * 111.0) / 2.0 + 5.0  # +5km padding

                reservoirs[name] = {
                    "name": name,
                    "polygon": geom,
                    "bounds": bounds,
                    "center": (geom.centroid.y, geom.centroid.x),  # (lat, lon)
                    "radius_km": radius_km
                }
                
            sf.close()
            ReservoirLoader._cache = reservoirs
        except Exception as e:
            print(f"Eroare la incarcarea lacurilor de acumulare: {e}")
            
        return ReservoirLoader._cache if ReservoirLoader._cache else {}
