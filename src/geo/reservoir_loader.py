import os
import threading
import shapefile
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

class ReservoirLoader:
    _cache = None
    _lock = threading.Lock()
    _DEM_AUGMENT_PATH = "data/geo/reservoirs/dem_augment.json"
    _LEVELS_PATH = "data/geo/reservoirs/reservoir_levels.json"          # SWOT (lac + rau)
    _LEVELS_S2_PATH = "data/geo/reservoirs/reservoir_levels_s2.json"    # Sentinel-2 (optic)

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
            ReservoirLoader._augment_with_dem(reservoirs)
            ReservoirLoader._cache = reservoirs
        except Exception as e:
            print(f"Eroare la incarcarea lacurilor de acumulare: {e}")

        return ReservoirLoader._cache if ReservoirLoader._cache else {}

    @staticmethod
    def get_covered_reservoirs(shapefile_path="data/geo/reservoirs/LacuriAcumulare.shp") -> dict:
        """Setul folosit de aplicatie: doar lacurile cu nivel curent din SWOT (scopul proiectului).

        Cade pe setul complet daca RESERVOIRS_SWOT_COVERED_ONLY e False sau daca nu exista date SWOT
        (ca aplicatia sa nu ramana fara lacuri inainte de a rula build_reservoir_levels.py).
        """
        from config import RESERVOIRS_SWOT_COVERED_ONLY
        all_res = ReservoirLoader.get_all_reservoirs(shapefile_path)
        if not RESERVOIRS_SWOT_COVERED_ONLY:
            return all_res
        covered = {n: r for n, r in all_res.items() if r.get("level_source") in ("swot", "s2")}
        return covered if covered else all_res

    @staticmethod
    def _load_json(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            import json
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            print(f"Nu s-a putut citi {path}: {e}")
            return {}

    @staticmethod
    def _augment_with_dem(reservoirs: dict) -> None:
        """Ataseaza fiecarui lac curba stage-storage, suprafata bazinului si nivelul curent.

        Curba + bazinul vin din dem_augment.json (precalculat din DEM) unde exista, altfel o curba
        parametrica din atribute. Curba se extinde apoi sub NNR (partea submersa) ca sa poata porni
        de la un nivel curent. Nivelul curent vine din reservoir_levels.json (SWOT); fara el, se
        porneste implicit de la NNR. Astfel aplicatia merge si cu date partiale.
        """
        from src.geo.stage_storage import StageStorageCurve

        aug = ReservoirLoader._load_json(ReservoirLoader._DEM_AUGMENT_PATH)
        # Nivele curente: SWOT (primar) + Sentinel-2 (umple golurile pentru lacurile montane);
        # SWOT are prioritate acolo unde exista ambele.
        levels = {**ReservoirLoader._load_json(ReservoirLoader._LEVELS_S2_PATH),
                  **ReservoirLoader._load_json(ReservoirLoader._LEVELS_PATH)}

        for name, r in reservoirs.items():
            entry = aug.get(name)
            if entry:
                curve = StageStorageCurve.from_dict(entry["stage_storage"])
                r["catchment_km2"] = entry.get("catchment_km2")
                r["catchment_flag"] = entry.get("catchment_flag")
            else:
                wl = r.get("waterline_attr_m") or 0.0
                curve = StageStorageCurve.from_attributes(
                    r["max_volume_m3"], r["surface_area_m2"], wl if wl > 0 else 0.0)
                r["catchment_km2"] = None
                r["catchment_flag"] = "not_built"

            curve = curve.with_submerged_branch(r["surface_area_m2"])
            r["stage_storage"] = curve
            ReservoirLoader._attach_current_level(r, curve, levels.get(name))

    @staticmethod
    def _attach_current_level(r: dict, curve, lvl: dict | None) -> None:
        """Stabileste volumul de pornire din nivelul SWOT (cota luciului -> volum pe curba),
        sau None (estimatorul porneste de la NNR) daca lacul nu are observatie SWOT."""
        v_nnr = r.get("max_volume_m3") or 0.0
        if lvl and lvl.get("wse_m") is not None and v_nnr > 0:
            wse = float(lvl["wse_m"])
            # Volumul curent se plafoneaza la NNR: starea de exploatare e cel mult "plin" (100%).
            # Sub luciul apei curba (submersa) da fractiunea reala; peste NNR (banda de atenuare,
            # volum urias pentru lacuri de campie) nu e o stare de pornire plauzibila din SWOT.
            v0 = min(max(curve.volume_for_wse(wse), 0.0), v_nnr)
            r["current_volume_m3"] = v0
            r["current_fill_frac"] = v0 / v_nnr
            r["current_wse_m"] = wse
            r["level_source"] = lvl.get("source", "swot")
            r["level_product"] = lvl.get("product")     # "lake" | "river"
            r["level_as_of"] = lvl.get("as_of")
        else:
            r["current_volume_m3"] = None      # -> estimatorul porneste de la NNR
            r["current_fill_frac"] = None
            r["current_wse_m"] = None
            r["level_source"] = "assumed_nnr"
            r["level_as_of"] = None

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

        # Suprafata luciului de apa: aria poligonului in CRS-ul nativ al shapefile-ului
        # (Stereo 70, unitate metrul), deci direct in m^2. Coincide cu campul `suprafata_`
        # (in km^2) inmultit cu 1e6, dar geometria e mereu disponibila si consistenta cu
        # masca poligonala folosita la calculul MAP.
        surface_area_m2 = float(geom_stereo.area)

        # Volumul maxim (la Nivelul Normal de Retentie), stocat in shapefile in milioane m^3.
        vol_mil_m3 = ReservoirLoader._safe_float(rec, "vol_mil_m3")
        max_volume_m3 = vol_mil_m3 * 1.0e6 if vol_mil_m3 else 0.0

        # Cota luciului de apa din atribut (poate fi 0/negativa/eronata); folosita doar ca
        # rezerva pentru waterline cand DEM-ul nu e disponibil - cota DEM e preferata.
        waterline_attr_m = ReservoirLoader._safe_float(rec, "elevatie")

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
            "radius_km": radius_km,
            "surface_area_m2": surface_area_m2,
            "vol_mil_m3": vol_mil_m3,
            "max_volume_m3": max_volume_m3,
            "waterline_attr_m": waterline_attr_m,
        }
        return name, data

    @staticmethod
    def _safe_float(rec, field: str) -> float:
        """Citeste un camp numeric dintr-un record pyshp, tolerand valori lipsa/nevalide (-> 0.0)."""
        try:
            value = rec[field]
        except (KeyError, IndexError, TypeError):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
