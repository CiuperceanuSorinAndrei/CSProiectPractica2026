"""Orchestrator: coordoneaza procesarea cadrelor si pastreaza starea intre ele.

Preprocesarea stateless (citire netCDF4 + crop + detectie) e delegata catre
frame_preprocessor si memoizata; aici se face partea secventiala (tracking Kalman +
metrici + volum) si se coordoneaza warm-up-ul de fundal. Starea cinematica e detinuta
de StormTracker; aici pastram doar masca globala prezisa pentru metricile CSI/FAR/POD.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
import threading
import time

import numpy as np

from src.core.storm_tracker import StormTracker
from src.core.forecast_metrics import ForecastMetrics
from frame_preprocessor import FrameGeometry, FramePrep, compute_geometry, preprocess

from config import RAIN_THRESHOLD_MIN, MAX_TRACKING_DISTANCE_PX


class ServerBusy(Exception):
    """Ridicata cand un alt cadru este deja in procesare (lock-ul orchestratorului e ocupat).

    Spre deosebire de o eroare reala (fisier corupt / in afara imaginii), care intoarce
    None, aceasta semnaleaza o stare TRANZITORIE: apelantul ar trebui sa sara peste update
    (PreventUpdate), nu sa afiseze o eroare in interfata.
    """


@dataclass
class FrameResult:
    """Rezultatul procesarii unui cadru complet."""
    tracked_cells: list[dict[str, Any]]
    rain_rate: np.ndarray
    rain_rate_masked: np.ma.MaskedArray
    lon_grid: np.ndarray
    lat_grid: np.ndarray
    max_rain: float
    mean_centroid_error: float
    mean_size_error: float
    num_tracked: int
    roi_volume_m3: float  # Volumul de precipitatii in ROI in metri cubi / ora
    predicted_roi_volume_m3: float  # Volumul prezis in ROI pentru cadrul urmator
    global_csi: float | None
    global_far: float | None
    global_pod: float | None


# Numarul maxim de cadre preprocesate pastrate in cache (LRU). La ~1 MB/cadru (zoom implicit)
# acopera intervale tipice pastrand memoria marginita.
_PREP_CACHE_MAXSIZE = 256

# Warm-up de fundal: cat asteapta dupa ultima activitate din fata inainte de a pregati
# urmatorul cadru (cedeaza prioritate interactiunii), verificand starea la fiecare poll.
_WARMUP_GRACE_S = 0.6
_WARMUP_POLL_S = 0.05


class Orchestrator:
    """Coordoneaza procesarea cadrelor si pastreaza starea cinematica intre ele."""

    def __init__(self) -> None:
        self._tracker = StormTracker(max_dist_pixels=MAX_TRACKING_DISTANCE_PX)
        self._lock = threading.Lock()
        self._previous_global_predicted_mask: Any = None

        # Cache geometrie (o singura intrare - doar geometria curenta conteaza) si cache
        # de preprocesare per fisier (LRU). Ambele sunt valide doar pentru geometria curenta
        # si se invalideaza la schimbarea ei (zoom/locatie/raza).
        self._geom_key: tuple | None = None
        self._geom: FrameGeometry | None = None
        self._prep_cache: OrderedDict[str, FramePrep] = OrderedDict()

        # Pre-incarcare progresiva in fundal (warm-up). Thread daemon care umple cache-ul de
        # preprocesare cand utilizatorul e inactiv, ca salturile reci sa devina rapide.
        self._last_activity: float = 0.0           # monotonic-ul ultimei activitati din fata
        self._warm_lock = threading.Lock()         # protejeaza ciclul de viata al thread-ului
        self._warm_thread: threading.Thread | None = None
        self._warm_stop: threading.Event | None = None
        self._warm_geom_key: tuple | None = None
        self._warm_complete_key: tuple | None = None
        self._warm_total: int = 0

    def reset_tracking(self) -> None:
        """Goleste complet starea de tracking (Kalman + masca globala prezisa)."""
        with self._lock:
            self._tracker.reset()
            self._previous_global_predicted_mask = None

    def process_frame(
        self,
        file_path: str,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float,
    ) -> FrameResult | None:
        """Proceseaza un cadru complet: citire -> crop -> detectie -> tracking.

        Etapa de preprocesare (citire/crop/proiectie/detectie) este stateless si memoizata:
        geometria (grile Lon/Lat, ROI, arie pixel) se calculeaza o singura data per geometrie
        (grid-reuse), iar rezultatul per-cadru se cache-uieste dupa numele fisierului. Doar
        tracking-ul (Kalman + Optical Flow) ramane secvential si ruleaza de fiecare data.
        """
        # Marcam activitate in fata (chiar si la coliziune): warm-up-ul de fundal cedeaza prioritate.
        self._last_activity = time.monotonic()
        if not self._lock.acquire(blocking=False):
            # Server ocupat (alt cadru in procesare) -> stare tranzitorie, nu eroare reala.
            raise ServerBusy()

        try:
            prep = self._get_frame_prep(
                file_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km,
            )
            if prep is None:
                return None  # eroare reala: fisier necitibil / in afara imaginii satelitului
            return self._track_and_assemble(prep, self._geom, center_lat, center_lon, radius_km)
        finally:
            self._lock.release()

    # ---- etapa stateless, memoizata (citire/crop/proiectie/detectie) ------
    def _get_frame_prep(
        self,
        file_path: str,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float,
    ) -> FramePrep | None:
        """Intoarce preprocesarea cadrului din cache sau o calculeaza (citind discul doar la
        miss). Invalideaza cache-urile cand geometria (bbox/centru/raza) se schimba."""
        geom_key = (lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
        if geom_key != self._geom_key:
            # Geometrie noua: vechile grile si preparari nu mai sunt valide.
            self._geom_key = geom_key
            self._geom = None
            self._prep_cache.clear()

        cached = self._prep_cache.get(file_path)
        if cached is not None and self._geom is not None:
            self._prep_cache.move_to_end(file_path)  # LRU: marcam drept recent folosit
            return cached

        return self._compute_prep(
            file_path, lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km,
        )

    def _compute_prep(
        self,
        file_path: str,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float,
    ) -> FramePrep | None:
        """Calculeaza preprocesarea cadrului (netCDF4 brut: citire fereastra + detectie) si o
        pune in cache. Geometria (grid-reuse) se calculeaza o singura data per geometrie.
        Presupune ca self._geom_key e deja stabilit de apelant (_get_frame_prep)."""
        bbox = (lon_min, lon_max, lat_min, lat_max)
        if self._geom is None:
            self._geom = compute_geometry(file_path, bbox, (center_lat, center_lon), radius_km)
            if self._geom is None:
                return None  # bbox in afara imaginii satelitului
        prep = preprocess(file_path, self._geom, bbox)
        if prep is None:
            return None  # fisier necitibil
        self._prep_cache[file_path] = prep
        if len(self._prep_cache) > _PREP_CACHE_MAXSIZE:
            self._prep_cache.popitem(last=False)  # evacuam cel mai vechi (LRU)
        return prep

    # ---- etapa secventiala (tracking + metrici + volum, NU se cache-uieste) ----
    def _track_and_assemble(self, prep: FramePrep, geom: FrameGeometry,
                            center_lat: float, center_lon: float, radius_km: float) -> FrameResult:
        """Tracking cinematic + metrici globale + volum. Depinde de cadrele anterioare
        (Kalman/Optical Flow/masca globala prezisa), deci ruleaza secvential per cadru."""
        rain_rate = prep.rain_rate
        roi_mask = geom.roi_mask
        pixel_area_km2 = geom.pixel_area_km2

        # Tracking cinematic hibrid (Kalman + Optical Flow). StormTracker gestioneaza intern
        # resetarea la schimbarea rezolutiei grilei (zoom in/out pe harta). Lucram pe copii
        # superficiale ca tracking-ul sa nu mute celulele memoizate (ex. _cached_mask).
        cells_for_tracking = [dict(c) for c in prep.filtered_cells]
        tracked_cells = self._tracker.track(cells_for_tracking, rain_rate)

        # --- CALCUL METRICI GLOBALE (doar in ROI) ---
        global_csi, global_far, global_pod = None, None, None
        prev_global = self._previous_global_predicted_mask
        if prev_global is not None and prev_global.shape == rain_rate.shape:
            obs_mask = (rain_rate >= RAIN_THRESHOLD_MIN) & roi_mask
            pred_mask = prev_global & roi_mask

            # Inregistram metrici doar daca exista activitate in ROI (ploaie observata sau prezisa)
            if np.any(obs_mask) or np.any(pred_mask):
                global_csi = ForecastMetrics.csi(obs_mask, pred_mask)
                global_far = ForecastMetrics.far(obs_mask, pred_mask)
                global_pod = ForecastMetrics.pod(obs_mask, pred_mask)

        # 1 mm/h ploaie = 0.25 mm/15min = 250 m^3/km^2 acumulati in 15 minute
        valid_rain_mask = (rain_rate >= RAIN_THRESHOLD_MIN) & roi_mask
        roi_volume_m3 = float(np.sum(rain_rate[valid_rain_mask] * pixel_area_km2[valid_rain_mask] * 250.0))

        # Calcul Volum Prezis Acumulat in ROI pentru urmatoarele N cadre (Max 3 ore = 12 cadre)
        MAX_FORECAST_FRAMES = 12
        total_predicted_volume_m3 = 0.0

        for cell in tracked_cells:
            # Verificam daca centroizii curenti sunt in ROI
            c_dist = self._haversine_dist_grid(
                center_lat, center_lon, np.array([cell["geo_lat"]]), np.array([cell["geo_lon"]]),
            )[0]
            cell["in_roi"] = bool(c_dist <= radius_km)

            # Simulam si integram volumul pe traiectoria viitoare
            if cell.get("is_tracked", False) and "coords" in cell:
                mean_int = cell.get("mean_intensity", 0.0)
                vx = cell.get("v_x", 0.0)
                vy = cell.get("v_y", 0.0)

                # Reconstruim masca curenta
                c_mask = np.zeros(rain_rate.shape, dtype=bool)
                for cy, cx in cell["coords"]:
                    c_mask[int(cy), int(cx)] = True

                # Daca sta pe loc complet (viteza aproape 0), previne bucla artificiala pt 3 ore
                if abs(vx) < 0.1 and abs(vy) < 0.1:
                    frames_to_sim = 1
                else:
                    frames_to_sim = MAX_FORECAST_FRAMES

                for step in range(1, frames_to_sim + 1):
                    # translate_mask(mask, shift_y, shift_x): vy pe axa y (randuri), vx pe axa x (coloane)
                    future_mask = StormTracker.translate_mask(c_mask, step * vy, step * vx)
                    # astype(bool) evita fancy-indexing pe pixel_area_km2 (masca e uint8)
                    overlap_mask = future_mask.astype(bool) & roi_mask
                    num_pixels_overlap = np.sum(overlap_mask)

                    if num_pixels_overlap > 0:
                        area_overlap_km2 = np.sum(pixel_area_km2[overlap_mask])
                        vol_step_m3 = float(area_overlap_km2 * mean_int * 250.0)
                        total_predicted_volume_m3 += vol_step_m3
                    elif step > 1 and total_predicted_volume_m3 > 0:
                        # Am iesit deja din ROI, n-are rost sa simulam restul cadrelor
                        break

        # Construim noua masca globala prezisa pentru T+1
        new_global_pred_mask = np.zeros(rain_rate.shape, dtype=bool)
        for cell in tracked_cells:
            if "predicted_mask" in cell and cell.get("is_tracked", False):
                new_global_pred_mask |= cell["predicted_mask"].astype(bool)

        # Salvam starea globala pentru cadrul urmator
        self._previous_global_predicted_mask = new_global_pred_mask

        # Calcul metrici de eroare pe celulele urmarite
        valid_errors = [
            c["prediction_error_pixels"]
            for c in tracked_cells
            if c.get("is_tracked", False)
        ]
        size_errors = [
            c.get("size_error_percent", 0.0)
            for c in tracked_cells
            if c.get("is_tracked", False)
        ]

        # rain_rate_masked e doar pentru randare; il reconstruim ieftin (nu merita cache-uit)
        rain_rate_masked = np.ma.masked_where(rain_rate < RAIN_THRESHOLD_MIN, rain_rate)
        return FrameResult(
            tracked_cells=tracked_cells,
            rain_rate=rain_rate,
            rain_rate_masked=rain_rate_masked,
            lon_grid=geom.lon_grid,
            lat_grid=geom.lat_grid,
            max_rain=prep.max_rain,
            mean_centroid_error=float(np.mean(valid_errors)) if valid_errors else 0.0,
            mean_size_error=float(np.mean(size_errors)) if size_errors else 0.0,
            num_tracked=len(valid_errors),
            roi_volume_m3=roi_volume_m3,
            predicted_roi_volume_m3=total_predicted_volume_m3,
            global_csi=global_csi,
            global_far=global_far,
            global_pod=global_pod,
        )

    # ---- pre-incarcare progresiva in fundal (warm-up) ---------------------
    def start_warmup(
        self,
        file_paths: list[str],
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
        center_lat: float, center_lon: float, radius_km: float,
    ) -> None:
        """Porneste (sau actualizeaza) un thread daemon care preproceseaza, cand utilizatorul
        e inactiv, toate cadrele din interval pentru geometria curenta - ca salturile reci sa
        devina rapide. Idempotent: nu reporneste daca geometria e deja in lucru sau completa."""
        geom_key = (lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
        with self._warm_lock:
            alive = self._warm_thread is not None and self._warm_thread.is_alive()
            if self._warm_complete_key == geom_key or (self._warm_geom_key == geom_key and alive):
                return
            # Oprim un eventual warm-up vechi (alta geometrie) inainte de a porni unul nou.
            if self._warm_stop is not None:
                self._warm_stop.set()
            stop = threading.Event()
            geom_args = (lon_min, lon_max, lat_min, lat_max, center_lat, center_lon, radius_km)
            self._warm_stop = stop
            self._warm_geom_key = geom_key
            self._warm_total = len(file_paths)
            self._warm_thread = threading.Thread(
                target=self._warmup_loop,
                args=(list(file_paths), geom_args, geom_key, stop),
                daemon=True,
            )
            self._warm_thread.start()

    def stop_warmup(self) -> None:
        """Opreste warm-up-ul curent (ex. la trecerea in modul LIVE)."""
        with self._warm_lock:
            if self._warm_stop is not None:
                self._warm_stop.set()
            self._warm_thread = None
            self._warm_geom_key = None
            self._warm_complete_key = None
            self._warm_total = 0

    def warm_status(self) -> tuple[int, int]:
        """(cadre pregatite, total) pentru afisarea progresului warm-up-ului."""
        total = self._warm_total
        if total <= 0:
            return 0, 0
        return min(len(self._prep_cache), total), total

    def _warmup_loop(self, file_paths: list[str], geom_args: tuple, geom_key: tuple,
                     stop: threading.Event) -> None:
        """Bucla de fundal: pregateste cate un cadru, dar numai cand utilizatorul e inactiv
        (cedeaza prioritate) si fara sa atinga starea de tracking."""
        i = 0
        while i < len(file_paths):
            if stop.is_set():
                return
            # Cedam prioritate: cat timp a existat activitate recenta in fata, asteptam.
            if time.monotonic() - self._last_activity < _WARMUP_GRACE_S:
                time.sleep(_WARMUP_POLL_S)
                continue
            # Prindem lock-ul fara a bloca; daca e ocupat (frontul lucreaza), reincercam mai tarziu.
            if not self._lock.acquire(blocking=False):
                time.sleep(_WARMUP_POLL_S)
                continue
            try:
                if not self._warm_one(file_paths[i], geom_key, geom_args):
                    return  # geometria s-a schimbat -> warm-up-ul nu mai e valid
            finally:
                self._lock.release()
            i += 1

        # Am terminat de pregatit tot intervalul: marcam ca sa nu repornim degeaba la fiecare update.
        with self._warm_lock:
            if not stop.is_set():
                self._warm_complete_key = geom_key

    def _warm_one(self, file_path: str, geom_key: tuple, geom_args: tuple) -> bool:
        """Pregateste un cadru pentru warm-up (apelat sub self._lock). Intoarce False daca
        geometria s-a schimbat intre timp (apelantul trebuie sa se opreasca)."""
        if geom_key != self._geom_key:
            return False
        if file_path not in self._prep_cache:
            # Ignoram None (fisier necitibil): il va reincerca vizita reala din fata.
            self._compute_prep(file_path, *geom_args)
        return True

    # Distanta Haversine (km) intre un punct fix si un grid de puncte
    @staticmethod
    def _haversine_dist_grid(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c
