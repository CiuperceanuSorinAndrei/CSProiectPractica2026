"""Estimarea gradului de umplere a unui lac de acumulare din precipitatia acumulata.

Precipitatia medie areala (MAP) urmarita de aplicatie este o *adancime* de apa
(mm, echivalent cu L/m^2). Inmultind adancimea cu suprafata luciului de apa obtinem
volumul de apa cazut direct pe lac (m^3); raportat la volumul maxim al acumularii
(campul `vol_mil_m3` din shapefile, volumul la Nivelul Normal de Retentie) rezulta
procentul din capacitate.

    volum_apa_m3 = (map_mm / 1000) * suprafata_m2
    procent      = 100 * volum_apa_m3 / volum_maxim_m3

Clasa este stateless (doar metode statice pe scalari), deci este usor de testat izolat,
fara shapefile sau Dash. Sursa suprafetei si a volumului maxim este `ReservoirLoader`,
care ataseaza `surface_area_m2` si `max_volume_m3` fiecarui lac.
"""
from __future__ import annotations

MM_TO_M = 1.0e-3  # 1 mm de precipitatie = 1 L/m^2 = 0.001 m adancime


class ReservoirFillEstimator:
    """Converteste MAP acumulat (L/m^2) in volum (m^3) si procent din volumul maxim al lacului."""

    @staticmethod
    def accumulated_volume_m3(map_mm: float, surface_area_m2: float) -> float:
        """Volumul de apa (m^3) rezultat dintr-o adancime `map_mm` (mm) peste `surface_area_m2` (m^2).

        Intoarce 0.0 pentru intrari lipsa sau nepozitive (nicio ploaie / arie necunoscuta).
        """
        if not map_mm or not surface_area_m2 or map_mm <= 0.0 or surface_area_m2 <= 0.0:
            return 0.0
        return map_mm * MM_TO_M * surface_area_m2

    @staticmethod
    def fill_percentage(map_mm: float, surface_area_m2: float, max_volume_m3: float) -> float | None:
        """Procentul din volumul maxim acumulat de o adancime `map_mm` peste `surface_area_m2`.

        Intoarce None cand capacitatea maxima lipseste sau este nepozitiva (nu putem raporta
        la un volum necunoscut). Nu se plafoneaza la 100%: un episod extrem poate depasi
        capacitatea pe hartie, iar valoarea reala este mai informativa decat o trunchiere tacuta.
        """
        if not max_volume_m3 or max_volume_m3 <= 0.0:
            return None
        volume_m3 = ReservoirFillEstimator.accumulated_volume_m3(map_mm, surface_area_m2)
        return 100.0 * volume_m3 / max_volume_m3

    @staticmethod
    def fill_percentage_for(map_mm: float, reservoir: dict | None) -> float | None:
        """Varianta convenabila: extrage `surface_area_m2` si `max_volume_m3` dintr-o intrare
        `ReservoirLoader`. Intoarce None cand nu exista lac selectat sau ii lipsesc datele."""
        if not reservoir:
            return None
        return ReservoirFillEstimator.fill_percentage(
            map_mm,
            reservoir.get("surface_area_m2", 0.0),
            reservoir.get("max_volume_m3", 0.0),
        )
