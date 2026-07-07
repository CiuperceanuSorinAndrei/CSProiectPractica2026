"""Service for fetching historical meteorological data from Open-Meteo."""
from datetime import date, datetime
import requests
import logging

logger = logging.getLogger(__name__)

class MeteoService:
    @staticmethod
    def fetch_historical_gap(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> dict:
        """Fetches accumulated precipitation and evaporation between start_dt and end_dt.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_dt: The date of the last known satellite measurement.
            end_dt: The start date of the current radar sequence.
            
        Returns:
            Dictionary with 'precipitation_mm' and 'evaporation_mm' totals for the gap.
        """
        # If dates are the same or gap is negative, no gap to fill
        start_d = start_dt.date()
        end_d = end_dt.date()
        if start_d >= end_d:
            return {"precipitation_mm": 0.0, "evaporation_mm": 0.0}

        try:
            url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&start_date={start_d.isoformat()}&end_date={end_d.isoformat()}"
                f"&daily=precipitation_sum,et0_fao_evapotranspiration"
                f"&timezone=auto"
            )
            logger.info(f"Fetching historical gap data from Open-Meteo: {url}")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            daily = data.get("daily", {})
            precip = sum([p for p in daily.get("precipitation_sum", []) if p is not None])
            evap = sum([e for e in daily.get("et0_fao_evapotranspiration", []) if e is not None])
            
            logger.info(f"Historical gap {start_d} to {end_d}: {precip:.1f}mm rain, {evap:.1f}mm evap.")
            return {
                "precipitation_mm": precip,
                "evaporation_mm": evap
            }
        except Exception as e:
            logger.error(f"Failed to fetch historical gap data: {e}")
            return {"precipitation_mm": 0.0, "evaporation_mm": 0.0}
