# app/tools/weather.py
from semantic_kernel.functions import kernel_function
import requests
import json
from app.utils.logger import setup_logger

logger = setup_logger("weather_tool")

class WeatherTools:
    @kernel_function(name="get_weather", description="Get 7-day weather forecast for a given city.")
    def get_weather(self, city: str) -> str:
        """
        Gets the 7-day weather forecast for a given city and returns a simple summary string.

        Fetch and summarize a seven-day forecast from Open-Meteo.
        """
        logger.info(f"Weather tool called with city={city}")

        city = city.strip()
        if not city:
            raise ValueError("City must not be empty")

        try:
            geocoding_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            geocoding_response.raise_for_status()
            locations = geocoding_response.json().get("results") or []
            if not locations:
                raise ValueError(f"No location found for {city!r}")

            location = locations[0]
            weather_response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                    "forecast_days": 7,
                    "timezone": "UTC",
                },
                timeout=10,
            )
            weather_response.raise_for_status()
            daily = weather_response.json()["daily"]

            highs = daily.get("temperature_2m_max") or []
            lows = daily.get("temperature_2m_min") or []
            codes = daily.get("weather_code") or []
            if not highs or not lows or not codes:
                raise ValueError("Open-Meteo returned an incomplete forecast")

            average_temperature = sum(
                (float(high) + float(low)) / 2
                for high, low in zip(highs, lows)
            ) / min(len(highs), len(lows))
            code = max(set(codes), key=codes.count)
            if code <= 1:
                conditions = "Sunny"
            elif code <= 3:
                conditions = "Cloudy"
            elif code >= 95:
                conditions = "Thunderstorms"
            elif code >= 71:
                conditions = "Snow"
            elif code >= 51:
                conditions = "Rain"
            elif code in (45, 48):
                conditions = "Fog"
            else:
                conditions = "Mixed"

            summary = {
                "city": location.get("name", city),
                "country": location.get("country"),
                "forecast_days": len(codes),
                "average_temperature_c": round(average_temperature, 1),
                "conditions": conditions,
            }
            return json.dumps(
                {
                    **summary,
                    "summary": (
                        f"{summary['city']}: {summary['average_temperature_c']}°C "
                        f"average, {conditions} over the next {len(codes)} days"
                    ),
                },
                ensure_ascii=False,
            )
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            logger.error("Weather lookup failed for %s: %s", city, exc)
            raise RuntimeError(f"Unable to retrieve weather for {city}: {exc}") from exc
