import requests
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Coordinates for districts in Maharashtra State
MAHARASHTRA_DISTRICT_COORDS: Dict[str, Tuple[float, float]] = {
    "Ahmednagar": (19.0948, 74.7480),
    "Ahilyanagar": (19.0948, 74.7480),
    "Akola": (20.7002, 77.0082),
    "Amravati": (20.9374, 77.7796),
    "Aurangabad": (19.8762, 75.3433),
    "Chhatrapati Sambhajinagar": (19.8762, 75.3433),
    "Beed": (18.9891, 75.7601),
    "Bhandara": (21.1704, 79.6547),
    "Buldhana": (20.5293, 76.1843),
    "Chandrapur": (19.9615, 79.2961),
    "Dhule": (20.9042, 74.7749),
    "Gadchiroli": (20.1849, 79.9972),
    "Gondia": (21.4603, 80.1961),
    "Hingoli": (19.7171, 77.1477),
    "Jalgaon": (21.0077, 75.5626),
    "Jalna": (19.8410, 75.8864),
    "Kolhapur": (16.7050, 74.2433),
    "Latur": (18.4088, 76.5604),
    "Mumbai City": (18.9388, 72.8353),
    "Mumbai Suburban": (19.0760, 72.8777),
    "Nagpur": (21.1458, 79.0882),
    "Nanded": (19.1383, 77.3210),
    "Nandurbar": (21.3712, 74.2415),
    "Nashik": (19.9975, 73.7898),
    "Osmanabad": (18.1861, 76.0419),
    "Dharashiv": (18.1861, 76.0419),
    "Palghar": (19.6966, 72.7699),
    "Parbhani": (19.2686, 76.7709),
    "Pune": (18.5204, 73.8567),
    "Raigad": (18.5158, 73.1822),
    "Ratnagiri": (16.9902, 73.3120),
    "Sangli": (16.8524, 74.5815),
    "Satara": (17.6805, 74.0183),
    "Sindhudurg": (16.1667, 73.6667),
    "Solapur": (17.6599, 75.9064),
    "Thane": (19.2183, 72.9781),
    "Wardha": (20.7453, 78.6022),
    "Washim": (20.1017, 77.1337),
    "Yavatmal": (20.3888, 78.1204)
}

# WMO Weather Interpretation Codes (Open-Meteo standard)
WMO_WEATHER_CODES: Dict[int, Tuple[str, str]] = {
    0: ("☀️ Clear Sky", "Sunny and clear weather conditions."),
    1: ("🌤️ Mainly Clear", "Mainly clear, pleasant conditions."),
    2: ("⛅ Partly Cloudy", "Partly cloudy skies with sunshine."),
    3: ("☁️ Overcast", "Overcast skies throughout the area."),
    45: ("🌫️ Foggy", "Foggy conditions present."),
    48: ("🌫️ Depositing Rime Fog", "Dense fog with moisture accumulation."),
    51: ("🌧️ Light Drizzle", "Light drizzle observed."),
    53: ("🌧️ Moderate Drizzle", "Moderate drizzle conditions."),
    55: ("🌧️ Heavy Drizzle", "Heavy drizzle accumulation."),
    61: ("🌧️ Light Rain", "Light rain showers."),
    63: ("🌧️ Moderate Rain", "Steady moderate rainfall."),
    65: ("🌧️ Heavy Rain", "Heavy rainfall expected."),
    80: ("🌦️ Light Rain Showers", "Passing light rain showers."),
    81: ("🌦️ Moderate Rain Showers", "Moderate rain showers."),
    82: ("⛈️ Violent Rain Showers", "Heavy localized rain showers."),
    95: ("🌩️ Thunderstorm", "Thunderstorm activity detected."),
    96: ("⛈️ Thunderstorm & Hail", "Thunderstorm with slight hail risks."),
    99: ("⛈️ Heavy Hailstorm", "Severe thunderstorm with heavy hail.")
}

class WeatherService:
    """Service to fetch real-time and forecast weather data from Open-Meteo API for Maharashtra districts."""

    def __init__(self, timeout: int = 6):
        self.timeout = timeout
        self.forecast_api_url = "https://api.open-meteo.com/v1/forecast"
        self.geocoding_api_url = "https://geocoding-api.open-meteo.com/v1/search"

    def get_district_coordinates(self, district_name: str) -> Tuple[float, float]:
        """Lookup coordinates from local Maharashtra mapping or query Open-Meteo Geocoding API."""
        clean_name = district_name.strip()
        
        # Direct lookup
        for key, coords in MAHARASHTRA_DISTRICT_COORDS.items():
            if key.lower() in clean_name.lower() or clean_name.lower() in key.lower():
                return coords

        # Dynamic Geocoding API Fallback
        try:
            params = {
                "name": f"{clean_name}, Maharashtra",
                "count": 1,
                "language": "en",
                "format": "json"
            }
            resp = requests.get(self.geocoding_api_url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if "results" in data and len(data["results"]) > 0:
                    lat = data["results"][0]["latitude"]
                    lon = data["results"][0]["longitude"]
                    return (lat, lon)
        except Exception as e:
            logger.warning(f"Geocoding lookup failed for {district_name}: {e}")

        # Default fallback to Pune coordinates
        return (18.5204, 73.8567)

    def fetch_live_weather(self, district_name: str) -> Dict[str, Any]:
        """Fetch current weather, soil parameters, and 7-day forecast for a Maharashtra district."""
        lat, lon = self.get_district_coordinates(district_name)

        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "weather_code",
                "surface_pressure",
                "wind_speed_10m",
                "soil_temperature_0_to_7cm",
                "soil_moisture_0_to_1cm"
            ],
            "daily": [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "et0_fao_evapotranspiration",
                "wind_speed_10m_max"
            ],
            "timezone": "Asia/Kolkata",
            "forecast_days": 7
        }

        try:
            response = requests.get(self.forecast_api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            current = data.get("current", {})
            daily = data.get("daily", {})

            w_code = current.get("weather_code", 0)
            condition_title, condition_desc = WMO_WEATHER_CODES.get(
                w_code, ("🌤️ Fair Weather", "Normal weather conditions.")
            )

            # Summarize 7-day precipitation & FAO Evapotranspiration
            precip_7d = sum(daily.get("precipitation_sum", [0]))
            et0_7d = sum(daily.get("et0_fao_evapotranspiration", [0]))

            # Formatting 7-day forecast table
            forecast_list = []
            dates = daily.get("time", [])
            t_max = daily.get("temperature_2m_max", [])
            t_min = daily.get("temperature_2m_min", [])
            p_sum = daily.get("precipitation_sum", [])
            w_codes = daily.get("weather_code", [])

            for i in range(len(dates)):
                code_i = w_codes[i] if i < len(w_codes) else 0
                c_title, _ = WMO_WEATHER_CODES.get(code_i, ("🌤️ Fair", ""))
                forecast_list.append({
                    "Date": dates[i],
                    "Condition": c_title,
                    "Max Temp (°C)": t_max[i] if i < len(t_max) else "N/A",
                    "Min Temp (°C)": t_min[i] if i < len(t_min) else "N/A",
                    "Precipitation (mm)": p_sum[i] if i < len(p_sum) else 0.0
                })

            # Agricultural Alerts
            alerts = []
            curr_humidity = current.get("relative_humidity_2m", 50)
            curr_temp = current.get("temperature_2m", 25)
            soil_moist = current.get("soil_moisture_0_to_1cm", 0.2)

            if curr_humidity > 80:
                alerts.append("⚠️ High Humidity (>80%): Elevated risk of fungal infections in Grape, Tomato & Cotton crops. Consider protective spray.")
            if precip_7d > 50:
                alerts.append("🌧️ Heavy Rainfall Advisory: Expected total precipitation >50mm in next 7 days. Ensure field drainage.")
            elif precip_7d < 2.0 and et0_7d > 25.0:
                alerts.append("☀️ Dry Spell & High Evapotranspiration: Low rainfall expected. Schedule irrigation for young crops.")
            if curr_temp > 38.0:
                alerts.append("🔥 High Temperature Warning (>38°C): Heat stress risk for summer crops & fruit orchards.")

            return {
                "success": True,
                "district": district_name,
                "latitude": lat,
                "longitude": lon,
                "current_temp": current.get("temperature_2m", 25.0),
                "feels_like": current.get("apparent_temperature", 25.0),
                "humidity": curr_humidity,
                "wind_speed": current.get("wind_speed_10m", 0.0),
                "current_precip": current.get("precipitation", 0.0),
                "soil_temp_0_7cm": current.get("soil_temperature_0_to_7cm", 25.0),
                "soil_moisture_0_1cm": soil_moist,
                "weather_code": w_code,
                "condition_title": condition_title,
                "condition_desc": condition_desc,
                "precip_7d_total": round(precip_7d, 1),
                "et0_7d_total": round(et0_7d, 1),
                "forecast_7d": forecast_list,
                "agri_alerts": alerts
            }

        except Exception as e:
            logger.error(f"Error fetching Open-Meteo weather data: {e}")
            return {
                "success": False,
                "district": district_name,
                "error": str(e)
            }

    def fetch_annual_rainfall(self, district_name: str) -> Dict[str, Any]:
        """
        Estimate annual rainfall for a district by fetching the last 92 days of historical
        precipitation via the Open-Meteo forecast API (past_days parameter, no lag issue),
        then scaling up: monthly_avg = total / 3 months → annual = monthly_avg × 12.
        """
        lat, lon = self.get_district_coordinates(district_name)

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "timezone": "Asia/Kolkata",
            "past_days": 92,       # ~3 months of reliable historical data
            "forecast_days": 1     # minimal future, we only need past
        }

        try:
            response = requests.get(
                self.forecast_api_url,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            daily_precip = data.get("daily", {}).get("precipitation_sum", [])
            dates = data.get("daily", {}).get("time", [])

            # Use only past days (exclude forecast day)
            import datetime
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            past_precip = [
                v for d, v in zip(dates, daily_precip)
                if d < today_str and v is not None
            ]

            days_used = len(past_precip)
            if days_used < 30:
                return {
                    "success": False,
                    "error": f"Insufficient data: only {days_used} days retrieved."
                }

            total_past = sum(past_precip)
            # Scale to 365 days: proportional extrapolation
            annual_estimate = round((total_past / days_used) * 365, 1)

            # Sanity clamp: Maharashtra annual rainfall range is 300–2500 mm
            annual_estimate = max(300.0, min(2500.0, annual_estimate))

            months_used = round(days_used / 30.4, 1)
            return {
                "success": True,
                "annual_rainfall_mm": annual_estimate,
                "days_used": days_used,
                "months_used": months_used,
                "total_past_mm": round(total_past, 1),
                "method": f"Daily avg ({round(total_past/days_used, 2)} mm/day) × 365 days"
            }
        except Exception as e:
            logger.error(f"Failed to estimate annual rainfall for {district_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }


_weather_service_instance = None

def get_weather_service() -> WeatherService:
    global _weather_service_instance
    if _weather_service_instance is None:
        _weather_service_instance = WeatherService()
    return _weather_service_instance
