import httpx
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from models.pydantic_models import WeatherData

class WeatherService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    async def get_current_weather(self, location: str) -> Optional[WeatherData]:
        """Get current weather data for a location"""
        if self.api_key == "demo_key":
            return self._get_mock_weather_data(location)
            
        try:
            url = f"{self.base_url}/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                return self._parse_weather_response(data)
                
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            # Fallback to mock data
            return self._get_mock_weather_data(location)
    
    async def get_weather_by_coordinates(self, lat: float, lon: float) -> Optional[WeatherData]:
        """Get current weather data by coordinates"""
        if self.api_key == "demo_key":
            return self._get_mock_weather_data(f"{lat},{lon}")
            
        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                return self._parse_weather_response(data)
                
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            # Fallback to mock data
            return self._get_mock_weather_data(f"{lat},{lon}")
    
    def _parse_weather_response(self, data: Dict[str, Any]) -> WeatherData:
        """Parse OpenWeatherMap API response"""
        return WeatherData(
            id=1,  # Mock ID for database
            location=data.get("name", "Unknown"),
            latitude=data.get("coord", {}).get("lat"),
            longitude=data.get("coord", {}).get("lon"),
            temperature=data.get("main", {}).get("temp", 0),
            humidity=data.get("main", {}).get("humidity"),
            rainfall=data.get("rain", {}).get("1h", 0),  # Rain in last hour
            wind_speed=data.get("wind", {}).get("speed"),
            weather_condition=data.get("weather", [{}])[0].get("description", ""),
            date=datetime.now()
        )
    
    def _get_mock_weather_data(self, location: str) -> WeatherData:
        """Return mock weather data for demo purposes"""
        # Different mock data based on location
        mock_data = {
            "ranchi": {
                "temperature": 28.5,
                "humidity": 65.0,
                "rainfall": 2.5,
                "wind_speed": 8.0,
                "weather_condition": "partly cloudy",
                "latitude": 23.3441,
                "longitude": 85.3096
            },
            "delhi": {
                "temperature": 32.0,
                "humidity": 45.0,
                "rainfall": 0.0,
                "wind_speed": 12.0,
                "weather_condition": "clear sky",
                "latitude": 28.6139,
                "longitude": 77.2090
            },
            "mumbai": {
                "temperature": 30.0,
                "humidity": 80.0,
                "rainfall": 5.2,
                "wind_speed": 15.0,
                "weather_condition": "light rain",
                "latitude": 19.0760,
                "longitude": 72.8777
            }
        }
        
        # Default to Ranchi data if location not found
        location_key = location.lower().split(",")[0]  # Handle coordinates
        weather_info = mock_data.get(location_key, mock_data["ranchi"])
        
        return WeatherData(
            id=1,
            location=location.title(),
            **weather_info,
            date=datetime.now()
        )
