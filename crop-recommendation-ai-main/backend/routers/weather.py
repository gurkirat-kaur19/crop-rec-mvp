from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from services.weather_service import WeatherService
from models.pydantic_models import WeatherData
from database import get_db
import os

router = APIRouter()
weather_service = WeatherService(api_key=os.getenv("OPENWEATHER_API_KEY", "demo_key"))

@router.get("/current/{location}", response_model=WeatherData)
async def get_current_weather(location: str):
    """Get current weather data for a location"""
    try:
        weather_data = await weather_service.get_current_weather(location)
        if not weather_data:
            raise HTTPException(status_code=404, detail="Weather data not found for this location")
        return weather_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")

@router.get("/coordinates/{lat}/{lon}", response_model=WeatherData)
async def get_weather_by_coordinates(lat: float, lon: float):
    """Get current weather data by coordinates"""
    try:
        weather_data = await weather_service.get_weather_by_coordinates(lat, lon)
        if not weather_data:
            raise HTTPException(status_code=404, detail="Weather data not found for these coordinates")
        return weather_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")

@router.get("/demo/{location}")
async def get_demo_weather(location: str):
    """Get demo weather data for testing purposes"""
    # Return mock weather data for demo
    demo_data = {
        "id": 1,
        "location": location,
        "latitude": 23.3441,
        "longitude": 85.3096,
        "temperature": 28.5,
        "humidity": 65.0,
        "rainfall": 2.5,
        "wind_speed": 8.0,
        "weather_condition": "partly cloudy",
        "date": "2024-01-15T10:00:00"
    }
    return demo_data
