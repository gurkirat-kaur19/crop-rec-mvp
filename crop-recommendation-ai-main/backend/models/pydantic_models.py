from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# User models
class UserBase(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    location: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    created_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True

# Soil Data models
class SoilDataBase(BaseModel):
    ph_level: float
    nitrogen: float
    phosphorus: float
    potassium: float
    organic_matter: Optional[float] = None
    moisture_content: Optional[float] = None
    soil_type: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_name: Optional[str] = None

class SoilDataCreate(SoilDataBase):
    user_id: Optional[int] = None

class SoilData(SoilDataBase):
    id: int
    user_id: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True

# Weather Data models
class WeatherDataBase(BaseModel):
    location: str
    temperature: float
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    wind_speed: Optional[float] = None
    weather_condition: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class WeatherData(WeatherDataBase):
    id: int
    date: datetime
    
    class Config:
        from_attributes = True

# Crop models
class CropBase(BaseModel):
    name: str
    name_hindi: Optional[str] = None
    scientific_name: Optional[str] = None
    crop_type: str
    season: str
    min_temp: Optional[float] = None
    max_temp: Optional[float] = None
    min_rainfall: Optional[float] = None
    max_rainfall: Optional[float] = None
    min_ph: Optional[float] = None
    max_ph: Optional[float] = None
    nitrogen_requirement: Optional[str] = None
    phosphorus_requirement: Optional[str] = None
    potassium_requirement: Optional[str] = None
    avg_yield_per_hectare: Optional[float] = None
    avg_price_per_kg: Optional[float] = None
    cultivation_cost_per_hectare: Optional[float] = None
    growth_duration_days: Optional[int] = None
    description: Optional[str] = None

class CropCreate(CropBase):
    pass

class Crop(CropBase):
    id: int
    
    class Config:
        from_attributes = True

# Recommendation models
class RecommendationInput(BaseModel):
    # Soil parameters
    soil_ph: float
    nitrogen: float
    phosphorus: float
    potassium: float
    
    # Weather parameters
    temperature: float
    humidity: float
    rainfall: float
    
    # Location
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # User info (optional for anonymous recommendations)
    user_id: Optional[int] = None

class RecommendationResult(BaseModel):
    crop_id: int
    crop_name: str
    crop_name_hindi: Optional[str] = None
    confidence_score: float
    predicted_yield: Optional[float] = None
    estimated_profit: Optional[float] = None
    sustainability_score: Optional[float] = None
    season: Optional[str] = None
    remarks: Optional[str] = None

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationResult]
    input_parameters: RecommendationInput
    timestamp: datetime
    
class Recommendation(BaseModel):
    id: int
    user_id: Optional[int]
    crop_id: int
    soil_ph: float
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    rainfall: float
    location: Optional[str]
    confidence_score: float
    predicted_yield: Optional[float]
    estimated_profit: Optional[float]
    sustainability_score: Optional[float]
    season: Optional[str]
    remarks: Optional[str]
    created_at: datetime
    
    # Related objects
    crop: Optional[Crop] = None
    user: Optional[User] = None
    
    class Config:
        from_attributes = True
