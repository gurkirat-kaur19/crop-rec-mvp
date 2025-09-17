from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    phone = Column(String(20), nullable=True)
    location = Column(String(100), nullable=True)
    state = Column(String(50), nullable=True)
    district = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    recommendations = relationship("Recommendation", back_populates="user")

class Crop(Base):
    __tablename__ = "crops"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    name_hindi = Column(String(100), nullable=True)
    scientific_name = Column(String(100), nullable=True)
    crop_type = Column(String(50), nullable=False)  # cereal, vegetable, fruit, etc.
    season = Column(String(50), nullable=False)  # kharif, rabi, zaid
    
    # Growing conditions
    min_temp = Column(Float, nullable=True)
    max_temp = Column(Float, nullable=True)
    min_rainfall = Column(Float, nullable=True)
    max_rainfall = Column(Float, nullable=True)
    min_ph = Column(Float, nullable=True)
    max_ph = Column(Float, nullable=True)
    
    # Soil requirements
    nitrogen_requirement = Column(String(20), nullable=True)  # low, medium, high
    phosphorus_requirement = Column(String(20), nullable=True)
    potassium_requirement = Column(String(20), nullable=True)
    
    # Economic data
    avg_yield_per_hectare = Column(Float, nullable=True)
    avg_price_per_kg = Column(Float, nullable=True)
    cultivation_cost_per_hectare = Column(Float, nullable=True)
    
    # Additional info
    growth_duration_days = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    
    # Relationships
    recommendations = relationship("Recommendation", back_populates="crop")

class SoilData(Base):
    __tablename__ = "soil_data"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Location
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location_name = Column(String(100), nullable=True)
    
    # Soil parameters
    ph_level = Column(Float, nullable=False)
    nitrogen = Column(Float, nullable=False)  # N content
    phosphorus = Column(Float, nullable=False)  # P content
    potassium = Column(Float, nullable=False)  # K content
    organic_matter = Column(Float, nullable=True)
    moisture_content = Column(Float, nullable=True)
    soil_type = Column(String(50), nullable=True)  # clay, sandy, loamy, etc.
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")

class WeatherData(Base):
    __tablename__ = "weather_data"
    
    id = Column(Integer, primary_key=True, index=True)
    location = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Current weather
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=True)
    rainfall = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    weather_condition = Column(String(50), nullable=True)
    
    # Date
    date = Column(DateTime(timezone=True), server_default=func.now())
    
class Recommendation(Base):
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    crop_id = Column(Integer, ForeignKey("crops.id"))
    
    # Input parameters used for recommendation
    soil_ph = Column(Float, nullable=False)
    nitrogen = Column(Float, nullable=False)
    phosphorus = Column(Float, nullable=False)
    potassium = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False)
    location = Column(String(100), nullable=True)
    
    # Prediction results
    confidence_score = Column(Float, nullable=False)  # 0-1
    predicted_yield = Column(Float, nullable=True)
    estimated_profit = Column(Float, nullable=True)
    sustainability_score = Column(Float, nullable=True)
    
    # Additional info
    season = Column(String(50), nullable=True)
    remarks = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="recommendations")
    crop = relationship("Crop", back_populates="recommendations")
