from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
from typing import List, Dict, Any
import os

# Load models and data
import sys
from pathlib import Path

# Get the parent directory
parent_dir = Path(__file__).parent.parent

model = joblib.load(parent_dir / 'models' / 'crop_model.pkl')
label_encoder = joblib.load(parent_dir / 'models' / 'label_encoder.pkl')

with open(parent_dir / 'models' / 'crop_info.json', 'r') as f:
    crop_info = json.load(f)

# Create FastAPI app
app = FastAPI(title="Crop Recommendation API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class SoilData(BaseModel):
    N: float = Field(..., ge=0, le=150, description="Nitrogen content in soil")
    P: float = Field(..., ge=0, le=200, description="Phosphorous content in soil")
    K: float = Field(..., ge=0, le=210, description="Potassium content in soil")
    temperature: float = Field(..., ge=10, le=50, description="Temperature in Celsius")
    humidity: float = Field(..., ge=30, le=100, description="Humidity percentage")
    ph: float = Field(..., ge=3, le=10, description="pH value of soil")
    rainfall: float = Field(..., ge=0, le=400, description="Rainfall in mm")

class CropRecommendation(BaseModel):
    recommended_crop: str
    confidence: float
    top_3_crops: List[Dict[str, Any]]
    soil_analysis: Dict[str, str]
    crop_details: Dict[str, Any]

# Crop information database
CROP_DETAILS = {
    "rice": {
        "hindi_name": "धान",
        "season": "Kharif (June-October)",
        "water_requirement": "High (1200-1500mm)",
        "harvest_time": "120-150 days",
        "market_price": "₹1800-2200/quintal",
        "tips": "Requires flooded fields, warm climate"
    },
    "wheat": {
        "hindi_name": "गेहूं",
        "season": "Rabi (October-March)",
        "water_requirement": "Medium (450-650mm)",
        "harvest_time": "120-150 days",
        "market_price": "₹2000-2500/quintal",
        "tips": "Cool season crop, needs good drainage"
    },
    "maize": {
        "hindi_name": "मक्का",
        "season": "Kharif/Rabi",
        "water_requirement": "Medium (500-800mm)",
        "harvest_time": "90-120 days",
        "market_price": "₹1500-2000/quintal",
        "tips": "Versatile crop, good for crop rotation"
    },
    "cotton": {
        "hindi_name": "कपास",
        "season": "Kharif (April-October)",
        "water_requirement": "High (700-1300mm)",
        "harvest_time": "180-200 days",
        "market_price": "₹5500-6500/quintal",
        "tips": "Cash crop, requires warm climate"
    },
    "jute": {
        "hindi_name": "जूट",
        "season": "Kharif (March-July)",
        "water_requirement": "High (1200-1500mm)",
        "harvest_time": "120-150 days",
        "market_price": "₹4000-5000/quintal",
        "tips": "Fiber crop, grows well in humid areas"
    },
    "sugarcane": {
        "hindi_name": "गन्ना",
        "season": "All seasons",
        "water_requirement": "Very High (1500-2500mm)",
        "harvest_time": "12-18 months",
        "market_price": "₹300-350/quintal",
        "tips": "Long duration crop, high water needs"
    },
    "groundnut": {
        "hindi_name": "मूंगफली",
        "season": "Kharif/Rabi",
        "water_requirement": "Medium (500-700mm)",
        "harvest_time": "100-120 days",
        "market_price": "₹5000-6000/quintal",
        "tips": "Oil seed crop, improves soil nitrogen"
    },
    "banana": {
        "hindi_name": "केला",
        "season": "All seasons",
        "water_requirement": "High (1200-2200mm)",
        "harvest_time": "12-15 months",
        "market_price": "₹1000-1500/quintal",
        "tips": "Perennial crop, regular water needed"
    },
    "tomato": {
        "hindi_name": "टमाटर",
        "season": "All seasons",
        "water_requirement": "Medium (600-800mm)",
        "harvest_time": "60-90 days",
        "market_price": "₹1500-3000/quintal",
        "tips": "High value vegetable crop"
    },
    "chickpea": {
        "hindi_name": "चना",
        "season": "Rabi (October-March)",
        "water_requirement": "Low (350-500mm)",
        "harvest_time": "90-120 days",
        "market_price": "₹4500-5500/quintal",
        "tips": "Pulse crop, drought tolerant"
    }
}

def analyze_soil(data: SoilData) -> Dict[str, str]:
    """Analyze soil parameters and provide recommendations"""
    analysis = {}
    
    # N analysis
    if data.N < 50:
        analysis["nitrogen"] = "Low - Consider adding urea or organic manure"
    elif data.N > 100:
        analysis["nitrogen"] = "High - Good for leafy crops"
    else:
        analysis["nitrogen"] = "Optimal - Suitable for most crops"
    
    # P analysis
    if data.P < 25:
        analysis["phosphorus"] = "Low - Add phosphate fertilizers"
    elif data.P > 75:
        analysis["phosphorus"] = "High - Good for root development"
    else:
        analysis["phosphorus"] = "Optimal - Balanced for growth"
    
    # K analysis
    if data.K < 30:
        analysis["potassium"] = "Low - Add potash fertilizers"
    elif data.K > 70:
        analysis["potassium"] = "High - Good for fruit crops"
    else:
        analysis["potassium"] = "Optimal - Good for overall health"
    
    # pH analysis
    if data.ph < 5.5:
        analysis["ph"] = "Acidic - Add lime to increase pH"
    elif data.ph > 7.5:
        analysis["ph"] = "Alkaline - Add sulfur or organic matter"
    else:
        analysis["ph"] = "Neutral - Ideal for most crops"
    
    # Rainfall analysis
    if data.rainfall < 50:
        analysis["water"] = "Low rainfall - Irrigation required"
    elif data.rainfall > 200:
        analysis["water"] = "High rainfall - Ensure good drainage"
    else:
        analysis["water"] = "Moderate rainfall - Supplemental irrigation may help"
    
    return analysis

@app.get("/")
async def root():
    return {
        "message": "Crop Recommendation API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/crops", "/features", "/health"]
    }

@app.post("/predict", response_model=CropRecommendation)
async def predict_crop(soil_data: SoilData):
    """Predict the best crop based on soil and weather conditions"""
    try:
        # Prepare input features
        features = np.array([[
            soil_data.N, 
            soil_data.P, 
            soil_data.K,
            soil_data.temperature, 
            soil_data.humidity,
            soil_data.ph, 
            soil_data.rainfall
        ]])
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features)[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = []
        
        for idx in top_3_indices:
            crop_name = label_encoder.inverse_transform([idx])[0]
            confidence = float(probabilities[idx])
            top_3_crops.append({
                "crop": crop_name,
                "confidence": round(confidence * 100, 2),
                "hindi_name": CROP_DETAILS.get(crop_name, {}).get("hindi_name", "")
            })
        
        # Get the recommended crop
        recommended = top_3_crops[0]["crop"]
        
        # Analyze soil
        soil_analysis = analyze_soil(soil_data)
        
        # Get crop details
        crop_details = CROP_DETAILS.get(recommended, {
            "hindi_name": "N/A",
            "season": "N/A",
            "water_requirement": "N/A",
            "harvest_time": "N/A",
            "market_price": "N/A",
            "tips": "N/A"
        })
        
        return CropRecommendation(
            recommended_crop=recommended,
            confidence=top_3_crops[0]["confidence"],
            top_3_crops=top_3_crops,
            soil_analysis=soil_analysis,
            crop_details=crop_details
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crops")
async def get_crops():
    """Get list of all available crops"""
    return {
        "crops": crop_info["crops"],
        "total": len(crop_info["crops"])
    }

@app.get("/features")
async def get_features():
    """Get feature ranges and information"""
    return {
        "features": crop_info["features"],
        "ranges": crop_info["feature_ranges"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "loaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=800)
