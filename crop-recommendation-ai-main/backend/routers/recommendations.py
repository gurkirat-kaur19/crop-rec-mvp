from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from models.pydantic_models import (
    RecommendationInput, 
    RecommendationResponse, 
    Recommendation,
    SoilDataCreate,
    SoilData
)
from models.database_models import (
    Recommendation as DBRecommendation,
    SoilData as DBSoilData,
    Crop as DBCrop
)
from services.recommendation_service import RecommendationService
from database import get_db

router = APIRouter()
recommendation_service = RecommendationService()

@router.post("/predict", response_model=RecommendationResponse)
async def get_crop_recommendations(
    input_data: RecommendationInput, 
    db: Session = Depends(get_db)
):
    """Get crop recommendations based on soil and weather data"""
    try:
        # Get recommendations from ML service
        recommendations = await recommendation_service.get_recommendations(input_data, db)
        
        return RecommendationResponse(
            recommendations=recommendations,
            input_parameters=input_data,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.post("/soil-data", response_model=SoilData)
def save_soil_data(soil_data: SoilDataCreate, db: Session = Depends(get_db)):
    """Save soil data for a user"""
    db_soil_data = DBSoilData(**soil_data.dict())
    db.add(db_soil_data)
    db.commit()
    db.refresh(db_soil_data)
    return db_soil_data

@router.get("/history/{user_id}", response_model=List[Recommendation])
def get_user_recommendations(user_id: int, skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    """Get recommendation history for a user"""
    recommendations = (
        db.query(DBRecommendation)
        .filter(DBRecommendation.user_id == user_id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return recommendations

@router.get("/soil-data/{user_id}", response_model=List[SoilData])
def get_user_soil_data(user_id: int, db: Session = Depends(get_db)):
    """Get soil data history for a user"""
    soil_data = (
        db.query(DBSoilData)
        .filter(DBSoilData.user_id == user_id)
        .order_by(DBSoilData.created_at.desc())
        .all()
    )
    return soil_data
