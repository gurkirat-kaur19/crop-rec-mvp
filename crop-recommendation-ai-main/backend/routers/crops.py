from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from models.pydantic_models import Crop, CropCreate
from models.database_models import Crop as DBCrop
from database import get_db

router = APIRouter()

@router.get("/", response_model=List[Crop])
def get_all_crops(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all crops"""
    crops = db.query(DBCrop).offset(skip).limit(limit).all()
    return crops

@router.get("/{crop_id}", response_model=Crop)
def get_crop(crop_id: int, db: Session = Depends(get_db)):
    """Get a specific crop by ID"""
    crop = db.query(DBCrop).filter(DBCrop.id == crop_id).first()
    if crop is None:
        raise HTTPException(status_code=404, detail="Crop not found")
    return crop

@router.get("/season/{season}", response_model=List[Crop])
def get_crops_by_season(season: str, db: Session = Depends(get_db)):
    """Get crops by season (kharif, rabi, zaid)"""
    crops = db.query(DBCrop).filter(DBCrop.season.ilike(f"%{season}%")).all()
    return crops

@router.get("/type/{crop_type}", response_model=List[Crop])
def get_crops_by_type(crop_type: str, db: Session = Depends(get_db)):
    """Get crops by type (cereal, vegetable, fruit, etc.)"""
    crops = db.query(DBCrop).filter(DBCrop.crop_type.ilike(f"%{crop_type}%")).all()
    return crops

@router.post("/", response_model=Crop)
def create_crop(crop: CropCreate, db: Session = Depends(get_db)):
    """Create a new crop (for admin use)"""
    db_crop = DBCrop(**crop.dict())
    db.add(db_crop)
    db.commit()
    db.refresh(db_crop)
    return db_crop
