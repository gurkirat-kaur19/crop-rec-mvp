from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from models.pydantic_models import User, UserCreate
from models.database_models import User as DBUser
from database import get_db

router = APIRouter()

@router.post("/register", response_model=User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(DBUser).filter(DBUser.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create new user
    db_user = DBUser(
        name=user.name,
        email=user.email,
        phone=user.phone,
        location=user.location,
        state=user.state,
        district=user.district
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@router.get("/users/{user_id}", response_model=User)
def get_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(DBUser).filter(DBUser.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.get("/users/email/{email}", response_model=User)
def get_user_by_email(email: str, db: Session = Depends(get_db)):
    db_user = db.query(DBUser).filter(DBUser.email == email).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user
