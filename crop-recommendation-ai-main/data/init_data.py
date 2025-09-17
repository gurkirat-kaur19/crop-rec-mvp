import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from sqlalchemy.orm import sessionmaker
from database import engine, create_tables
from models.database_models import Crop

def create_sample_crops():
    """Create sample crop data"""
    # Ensure tables exist
    create_tables()

    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Check if crops already exist
    if session.query(Crop).first():
        print("Crops already exist in database")
        session.close()
        return
    
    sample_crops = [
        {
            'name': 'rice',
            'name_hindi': 'चावल',
            'scientific_name': 'Oryza sativa',
            'crop_type': 'cereal',
            'season': 'kharif',
            'min_temp': 20.0,
            'max_temp': 30.0,
            'min_rainfall': 150.0,
            'max_rainfall': 300.0,
            'min_ph': 5.5,
            'max_ph': 7.0,
            'nitrogen_requirement': 'high',
            'phosphorus_requirement': 'medium',
            'potassium_requirement': 'medium',
            'avg_yield_per_hectare': 4.5,
            'avg_price_per_kg': 25.0,
            'cultivation_cost_per_hectare': 40000.0,
            'growth_duration_days': 120,
            'description': 'Rice is the staple food crop, suitable for high rainfall areas with good water availability.'
        },
        {
            'name': 'wheat',
            'name_hindi': 'गेहूं',
            'scientific_name': 'Triticum aestivum',
            'crop_type': 'cereal',
            'season': 'rabi',
            'min_temp': 15.0,
            'max_temp': 25.0,
            'min_rainfall': 50.0,
            'max_rainfall': 100.0,
            'min_ph': 6.0,
            'max_ph': 7.5,
            'nitrogen_requirement': 'high',
            'phosphorus_requirement': 'medium',
            'potassium_requirement': 'medium',
            'avg_yield_per_hectare': 4.2,
            'avg_price_per_kg': 22.0,
            'cultivation_cost_per_hectare': 35000.0,
            'growth_duration_days': 150,
            'description': 'Wheat is a major rabi crop, suitable for cooler temperatures and moderate rainfall.'
        },
        {
            'name': 'maize',
            'name_hindi': 'मक्का',
            'scientific_name': 'Zea mays',
            'crop_type': 'cereal',
            'season': 'kharif',
            'min_temp': 18.0,
            'max_temp': 27.0,
            'min_rainfall': 60.0,
            'max_rainfall': 110.0,
            'min_ph': 5.8,
            'max_ph': 7.0,
            'nitrogen_requirement': 'high',
            'phosphorus_requirement': 'medium',
            'potassium_requirement': 'high',
            'avg_yield_per_hectare': 6.5,
            'avg_price_per_kg': 20.0,
            'cultivation_cost_per_hectare': 35000.0,
            'growth_duration_days': 90,
            'description': 'Maize is versatile crop suitable for moderate rainfall and warm temperatures.'
        },
        {
            'name': 'cotton',
            'name_hindi': 'कपास',
            'scientific_name': 'Gossypium hirsutum',
            'crop_type': 'cash crop',
            'season': 'kharif',
            'min_temp': 21.0,
            'max_temp': 30.0,
            'min_rainfall': 50.0,
            'max_rainfall': 100.0,
            'min_ph': 5.8,
            'max_ph': 8.0,
            'nitrogen_requirement': 'high',
            'phosphorus_requirement': 'high',
            'potassium_requirement': 'high',
            'avg_yield_per_hectare': 2.8,
            'avg_price_per_kg': 55.0,
            'cultivation_cost_per_hectare': 45000.0,
            'growth_duration_days': 180,
            'description': 'Cotton is a major cash crop requiring warm climate and moderate water.'
        },
        {
            'name': 'sugarcane',
            'name_hindi': 'गन्ना',
            'scientific_name': 'Saccharum officinarum',
            'crop_type': 'cash crop',
            'season': 'kharif',
            'min_temp': 21.0,
            'max_temp': 27.0,
            'min_rainfall': 75.0,
            'max_rainfall': 150.0,
            'min_ph': 6.0,
            'max_ph': 7.5,
            'nitrogen_requirement': 'high',
            'phosphorus_requirement': 'medium',
            'potassium_requirement': 'high',
            'avg_yield_per_hectare': 75.0,
            'avg_price_per_kg': 3.5,
            'cultivation_cost_per_hectare': 80000.0,
            'growth_duration_days': 365,
            'description': 'Sugarcane requires consistent water supply and warm climate for optimal growth.'
        }
    ]
    
    for crop_data in sample_crops:
        crop = Crop(**crop_data)
        session.add(crop)
    
    session.commit()
    session.close()
    print(f"Created {len(sample_crops)} sample crops in database")

if __name__ == "__main__":
    create_sample_crops()
