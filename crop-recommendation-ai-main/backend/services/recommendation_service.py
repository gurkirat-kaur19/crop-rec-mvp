import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_models'))

from crop_model import CropRecommendationModel
from models.pydantic_models import RecommendationInput, RecommendationResult
from models.database_models import Crop as DBCrop, Recommendation as DBRecommendation
from sqlalchemy.orm import Session
from typing import List
import random

class RecommendationService:
    def __init__(self):
        self.model = CropRecommendationModel()
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ml_models', 'crop_model.pkl')
        
        # Try to load existing model or train new one
        try:
            self.model.load_model(model_path)
        except Exception as e:
            print(f"Loading model failed: {e}. Training new model...")
            self.model.train()
            # Save the trained model
            try:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.model.save_model(model_path)
            except Exception as save_error:
                print(f"Could not save model: {save_error}")
    
    async def get_recommendations(self, input_data: RecommendationInput, db: Session) -> List[RecommendationResult]:
        """Generate crop recommendations based on input parameters"""
        
        # Prepare features for ML model
        features = {
            'nitrogen': input_data.nitrogen,
            'phosphorus': input_data.phosphorus,
            'potassium': input_data.potassium,
            'temperature': input_data.temperature,
            'humidity': input_data.humidity,
            'ph': input_data.soil_ph,
            'rainfall': input_data.rainfall
        }
        
        # Get predictions from ML model
        predictions = self.model.predict(features)
        
        # Convert predictions to recommendation results
        recommendations = []
        
        for i, (crop_name, confidence) in enumerate(predictions):
            # Try to get crop from database
            db_crop = db.query(DBCrop).filter(DBCrop.name.ilike(f"%{crop_name}%")).first()
            
            if not db_crop:
                # Create a basic crop record if not found
                db_crop = self._create_basic_crop_info(crop_name, db)
            
            # Calculate additional metrics
            predicted_yield = self._calculate_predicted_yield(db_crop, features, confidence)
            estimated_profit = self._calculate_estimated_profit(db_crop, predicted_yield)
            sustainability_score = self._calculate_sustainability_score(features, db_crop)
            season = self._determine_season(features, db_crop)
            
            # Generate remarks
            remarks = self._generate_remarks(crop_name, confidence, features)
            
            recommendation = RecommendationResult(
                crop_id=db_crop.id,
                crop_name=db_crop.name,
                crop_name_hindi=db_crop.name_hindi,
                confidence_score=round(confidence, 3),
                predicted_yield=predicted_yield,
                estimated_profit=estimated_profit,
                sustainability_score=sustainability_score,
                season=season,
                remarks=remarks
            )
            
            recommendations.append(recommendation)
            
            # Save recommendation to database if user_id provided
            if input_data.user_id:
                self._save_recommendation_to_db(input_data, recommendation, db)
        
        return recommendations
    
    def _create_basic_crop_info(self, crop_name: str, db: Session) -> DBCrop:
        """Create basic crop information if not found in database"""
        crop_info = {
            'apple': {'name_hindi': 'सेब', 'type': 'fruit', 'season': 'rabi', 'yield': 15.0, 'price': 80.0, 'cost': 120000},
            'banana': {'name_hindi': 'केला', 'type': 'fruit', 'season': 'kharif', 'yield': 25.0, 'price': 30.0, 'cost': 80000},
            'blackgram': {'name_hindi': 'उड़द', 'type': 'pulse', 'season': 'kharif', 'yield': 1.2, 'price': 75.0, 'cost': 25000},
            'chickpea': {'name_hindi': 'चना', 'type': 'pulse', 'season': 'rabi', 'yield': 2.5, 'price': 55.0, 'cost': 30000},
            'coconut': {'name_hindi': 'नारियल', 'type': 'fruit', 'season': 'kharif', 'yield': 8.0, 'price': 35.0, 'cost': 60000},
            'coffee': {'name_hindi': 'कॉफी', 'type': 'cash crop', 'season': 'kharif', 'yield': 1.5, 'price': 200.0, 'cost': 150000},
            'cotton': {'name_hindi': 'कपास', 'type': 'cash crop', 'season': 'kharif', 'yield': 2.8, 'price': 55.0, 'cost': 45000},
            'grapes': {'name_hindi': 'अंगूर', 'type': 'fruit', 'season': 'rabi', 'yield': 20.0, 'price': 60.0, 'cost': 200000},
            'jute': {'name_hindi': 'जूट', 'type': 'fiber', 'season': 'kharif', 'yield': 3.5, 'price': 40.0, 'cost': 35000},
            'kidneybeans': {'name_hindi': 'राजमा', 'type': 'pulse', 'season': 'rabi', 'yield': 1.8, 'price': 120.0, 'cost': 40000},
            'lentil': {'name_hindi': 'मसूर', 'type': 'pulse', 'season': 'rabi', 'yield': 1.5, 'price': 70.0, 'cost': 25000},
            'maize': {'name_hindi': 'मक्का', 'type': 'cereal', 'season': 'kharif', 'yield': 6.5, 'price': 20.0, 'cost': 35000},
            'mungbean': {'name_hindi': 'मूंग', 'type': 'pulse', 'season': 'kharif', 'yield': 1.0, 'price': 80.0, 'cost': 30000},
            'muskmelon': {'name_hindi': 'खरबूजा', 'type': 'fruit', 'season': 'zaid', 'yield': 12.0, 'price': 25.0, 'cost': 45000},
            'orange': {'name_hindi': 'संतरा', 'type': 'fruit', 'season': 'rabi', 'yield': 18.0, 'price': 40.0, 'cost': 100000},
            'papaya': {'name_hindi': 'पपीता', 'type': 'fruit', 'season': 'kharif', 'yield': 35.0, 'price': 20.0, 'cost': 60000},
            'pigeonpeas': {'name_hindi': 'अरहर', 'type': 'pulse', 'season': 'kharif', 'yield': 2.0, 'price': 65.0, 'cost': 35000},
            'pomegranate': {'name_hindi': 'अनार', 'type': 'fruit', 'season': 'rabi', 'yield': 12.0, 'price': 100.0, 'cost': 150000},
            'rice': {'name_hindi': 'चावल', 'type': 'cereal', 'season': 'kharif', 'yield': 4.5, 'price': 25.0, 'cost': 40000},
            'sugarcane': {'name_hindi': 'गन्ना', 'type': 'cash crop', 'season': 'kharif', 'yield': 75.0, 'price': 3.5, 'cost': 80000},
            'watermelon': {'name_hindi': 'तरबूज', 'type': 'fruit', 'season': 'zaid', 'yield': 20.0, 'price': 15.0, 'cost': 40000},
            'wheat': {'name_hindi': 'गेहूं', 'type': 'cereal', 'season': 'rabi', 'yield': 4.2, 'price': 22.0, 'cost': 35000},
            'mothbeans': {'name_hindi': 'मोठ', 'type': 'pulse', 'season': 'kharif', 'yield': 0.8, 'price': 60.0, 'cost': 20000}
        }
        
        info = crop_info.get(crop_name, {
            'name_hindi': crop_name,
            'type': 'crop',
            'season': 'kharif',
            'yield': 3.0,
            'price': 40.0,
            'cost': 50000
        })
        
        db_crop = DBCrop(
            name=crop_name,
            name_hindi=info['name_hindi'],
            crop_type=info['type'],
            season=info['season'],
            avg_yield_per_hectare=info['yield'],
            avg_price_per_kg=info['price'],
            cultivation_cost_per_hectare=info['cost'],
            growth_duration_days=90
        )
        
        db.add(db_crop)
        db.commit()
        db.refresh(db_crop)
        
        return db_crop
    
    def _calculate_predicted_yield(self, crop: DBCrop, features: dict, confidence: float) -> float:
        """Calculate predicted yield based on crop and conditions"""
        base_yield = crop.avg_yield_per_hectare or 3.0
        
        # Adjust yield based on confidence and conditions
        yield_factor = 0.7 + (confidence * 0.6)  # 0.7 to 1.3 range
        
        # Simple adjustments based on conditions
        if features['ph'] < 5.5 or features['ph'] > 8.0:
            yield_factor *= 0.9  # Reduce yield for extreme pH
        
        if features['temperature'] < 10 or features['temperature'] > 40:
            yield_factor *= 0.85  # Reduce yield for extreme temperature
        
        predicted_yield = base_yield * yield_factor
        return round(predicted_yield, 2)
    
    def _calculate_estimated_profit(self, crop: DBCrop, predicted_yield: float) -> float:
        """Calculate estimated profit per hectare"""
        price_per_kg = crop.avg_price_per_kg or 40.0
        cost_per_hectare = crop.cultivation_cost_per_hectare or 50000.0
        
        # Convert yield from tonnes to kg (if needed)
        if predicted_yield < 100:  # Assuming tonnes
            yield_kg = predicted_yield * 1000
        else:  # Already in kg
            yield_kg = predicted_yield
        
        revenue = yield_kg * price_per_kg
        profit = revenue - cost_per_hectare
        
        return round(profit, 2)
    
    def _calculate_sustainability_score(self, features: dict, crop: DBCrop) -> float:
        """Calculate sustainability score (0-1)"""
        score = 0.7  # Base score
        
        # Adjust based on water requirement vs availability
        if features['rainfall'] > 100 and crop.season == 'kharif':
            score += 0.1  # Good for monsoon crops
        elif features['rainfall'] < 50 and crop.season == 'rabi':
            score += 0.1  # Good for winter crops with less water
        
        # Soil health factors
        if 6.0 <= features['ph'] <= 7.5:
            score += 0.1  # Optimal pH range
        
        # Nutrient balance
        npk_balance = abs(features['nitrogen'] - features['phosphorus']) / max(features['nitrogen'], features['phosphorus'])
        if npk_balance < 0.5:
            score += 0.1  # Good nutrient balance
        
        return min(round(score, 2), 1.0)
    
    def _determine_season(self, features: dict, crop: DBCrop) -> str:
        """Determine the appropriate season based on conditions"""
        if crop.season:
            return crop.season
        
        # Simple season determination based on temperature and rainfall
        temp = features['temperature']
        rainfall = features['rainfall']
        
        if temp >= 25 and rainfall >= 100:
            return 'kharif'  # Monsoon season
        elif temp <= 25 and rainfall <= 50:
            return 'rabi'    # Winter season
        else:
            return 'zaid'    # Summer season
    
    def _generate_remarks(self, crop_name: str, confidence: float, features: dict) -> str:
        """Generate helpful remarks for the recommendation"""
        remarks = []
        
        if confidence >= 0.8:
            remarks.append(f"{crop_name.title()} is highly suitable for these conditions.")
        elif confidence >= 0.6:
            remarks.append(f"{crop_name.title()} is moderately suitable.")
        else:
            remarks.append(f"{crop_name.title()} may face some challenges in these conditions.")
        
        # Add specific advice
        if features['ph'] < 6.0:
            remarks.append("Consider lime application to increase soil pH.")
        elif features['ph'] > 7.5:
            remarks.append("Consider organic matter addition to balance soil pH.")
        
        if features['nitrogen'] < 40:
            remarks.append("Nitrogen supplementation recommended.")
        
        if features['rainfall'] < 50:
            remarks.append("Irrigation will be essential.")
        elif features['rainfall'] > 200:
            remarks.append("Ensure proper drainage to prevent waterlogging.")
        
        return " ".join(remarks)
    
    def _save_recommendation_to_db(self, input_data: RecommendationInput, recommendation: RecommendationResult, db: Session):
        """Save recommendation to database"""
        try:
            db_recommendation = DBRecommendation(
                user_id=input_data.user_id,
                crop_id=recommendation.crop_id,
                soil_ph=input_data.soil_ph,
                nitrogen=input_data.nitrogen,
                phosphorus=input_data.phosphorus,
                potassium=input_data.potassium,
                temperature=input_data.temperature,
                humidity=input_data.humidity,
                rainfall=input_data.rainfall,
                location=input_data.location,
                confidence_score=recommendation.confidence_score,
                predicted_yield=recommendation.predicted_yield,
                estimated_profit=recommendation.estimated_profit,
                sustainability_score=recommendation.sustainability_score,
                season=recommendation.season,
                remarks=recommendation.remarks
            )
            
            db.add(db_recommendation)
            db.commit()
        except Exception as e:
            print(f"Error saving recommendation to database: {e}")
            db.rollback()
