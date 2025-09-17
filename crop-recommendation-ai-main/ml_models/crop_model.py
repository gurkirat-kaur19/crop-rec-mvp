import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import json
from typing import Dict, List, Tuple, Any

class CropRecommendationModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
        ]
        self.is_trained = False
        
    def create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic training data for crop recommendation"""
        np.random.seed(42)
        
        # Define crop requirements (simplified)
        crop_requirements = {
            'rice': {'N': (80, 120), 'P': (40, 60), 'K': (40, 60), 'temp': (20, 30), 'humidity': (80, 90), 'ph': (5.5, 7.0), 'rainfall': (150, 300)},
            'wheat': {'N': (50, 80), 'P': (30, 50), 'K': (30, 50), 'temp': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (50, 100)},
            'maize': {'N': (80, 120), 'P': (40, 60), 'K': (60, 80), 'temp': (18, 27), 'humidity': (60, 80), 'ph': (5.8, 7.0), 'rainfall': (60, 110)},
            'cotton': {'N': (120, 160), 'P': (60, 80), 'K': (50, 70), 'temp': (21, 30), 'humidity': (50, 80), 'ph': (5.8, 8.0), 'rainfall': (50, 100)},
            'sugarcane': {'N': (120, 180), 'P': (40, 60), 'K': (80, 120), 'temp': (21, 27), 'humidity': (75, 85), 'ph': (6.0, 7.5), 'rainfall': (75, 150)},
            'jute': {'N': (60, 100), 'P': (30, 50), 'K': (40, 60), 'temp': (20, 27), 'humidity': (70, 80), 'ph': (4.8, 5.8), 'rainfall': (120, 180)},
            'coffee': {'N': (100, 140), 'P': (60, 80), 'K': (80, 120), 'temp': (15, 25), 'humidity': (70, 80), 'ph': (6.0, 7.0), 'rainfall': (150, 250)},
            'coconut': {'N': (80, 120), 'P': (40, 60), 'K': (120, 180), 'temp': (27, 32), 'humidity': (80, 90), 'ph': (5.2, 8.0), 'rainfall': (130, 250)},
            'papaya': {'N': (100, 140), 'P': (60, 80), 'K': (100, 140), 'temp': (22, 32), 'humidity': (60, 90), 'ph': (6.0, 7.0), 'rainfall': (100, 200)},
            'orange': {'N': (80, 120), 'P': (40, 60), 'K': (60, 100), 'temp': (15, 25), 'humidity': (50, 70), 'ph': (5.5, 7.5), 'rainfall': (100, 120)},
            'apple': {'N': (60, 100), 'P': (40, 60), 'K': (60, 100), 'temp': (15, 25), 'humidity': (50, 60), 'ph': (5.5, 7.0), 'rainfall': (100, 125)},
            'muskmelon': {'N': (100, 120), 'P': (60, 80), 'K': (120, 140), 'temp': (24, 27), 'humidity': (50, 70), 'ph': (6.0, 7.0), 'rainfall': (20, 40)},
            'watermelon': {'N': (100, 120), 'P': (80, 100), 'K': (120, 140), 'temp': (24, 27), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (40, 60)},
            'grapes': {'N': (60, 100), 'P': (60, 80), 'K': (80, 120), 'temp': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.0), 'rainfall': (50, 75)},
            'banana': {'N': (100, 180), 'P': (60, 100), 'K': (300, 400), 'temp': (26, 30), 'humidity': (75, 85), 'ph': (4.5, 7.5), 'rainfall': (75, 180)},
            'pomegranate': {'N': (60, 100), 'P': (40, 60), 'K': (40, 60), 'temp': (15, 25), 'humidity': (35, 45), 'ph': (5.5, 7.5), 'rainfall': (50, 75)},
            'lentil': {'N': (20, 40), 'P': (60, 80), 'K': (20, 40), 'temp': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (25, 50)},
            'blackgram': {'N': (40, 60), 'P': (60, 80), 'K': (20, 40), 'temp': (25, 35), 'humidity': (65, 75), 'ph': (6.0, 7.0), 'rainfall': (60, 100)},
            'mungbean': {'N': (20, 40), 'P': (40, 60), 'K': (20, 40), 'temp': (25, 30), 'humidity': (70, 80), 'ph': (6.2, 7.2), 'rainfall': (50, 75)},
            'mothbeans': {'N': (20, 40), 'P': (40, 60), 'K': (20, 40), 'temp': (24, 27), 'humidity': (65, 75), 'ph': (6.0, 7.5), 'rainfall': (45, 65)},
            'pigeonpeas': {'N': (20, 40), 'P': (60, 80), 'K': (20, 40), 'temp': (18, 29), 'humidity': (60, 65), 'ph': (5.5, 7.5), 'rainfall': (60, 65)},
            'kidneybeans': {'N': (20, 40), 'P': (60, 80), 'K': (20, 40), 'temp': (15, 25), 'humidity': (70, 80), 'ph': (6.0, 7.0), 'rainfall': (45, 55)},
            'chickpea': {'N': (40, 60), 'P': (60, 80), 'K': (40, 60), 'temp': (20, 25), 'humidity': (70, 80), 'ph': (6.2, 7.8), 'rainfall': (40, 50)}
        }
        
        # Generate synthetic data
        data = []
        samples_per_crop = 100
        
        for crop, requirements in crop_requirements.items():
            for _ in range(samples_per_crop):
                sample = {
                    'N': np.random.uniform(requirements['N'][0], requirements['N'][1]),
                    'P': np.random.uniform(requirements['P'][0], requirements['P'][1]),
                    'K': np.random.uniform(requirements['K'][0], requirements['K'][1]),
                    'temperature': np.random.uniform(requirements['temp'][0], requirements['temp'][1]),
                    'humidity': np.random.uniform(requirements['humidity'][0], requirements['humidity'][1]),
                    'ph': np.random.uniform(requirements['ph'][0], requirements['ph'][1]),
                    'rainfall': np.random.uniform(requirements['rainfall'][0], requirements['rainfall'][1]),
                    'label': crop
                }
                data.append(sample)
        
        return pd.DataFrame(data)
    
    def train(self, data: pd.DataFrame = None):
        """Train the crop recommendation model"""
        if data is None:
            data = self.create_synthetic_data()
        
        # Prepare features and target
        X = data[self.feature_names]
        y = data['label']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model training completed. Accuracy: {accuracy:.2f}")
        self.is_trained = True
        
        return accuracy
    
    def predict(self, features: Dict[str, float]) -> List[Tuple[str, float]]:
        """Predict suitable crops for given conditions"""
        if not self.is_trained:
            print("Model not trained. Training with synthetic data...")
            self.train()
        
        # Prepare input features
        feature_array = np.array([[
            features['nitrogen'],
            features['phosphorus'],
            features['potassium'],
            features['temperature'],
            features['humidity'],
            features['ph'],
            features['rainfall']
        ]])
        
        # Scale features
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(feature_array_scaled)[0]
        
        # Get crop names and their probabilities
        crop_names = self.label_encoder.classes_
        predictions = list(zip(crop_names, probabilities))
        
        # Sort by probability (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:5]  # Return top 5 predictions
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
        else:
            print("Model file not found. Training new model...")
            self.train()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return importance
