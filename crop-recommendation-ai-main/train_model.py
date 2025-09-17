import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

# Load the dataset
print("Loading crop data...")
df = pd.read_csv('data/crop_data.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Crops available: {df['label'].unique()}")

# Prepare features and labels
X = df.drop('label', axis=1)
y = df['label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train the model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head())

# Save the model and label encoder
print("\nSaving model and encoder...")
joblib.dump(model, 'models/crop_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

# Save crop information
crop_info = {
    'crops': list(le.classes_),
    'features': list(X.columns),
    'feature_ranges': {
        'N': {'min': float(df['N'].min()), 'max': float(df['N'].max())},
        'P': {'min': float(df['P'].min()), 'max': float(df['P'].max())},
        'K': {'min': float(df['K'].min()), 'max': float(df['K'].max())},
        'temperature': {'min': float(df['temperature'].min()), 'max': float(df['temperature'].max())},
        'humidity': {'min': float(df['humidity'].min()), 'max': float(df['humidity'].max())},
        'ph': {'min': float(df['ph'].min()), 'max': float(df['ph'].max())},
        'rainfall': {'min': float(df['rainfall'].min()), 'max': float(df['rainfall'].max())}
    }
}

with open('models/crop_info.json', 'w') as f:
    json.dump(crop_info, f, indent=2)

print("\nModel training complete!")
print(f"Model saved to: models/crop_model.pkl")
print(f"Label encoder saved to: models/label_encoder.pkl")
print(f"Crop info saved to: models/crop_info.json")
