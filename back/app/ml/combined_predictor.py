"""
ML Inference Service - Combined Model
Handles predictions using the combined Random Forest model
"""
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any
from app.core.config import settings


class CombinedModelService:
    """Service for combined model predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.loaded = False
        self.last_error = None
        
    def load(self):
        """Load the combined model and preprocessor"""
        try:
            # Load preprocessor
            preprocessor_path = (Path(settings.ML_DIR) / "data" / "preprocessor_combined.joblib").resolve()
            if not preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
            
            preproc = joblib.load(preprocessor_path)
            self.scaler = preproc["scaler"]
            self.label_encoders = preproc["label_encoders"]
            self.feature_columns = preproc["feature_columns"]
            
            # Load model from MLflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            model_name = "combined_Random_Forest"
            model_uri = f"models:/{model_name}/latest"
            self.model = mlflow.sklearn.load_model(model_uri)
            
            self.loaded = True
            self.last_error = None
            print(f"✅ Combined model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading combined model: {e}")
            self.loaded = False
            self.last_error = str(e)
            raise
    
    def safe_label_encode(self, value: str, encoder_name: str) -> int:
        """Safely encode a label, fallback to first class if unseen"""
        encoder = self.label_encoders[encoder_name]
        classes = list(encoder.classes_)
        if value in classes:
            return int(encoder.transform([value])[0])
        # Fallback to first known class if unseen
        print(f"⚠️ Unknown {encoder_name} value: {value}, using fallback")
        return int(encoder.transform([classes[0]])[0])
    
    def predict(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a property
        
        Args:
            property_data: Dict with keys:
                - transaction: "rent" or "sale"
                - city: city name
                - region: region name
                - property_type: property type
                - surface: surface area (m²)
                - bathrooms: number of bathrooms
                - rooms: number of rooms
        
        Returns:
            Dict with prediction and metadata
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Build feature row
        encoded = {
            "transaction_encoded": self.safe_label_encode(property_data["transaction"], "transaction"),
            "city_encoded": self.safe_label_encode(property_data["city"], "city"),
            "region_encoded": self.safe_label_encode(property_data["region"], "region"),
            "property_type_encoded": self.safe_label_encode(property_data["property_type"], "property_type"),
            "surface": float(property_data["surface"]),
            "bathrooms": int(property_data["bathrooms"]),
            "rooms": int(property_data["rooms"]),
        }
        
        X = pd.DataFrame([[encoded[col] for col in self.feature_columns]], columns=self.feature_columns)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = float(self.model.predict(X_scaled)[0])
        
        return {
            "predicted_price": round(prediction, 2),
            "currency": "TND",
            "model": "combined_Random_Forest",
            "features_used": property_data
        }


# Global instance
combined_model_service = CombinedModelService()
