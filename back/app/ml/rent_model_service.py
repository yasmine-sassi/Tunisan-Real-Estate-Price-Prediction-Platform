"""
Rent Model Service - RandomForest pipeline from train5.py
"""
from pathlib import Path
from typing import Dict, Any, List

import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder

from app.core.config import settings


class RentModelService:
    """Service for rent price prediction using train5 pipeline"""

    def __init__(self):
        self.model = None
        self.knn_model = None
        self.scaler = None
        self.data = None
        self.data_encoded = None
        self.loaded = False
        self.last_error = None
        self.feature_columns = [
            "surface",
            "rooms",
            "bathrooms",
            "region",
            "property_type",
            "city",
            "price_segment",
            "has_piscine",
            "has_garage",
            "has_jardin",
            "has_terrasse",
            "has_ascenseur",
            "is_meuble",
            "has_chauffage",
            "has_climatisation",
        ]
        # All features for KNN
        self.knn_features = self.feature_columns.copy()

    def load(self):
        """Load model pipeline, data, and KNN model from disk"""
        try:
            model_path = Path(settings.RENT_MODEL_PATH) / "model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")

            self.model = joblib.load(model_path)
            
            # Load training data for KNN
            project_root = Path(__file__).parent.parent.parent.parent
            data_path = project_root / "ML" / "data" / "rent_processed.csv"
            if data_path.exists():
                self.data = pd.read_csv(data_path)
                self._fit_knn()
            else:
                print(f"⚠️ Training data not found at {data_path}, KNN neighbors will be unavailable")
            
            self.loaded = True
            self.last_error = None
            print("✅ Rent RandomForest model loaded successfully")
        except Exception as exc:
            self.loaded = False
            self.last_error = str(exc)
            print(f"❌ Error loading rent model: {exc}")
            raise

    def _fit_knn(self):
        """Fit KNN model on all features for similarity search with weighted importance"""
        try:
            # Copy data for encoding
            df_encoded = self.data[self.knn_features].copy()
            
            # Encode categorical features
            categorical_features = ["region", "property_type", "city", "price_segment"]
            for col in categorical_features:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
            # Convert boolean to numeric
            boolean_features = [
                "has_piscine", "has_garage", "has_jardin", "has_terrasse",
                "has_ascenseur", "is_meuble", "has_chauffage", "has_climatisation"
            ]
            for col in boolean_features:
                df_encoded[col] = df_encoded[col].astype(int)
            
            # Extract features and apply weights
            X_all = df_encoded.values
            
            # Apply feature weights: region and city get 3x weight, property_type gets 2x
            # This ensures properties in same city/region are prioritized
            feature_weights = []
            for feat in self.knn_features:
                if feat in ["region", "city"]:
                    feature_weights.append(3.0)  # 3x weight for location
                elif feat in ["property_type"]:
                    feature_weights.append(2.0)  # 2x weight for property type
                else:
                    feature_weights.append(1.0)  # 1x weight for other features
            
            # Apply weights
            X_weighted = X_all * np.array(feature_weights)
            
            # Standardize weighted features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_weighted)
            
            # Fit KNN on weighted features
            self.knn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
            self.knn_model.fit(X_scaled)
            self.data_encoded = df_encoded
            print("✅ KNN model fitted with weighted features (city/region prioritized)")
        except Exception as exc:
            print(f"⚠️ Could not fit KNN: {exc}")
            self.knn_model = None

    def find_similar_properties(self, payload: Dict[str, Any], n_neighbors: int = 5) -> List[Dict[str, Any]]:
        """Find similar properties using KNN with all features"""
        if self.knn_model is None or self.data is None:
            return []
        
        try:
            # Build query row with all features in the same order as training
            query_row = {
                "surface": float(payload["surface"]),
                "rooms": int(payload.get("rooms") or 0),
                "bathrooms": int(payload.get("bathrooms") or 0),
                "region": payload["region"],
                "property_type": payload["property_type"],
                "city": payload["city"],
                "price_segment": payload.get("price_segment") or "mid",
                "has_piscine": int(bool(payload.get("has_piscine"))),
                "has_garage": int(bool(payload.get("has_garage"))),
                "has_jardin": int(bool(payload.get("has_jardin"))),
                "has_terrasse": int(bool(payload.get("has_terrasse"))),
                "has_ascenseur": int(bool(payload.get("has_ascenseur"))),
                "is_meuble": int(bool(payload.get("is_meuble"))),
                "has_chauffage": int(bool(payload.get("has_chauffage"))),
                "has_climatisation": int(bool(payload.get("has_climatisation"))),
            }
            
            # Build query dataframe with all features
            query_df = pd.DataFrame([query_row])
            
            # Encode categorical features same way as training data
            categorical_features = ["region", "property_type", "city", "price_segment"]
            for col in categorical_features:
                le = LabelEncoder()
                unique_vals = self.data[col].astype(str).unique()
                le.fit(unique_vals)
                # Handle unseen categories
                query_val = str(query_df[col].iloc[0])
                if query_val not in unique_vals:
                    query_val = unique_vals[0]
                query_df[col] = le.transform([query_val])[0]
            
            # Reorder columns to match training features
            query_df = query_df[self.knn_features]
            
            # Apply same feature weights as training
            feature_weights = []
            for feat in self.knn_features:
                if feat in ["region", "city"]:
                    feature_weights.append(3.0)  # 3x weight for location
                elif feat in ["property_type"]:
                    feature_weights.append(2.0)  # 2x weight for property type
                else:
                    feature_weights.append(1.0)  # 1x weight for other features
            
            # Apply weights to query
            query_weighted = query_df.values * np.array(feature_weights)
            
            # Scale query using training scaler
            query_scaled = self.scaler.transform(query_weighted)
            
            # Find neighbors
            distances, indices = self.knn_model.kneighbors(query_scaled, n_neighbors=n_neighbors + 1)
            
            # Get similar properties
            similar_properties = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= len(self.data):
                    continue
                    
                prop = self.data.iloc[idx]
                similar_properties.append({
                    "region": str(prop.get("region", "")),
                    "city": str(prop.get("city", "")),
                    "property_type": str(prop.get("property_type", "")),
                    "surface": float(prop.get("surface", 0)),
                    "rooms": int(prop.get("rooms", 0)),
                    "bathrooms": int(prop.get("bathrooms", 0)),
                    "price": float(prop.get("price", 0)),
                    "similarity_score": float(1 / (1 + distance)),
                    "has_piscine": bool(prop.get("has_piscine", False)),
                    "has_garage": bool(prop.get("has_garage", False)),
                    "has_jardin": bool(prop.get("has_jardin", False)),
                    "has_terrasse": bool(prop.get("has_terrasse", False)),
                    "has_ascenseur": bool(prop.get("has_ascenseur", False)),
                })
            
            return similar_properties[:n_neighbors]
        except Exception as exc:
            print(f"⚠️ Error finding similar properties: {exc}")
            import traceback
            traceback.print_exc()
            return []

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        row = {
            "surface": float(payload["surface"]),
            "rooms": int(payload.get("rooms") or 0),
            "bathrooms": int(payload.get("bathrooms") or 0),
            "region": payload["region"],
            "property_type": payload["property_type"],
            "city": payload["city"],
            "price_segment": payload.get("price_segment") or "mid",
            "has_piscine": bool(payload.get("has_piscine")),
            "has_garage": bool(payload.get("has_garage")),
            "has_jardin": bool(payload.get("has_jardin")),
            "has_terrasse": bool(payload.get("has_terrasse")),
            "has_ascenseur": bool(payload.get("has_ascenseur")),
            "is_meuble": bool(payload.get("is_meuble")),
            "has_chauffage": bool(payload.get("has_chauffage")),
            "has_climatisation": bool(payload.get("has_climatisation")),
        }

        X = pd.DataFrame([[row[col] for col in self.feature_columns]], columns=self.feature_columns)
        prediction = float(self.model.predict(X)[0])
        
        # Find similar properties
        similar_properties = self.find_similar_properties(payload)

        return {
            "predicted_price": round(prediction, 2),
            "currency": "TND",
            "model": "rent_random_forest_train5",
            "similar_properties": similar_properties,
        }


rent_model_service = RentModelService()
