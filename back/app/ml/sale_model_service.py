"""
Sale Model Service - RandomForest pipeline from train6.py
"""
from pathlib import Path
from typing import Dict, Any, List

import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app.core.config import settings


class SaleModelService:
    """Service for sale price prediction using train6 pipeline"""

    def __init__(self):
        self.model = None
        self.feature_info = None
        self.knn_model = None
        self.scaler = None
        self.data = None
        self.loaded = False
        self.last_error = None
        self.numeric_features = ["surface", "rooms", "bathrooms"]

    def load(self):
        """Load model pipeline, feature info, data, and KNN model from disk"""
        try:
            model_dir = Path(settings.SALE_MODEL_PATH)
            model_path = model_dir / "model.pkl"
            feature_info_path = model_dir / "feature_info.pkl"

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            if not feature_info_path.exists():
                raise FileNotFoundError(f"Feature info not found at {feature_info_path}")

            self.model = joblib.load(model_path)
            self.feature_info = joblib.load(feature_info_path)
            
            # Load training data for KNN
            project_root = Path(__file__).parent.parent.parent.parent
            data_path = project_root / "ML" / "data" / "sale_processed.csv"
            if data_path.exists():
                self.data = pd.read_csv(data_path)
                self._fit_knn()
            else:
                print(f"⚠️ Training data not found at {data_path}, KNN neighbors will be unavailable")
            
            self.loaded = True
            self.last_error = None
            print("✅ Sale RandomForest model loaded successfully")
        except Exception as exc:
            self.loaded = False
            self.last_error = str(exc)
            print(f"❌ Error loading sale model: {exc}")
            raise

    def _build_feature_row(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        surface = float(payload.get("surface") or 0)
        rooms = int(payload.get("rooms") or 0)
        bathrooms = int(payload.get("bathrooms") or 0)

        feature_row = {
            "region": payload.get("region"),
            "city": payload.get("city"),
            "property_type": payload.get("property_type"),
            "price_segment": payload.get("price_segment") or "mid",
            "surface": surface,
            "rooms": rooms,
            "bathrooms": bathrooms,
            "property_type_cluster": int(payload.get("property_type_cluster") or 0),
            "has_piscine": int(bool(payload.get("has_piscine"))),
            "has_garage": int(bool(payload.get("has_garage"))),
            "has_jardin": int(bool(payload.get("has_jardin"))),
            "has_terrasse": int(bool(payload.get("has_terrasse"))),
            "has_ascenseur": int(bool(payload.get("has_ascenseur"))),
            "is_meuble": int(bool(payload.get("is_meuble"))),
            "has_chauffage": int(bool(payload.get("has_chauffage"))),
            "has_climatisation": int(bool(payload.get("has_climatisation"))),
        }

        feature_row["surface_per_room"] = surface / (rooms if rooms > 0 else 1)
        feature_row["bathroom_ratio"] = bathrooms / (rooms if rooms > 0 else 1)

        amenity_cols = [
            "has_piscine",
            "has_garage",
            "has_jardin",
            "has_terrasse",
            "has_ascenseur",
            "has_chauffage",
            "has_climatisation",
        ]
        feature_row["amenity_score"] = sum(feature_row[col] for col in amenity_cols)
        feature_row["luxury_score"] = (
            feature_row["has_piscine"] * 3
            + feature_row["has_jardin"] * 2
            + feature_row["has_garage"] * 2
            + feature_row["has_climatisation"] * 1.5
            + feature_row["has_terrasse"] * 1
            + feature_row["has_ascenseur"] * 1
            + feature_row["has_chauffage"] * 1
        )

        if surface <= 75:
            feature_row["size_category"] = "Small"
        elif surface <= 120:
            feature_row["size_category"] = "Medium"
        elif surface <= 180:
            feature_row["size_category"] = "Large"
        else:
            feature_row["size_category"] = "Very_Large"

        feature_row["room_density"] = rooms / (surface if surface > 0 else 1)

        return feature_row

    def _fit_knn(self):
        """Fit KNN model on numeric features for similarity search"""
        try:
            # Extract numeric features for KNN
            X_numeric = self.data[self.numeric_features].values
            
            # Standardize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_numeric)
            
            # Fit KNN
            self.knn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
            self.knn_model.fit(X_scaled)
            print("✅ KNN model fitted for similarity search")
        except Exception as exc:
            print(f"⚠️ Could not fit KNN: {exc}")
            self.knn_model = None

    def find_similar_properties(self, payload: Dict[str, Any], n_neighbors: int = 5) -> List[Dict[str, Any]]:
        """Find similar properties using KNN"""
        if self.knn_model is None or self.data is None:
            return []
        
        try:
            # Extract query features
            query = np.array([[
                float(payload.get("surface", 0)),
                int(payload.get("rooms") or 0),
                int(payload.get("bathrooms") or 0)
            ]])
            
            # Scale query
            query_scaled = self.scaler.transform(query)
            
            # Find neighbors (n_neighbors+1 to account for potential self-match)
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
                    "similarity_score": float(1 / (1 + distance)),  # Convert distance to similarity
                    "has_piscine": bool(prop.get("has_piscine", False)),
                    "has_garage": bool(prop.get("has_garage", False)),
                    "has_jardin": bool(prop.get("has_jardin", False)),
                    "has_terrasse": bool(prop.get("has_terrasse", False)),
                    "has_ascenseur": bool(prop.get("has_ascenseur", False)),
                })
            
            return similar_properties[:n_neighbors]
        except Exception as exc:
            print(f"⚠️ Error finding similar properties: {exc}")
            return []

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        row = self._build_feature_row(payload)
        feature_columns = self.feature_info.get("feature_columns", [])

        X = pd.DataFrame([[row.get(col) for col in feature_columns]], columns=feature_columns)
        prediction = float(self.model.predict(X)[0])
        
        # Find similar properties
        similar_properties = self.find_similar_properties(payload)

        return {
            "predicted_price": round(prediction, 2),
            "currency": "TND",
            "model": "sale_random_forest_train6",
            "similar_properties": similar_properties,
        }


sale_model_service = SaleModelService()
