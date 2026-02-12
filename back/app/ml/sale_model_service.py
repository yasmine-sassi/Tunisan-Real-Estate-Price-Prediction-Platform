"""
Sale Model Service - RandomForest pipeline from train6.py
"""
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from app.core.config import settings


class SaleModelService:
    """Service for sale price prediction using train6 pipeline"""

    def __init__(self):
        self.model = None
        self.feature_info = None
        self.loaded = False
        self.last_error = None

    def load(self):
        """Load model pipeline and feature info from disk"""
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

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        row = self._build_feature_row(payload)
        feature_columns = self.feature_info.get("feature_columns", [])

        X = pd.DataFrame([[row.get(col) for col in feature_columns]], columns=feature_columns)
        prediction = float(self.model.predict(X)[0])

        return {
            "predicted_price": round(prediction, 2),
            "currency": "TND",
            "model": "sale_random_forest_train6",
        }


sale_model_service = SaleModelService()
