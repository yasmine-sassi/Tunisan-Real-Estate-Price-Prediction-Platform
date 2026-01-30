"""
Combined Property Prediction Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional


class CombinedPropertyInput(BaseModel):
    """Input schema for combined model predictions"""
    transaction: str = Field(..., description="Transaction type: 'rent' or 'sale'")
    city: str = Field(..., description="City name (e.g., 'Tunis', 'Ariana', 'Sousse')")
    region: str = Field(..., description="Region/neighborhood name")
    property_type: str = Field(..., description="Property type (e.g., 'apartment', 'villa', 'house')")
    surface: float = Field(..., gt=0, description="Surface area in square meters")
    bathrooms: int = Field(..., ge=0, description="Number of bathrooms")
    rooms: int = Field(..., ge=0, description="Number of rooms")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction": "sale",
                "city": "Tunis",
                "region": "El Menzah",
                "property_type": "apartment",
                "surface": 120,
                "bathrooms": 2,
                "rooms": 4
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    predicted_price: float = Field(..., description="Predicted price")
    currency: str = Field(..., description="Currency code")
    model: str = Field(..., description="Model used for prediction")
    features_used: dict = Field(..., description="Input features used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 350000.50,
                "currency": "TND",
                "model": "combined_Random_Forest",
                "features_used": {
                    "transaction": "sale",
                    "city": "Tunis",
                    "region": "El Menzah",
                    "property_type": "apartment",
                    "surface": 120,
                    "bathrooms": 2,
                    "rooms": 4
                }
            }
        }
