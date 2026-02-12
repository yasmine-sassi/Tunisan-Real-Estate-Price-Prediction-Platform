"""
Sale Prediction API Endpoints (train6 model)
"""
from fastapi import APIRouter, HTTPException

from app.models.schemas import SalePredictionRequest, SalePredictionResponse
from app.ml.sale_model_service import sale_model_service

router = APIRouter()


@router.post("/predict", response_model=SalePredictionResponse)
async def predict_sale(request: SalePredictionRequest):
    """Predict sale price using the train6 RandomForest pipeline"""
    try:
        if not sale_model_service.loaded:
            sale_model_service.load()

        result = sale_model_service.predict(request.features.model_dump())
        return SalePredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@router.get("/model-info")
async def get_sale_model_info():
    """Get info about the sale model"""
    return {
        "loaded": sale_model_service.loaded,
        "error": sale_model_service.last_error,
        "model": "sale_random_forest_train6",
    }
