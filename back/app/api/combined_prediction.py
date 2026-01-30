"""
Combined Model Prediction API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
from app.models.combined_schemas import CombinedPropertyInput, PredictionResponse
from app.ml.combined_predictor import combined_model_service
from app.core.config import settings

router = APIRouter()


@router.post("/predict-combined", response_model=PredictionResponse)
async def predict_combined(property_data: CombinedPropertyInput):
    """
    Predict property price using the combined Random Forest model
    
    This endpoint uses a simpler but more accurate model that handles both
    rent and sale transactions in a single unified model.
    
    **Example Request:**
    ```json
    {
        "transaction": "sale",
        "city": "Tunis",
        "region": "El Menzah",
        "property_type": "apartment",
        "surface": 120,
        "bathrooms": 2,
        "rooms": 4
    }
    ```
    
    **Returns:** Predicted price in TND
    """
    try:
        if not combined_model_service.loaded:
            try:
                combined_model_service.load()
            except Exception as load_error:
                raise HTTPException(
                    status_code=503,
                    detail=f"Combined model not loaded: {load_error}"
                )
        
        # Make prediction
        result = combined_model_service.predict(property_data.model_dump())
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@router.get("/model-status")
async def get_model_status():
    """Check if the combined model is loaded and ready"""
    load_error = None
    if not combined_model_service.loaded:
        try:
            combined_model_service.load()
        except Exception as e:
            load_error = str(e)
    return {
        "loaded": combined_model_service.loaded,
        "model": "combined_Random_Forest",
        "model_type": "Random Forest with 600 estimators",
        "last_error": combined_model_service.last_error,
        "load_error": load_error,
        "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
        "preprocessor_path": str((Path(settings.ML_DIR) / "data" / "preprocessor_combined.joblib").resolve()),
        "model_uri": "models:/combined_Random_Forest/latest",
        "performance": {
            "r2_score": 0.8372,
            "mae": 48055,
            "note": "Trained on combined rent+sale dataset"
        }
    }
