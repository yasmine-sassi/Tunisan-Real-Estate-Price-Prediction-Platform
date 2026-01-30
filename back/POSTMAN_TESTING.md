# Combined Model API Testing with Postman

## Quick Start Guide

### 1. Start the FastAPI Server

From the `back/` directory, run:

```bash
cd c:\Users\Mediatek\Desktop\ML-project\back
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### 2. API Documentation

Once running, access the interactive API docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Postman Testing

### Endpoint: POST /api/v1/combined/predict-combined

**URL:** `http://localhost:8000/api/v1/combined/predict-combined`

**Method:** POST

**Headers:**
```
Content-Type: application/json
```

### Example Requests

#### Example 1: Apartment Sale in Tunis
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

#### Example 2: Apartment Rent in Ariana
```json
{
  "transaction": "rent",
  "city": "Ariana",
  "region": "Centre Ville",
  "property_type": "apartment",
  "surface": 85,
  "bathrooms": 1,
  "rooms": 3
}
```

#### Example 3: Villa Sale in Sousse
```json
{
  "transaction": "sale",
  "city": "Sousse",
  "region": "Sahloul",
  "property_type": "villa",
  "surface": 300,
  "bathrooms": 3,
  "rooms": 6
}
```

#### Example 4: Studio Rent in Sfax
```json
{
  "transaction": "rent",
  "city": "Sfax",
  "region": "Centre Ville",
  "property_type": "apartment",
  "surface": 45,
  "bathrooms": 1,
  "rooms": 1
}
```

### Expected Response Format

```json
{
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
```

---

## Additional Endpoints

### Check Model Status

**URL:** `http://localhost:8000/api/v1/combined/model-status`

**Method:** GET

**Response:**
```json
{
  "loaded": true,
  "model": "combined_Random_Forest",
  "model_type": "Random Forest with 600 estimators",
  "performance": {
    "r2_score": 0.8372,
    "mae": 48055,
    "note": "Trained on combined rent+sale dataset"
  }
}
```

### Health Check

**URL:** `http://localhost:8000/health`

**Method:** GET

---

## Troubleshooting

### Model Not Loading

If you see "Combined model not loaded" error:

1. Check that MLflow server is running:
```bash
cd c:\Users\Mediatek\Desktop\ML-project\ML
mlflow ui --port 5000
```

2. Verify the model exists in MLflow registry:
- Open http://localhost:5000
- Check "Models" tab for "combined_Random_Forest"

3. Verify preprocessor file exists:
```bash
Test-Path "c:\Users\Mediatek\Desktop\ML-project\ML\data\preprocessor_combined.joblib"
```

### Common Errors

**Error:** "Unknown city value"
- **Cause:** The city name doesn't exist in training data
- **Solution:** The model will use a fallback value and print a warning

**Error:** "Prediction error"
- **Cause:** Invalid input data types
- **Solution:** Ensure `surface` is float, `bathrooms` and `rooms` are integers

---

## Postman Collection Setup

1. Create a new Collection: "Real Estate Combined Model"

2. Add Request "Predict Property Price":
   - Method: POST
   - URL: `{{base_url}}/api/v1/combined/predict-combined`
   - Body: raw JSON (use examples above)

3. Add Environment variable:
   - Variable: `base_url`
   - Value: `http://localhost:8000`

4. Add Request "Model Status":
   - Method: GET
   - URL: `{{base_url}}/api/v1/combined/model-status`

---

## Model Performance

The combined Random Forest model achieves:
- **R² Score:** 0.8372 (83.72% variance explained)
- **MAE:** 48,055 TND
- **Overfitting Gap:** 7.5% (excellent generalization)

This significantly outperforms the previous split approach (R²=0.5172 for rent-only).
