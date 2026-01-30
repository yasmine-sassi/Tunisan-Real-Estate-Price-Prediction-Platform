"""
Example prediction script for the combined model.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

# Config
BASE_DIR = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "combined_Random_Forest")
MODEL_STAGE_OR_VERSION = os.getenv("MODEL_STAGE_OR_VERSION", "latest")  # or a version number like "1"

# Paths
PREPROCESSOR_PATH = BASE_DIR / "data" / "preprocessor_combined.joblib"


def safe_label_encode(value, encoder):
    classes = list(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])
    # Fallback to first known class if unseen
    return int(encoder.transform([classes[0]])[0])


def build_feature_row(raw_input, label_encoders, feature_columns):
    encoded = {
        "transaction_encoded": safe_label_encode(raw_input["transaction"], label_encoders["transaction"]),
        "city_encoded": safe_label_encode(raw_input["city"], label_encoders["city"]),
        "region_encoded": safe_label_encode(raw_input["region"], label_encoders["region"]),
        "property_type_encoded": safe_label_encode(raw_input["property_type"], label_encoders["property_type"]),
        "surface": raw_input["surface"],
        "bathrooms": raw_input["bathrooms"],
        "rooms": raw_input["rooms"],
    }
    return pd.DataFrame([[encoded[col] for col in feature_columns]], columns=feature_columns)


def main():
    # Load preprocessor (scaler + label encoders + feature columns)
    preproc = joblib.load(PREPROCESSOR_PATH)
    scaler = preproc["scaler"]
    label_encoders = preproc["label_encoders"]
    feature_columns = preproc["feature_columns"]

    # Example inputs (edit as needed)
    examples = [
        {
            "transaction": "sale",
            "city": "Tunis",
            "region": "El Menzah",
            "property_type": "apartment",
            "surface": 120,
            "bathrooms": 2,
            "rooms": 4,
        },
        {
            "transaction": "rent",
            "city": "Ariana",
            "region": "Ennasr",
            "property_type": "apartment",
            "surface": 85,
            "bathrooms": 1,
            "rooms": 3,
        },
        {
            "transaction": "sale",
            "city": "Sousse",
            "region": "Hammam Sousse",
            "property_type": "villa",
            "surface": 260,
            "bathrooms": 3,
            "rooms": 6,
        },
    ]

    # Load model from MLflow registry
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MODEL_STAGE_OR_VERSION.isdigit():
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE_OR_VERSION}"
    else:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE_OR_VERSION}"

    model = mlflow.sklearn.load_model(model_uri)

    for idx, example in enumerate(examples, start=1):
            X = build_feature_row(example, label_encoders, feature_columns)
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            print(f"Example {idx} input:", example)
            print(f"Example {idx} predicted price: {pred:,.2f} TND")


if __name__ == "__main__":
    main()
