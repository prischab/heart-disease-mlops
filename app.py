"""
app.py — FastAPI service that serves the trained heart-disease model.

Endpoints:
  GET  /health   -> is the service up and is the model loaded?
  POST /predict  -> send patient features, get a prediction back
  GET  /         -> basic info

Run it:
    uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs to try it interactively.
"""

import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from schema import PatientFeatures, PredictionResponse

MODEL_PATH = "model/model.pkl"
META_PATH = "model/metadata.json"

app = FastAPI(
    title="Heart Disease Predictor",
    description="A simple ML model served in production (educational demo).",
    version="1.0.0",
)

# Load the model + metadata ONCE at startup, not per request.
try:
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)
    FEATURE_ORDER = metadata["feature_names"]
except Exception as e:
    model = None
    metadata = None
    FEATURE_ORDER = []
    print(f"WARNING: could not load model: {e}")


@app.get("/")
def root():
    return {
        "service": "Heart Disease Predictor",
        "model_loaded": model is not None,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Health check — used later by monitoring and CI."""
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build the feature vector in the EXACT order the model was trained on
    row = [getattr(features, name) for name in FEATURE_ORDER]
    X = np.array(row).reshape(1, -1)

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][pred])  # confidence for predicted class

    return PredictionResponse(
        prediction=pred,
        label="disease" if pred == 1 else "no disease",
        probability=round(proba, 4),
    )