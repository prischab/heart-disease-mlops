"""
schema.py — Defines the shape of API requests and responses using Pydantic.

Pydantic validates incoming data automatically: if someone sends the wrong
type or a missing field, FastAPI returns a clear error instead of crashing.
The 13 fields match the features the model was trained on.
"""

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """The 13 inputs the heart-disease model expects."""
    age: float = Field(..., description="Age in years")
    sex: float = Field(..., description="Sex (1 = male, 0 = female)")
    cp: float = Field(..., description="Chest pain type (0-3)")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1/0)")
    restecg: float = Field(..., description="Resting ECG results (0-2)")
    thalach: float = Field(..., description="Maximum heart rate achieved")
    exang: float = Field(..., description="Exercise-induced angina (1/0)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: float = Field(..., description="Slope of peak exercise ST segment (0-2)")
    ca: float = Field(..., description="Number of major vessels (0-3)")
    thal: float = Field(..., description="Thalassemia (0-3)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
                "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
                "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = disease, 0 = no disease")
    label: str = Field(..., description="Human-readable label")
    probability: float = Field(..., description="Model confidence for the predicted class")