"""
test_app.py — Automated tests for the heart-disease prediction API.

These are what the CI pipeline runs on every push. If any test fails,
CI goes red and you know something broke before it ships.

Run locally:
    pytest

Uses FastAPI's TestClient, which calls the app in-process — no need to
have the server running separately.
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# A known-valid patient input (all 13 features)
VALID_PATIENT = {
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
    "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
}


def test_health_ok():
    """The health endpoint should report the service is up and model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_root_responds():
    """The root endpoint should return basic service info."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_valid_input():
    """A valid patient should get a well-formed prediction back."""
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()

    # Prediction must be 0 or 1
    assert body["prediction"] in (0, 1)
    # Label must match the prediction
    assert body["label"] in ("disease", "no disease")
    # Probability must be a valid confidence value
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_missing_field_rejected():
    """Missing a required feature should return a 422 validation error,
    not crash the server."""
    bad_input = dict(VALID_PATIENT)
    del bad_input["age"]  # remove a required field
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_predict_wrong_type_rejected():
    """A non-numeric value should be rejected with a validation error."""
    bad_input = dict(VALID_PATIENT)
    bad_input["age"] = "not a number"
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422