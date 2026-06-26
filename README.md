# Heart Disease Predictor — MLOps Pipeline

![CI](https://github.com/prischab/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)

> Educational project demonstrating production ML engineering — not a medical device.

An end-to-end machine learning pipeline: a classifier trained from data, served through a REST API, containerized with Docker, and automatically tested and built on every push via CI/CD. The model is intentionally simple — the focus is the **production engineering around it**.

---

## Why this project

Building a model is one thing; operating it reliably in production is another. This project demonstrates the second part — the MLOps skills that take a model from "runs in a notebook" to "ships and runs anywhere, with automated quality checks":

- **Containerization** with Docker — runs identically on any machine
- **CI/CD** with GitHub Actions — tests and builds automatically on every push
- **Automated testing** with pytest — including validation and error-handling cases
- **Clean separation** of training and serving

---

## Architecture

```
TRAINING (run once)
  data/heart.csv ─► preprocess ─► train RandomForest ─► save model.pkl

SERVING (runs in a Docker container)
  ┌──────────────────────────────────────────────┐
  │  Docker container                            │
  │    FastAPI app                               │
  │      POST /predict  ─► load model ─► predict │
  │      GET  /health   ─► status check          │
  └──────────────────────────────────────────────┘

CI/CD (runs on GitHub on every push)
  checkout ─► install ─► train ─► pytest ─► docker build  ─► ✓ / ✗
```

---

## Tech stack

| Concern | Tool |
|---|---|
| Model | scikit-learn (RandomForest) |
| Data | UCI Heart Disease dataset |
| API | FastAPI |
| Validation | Pydantic |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## The model

A RandomForest classifier predicting presence of heart disease from 13 clinical features (age, sex, chest pain type, resting blood pressure, cholesterol, etc.). Test accuracy ~0.98 on the held-out set. The model is deliberately simple — the engineering pipeline is the point of this project.

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service + model status (used by CI and monitoring) |
| `/predict` | POST | Send 13 features, get a prediction + confidence |
| `/docs` | GET | Interactive API documentation (auto-generated) |

**Example response:**
```json
{
  "prediction": 0,
  "label": "no disease",
  "probability": 0.97
}
```

---

## CI/CD pipeline

On every push to `main`, GitHub Actions automatically:

1. Checks out the code on a fresh Ubuntu runner
2. Installs Python and dependencies
3. Trains the model (`train.py`)
4. Runs the pytest suite (`python -m pytest`)
5. Builds the Docker image

A green check means tests passed and the container builds; a red X catches regressions before they ship. The badge at the top of this README reflects the latest run.

---

## Running locally

**Prerequisites:** Python 3.10+, Docker (optional, for the container path).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (reads data/heart.csv, saves model/model.pkl)
python train.py

# 3a. Run the API directly
uvicorn app:app --reload
#  -> open http://127.0.0.1:8000/docs

# 3b. OR run it in Docker
docker build -t heart-api .
docker run -p 8000:8000 heart-api
#  -> open http://127.0.0.1:8000/docs

# Run the tests
python -m pytest -v
```

---

## Project structure

```
heart-disease-mlops/
├── .github/workflows/
│   └── ci.yml            # CI/CD pipeline
├── data/
│   └── heart.csv         # UCI Heart Disease dataset
├── model/
│   ├── model.pkl         # trained model
│   └── metadata.json     # feature names, classes, accuracy
├── tests/
│   └── test_app.py       # pytest suite
├── train.py              # train + save the model
├── app.py                # FastAPI serving layer
├── schema.py             # request/response validation
├── conftest.py           # pytest path config
├── Dockerfile            # container definition
├── requirements.txt
└── .dockerignore
```

---

## Dataset

UCI Heart Disease dataset (Cleveland), commonly distributed as `heart.csv` (1025 rows, 13 features + binary target). A widely used public dataset for classification.

---

## Limitations & future work

- **Simple model by design** — the focus is the pipeline, not model performance. Hyperparameter tuning and cross-validation would improve a production model.
- **No live monitoring yet** — a `/metrics` endpoint (request counts, latency, prediction distribution) is a natural next addition.
- **No cloud deployment** — the container is ready to deploy to any host (Render, Railway, Fly.io, a cloud VM); a live endpoint would be the next step.
- **Model is committed for simplicity** — a production setup would version models with a registry (e.g. MLflow) rather than committing the artifact.

---

> Educational and portfolio project. Not validated for clinical use.
