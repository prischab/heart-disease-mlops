"""
train.py — Train a heart disease classifier from a CSV file and save it.

Loads data from data/heart.csv (you download this yourself — see README),
trains a simple classifier, evaluates it, and saves the model + metadata.

The model is deliberately simple. The focus of this project is the
production engineering around it (Docker, CI/CD, monitoring).

Usage:
    python train.py
"""

import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------- Config ----------
DATA_PATH = "data/heart.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")
TARGET_COLUMN = "target"   # the column we're predicting (1 = disease, 0 = no disease)


def main():
    # 1. Load the dataset from the CSV file
    if not os.path.exists(DATA_PATH):
        print(f"Could not find {DATA_PATH}.")
        print("Download heart.csv (see README) and put it in the data/ folder.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {DATA_PATH}")

    # 2. Separate features (X) from the target (y)
    if TARGET_COLUMN not in df.columns:
        print(f"Expected a '{TARGET_COLUMN}' column. Found: {list(df.columns)}")
        return

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    feature_names = list(X.columns)

    # 3. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

    # 6. Save the model + metadata the API will need later
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "target_column": TARGET_COLUMN,
        "classes": {"0": "no disease", "1": "disease"},
        "test_accuracy": round(acc, 4),
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {META_PATH}")
    print(f"Model expects {len(feature_names)} features: {feature_names}")


if __name__ == "__main__":
    main()