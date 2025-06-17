#!/usr/bin/env python
"""Train the model and save it to artifact/."""

import argparse
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from datetime import datetime

from config import MODEL_PATH, ENCODER_PATH
from data_loader import load_dataset
from preprocess import build_preprocess_pipeline

def train(data_path: Path, test_size: float = 0.2, random_state: int = 42):
    df = load_dataset(data_path)

    y = df['top_performer']
    preprocessor, feature_cols = build_preprocess_pipeline(df)
    X = df[feature_cols]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        random_state=random_state,
    )

    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    print(classification_report(y_val, y_pred))

    # Persist
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Saved model to {MODEL_PATH}")

    # Log simple metrics
    with open('logs/training.log', 'a') as f:
        ts = datetime.utcnow().isoformat()
        f.write(f"{ts},f1={f1_score(y_val, y_pred):.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training CSV')
    args = parser.parse_args()
    train(Path(args.data_path))
