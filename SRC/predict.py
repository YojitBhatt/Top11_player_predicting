"""Utility for making predictions and returning top 11 players."""

import joblib
import pandas as pd
from pathlib import Path
from config import MODEL_PATH

_model = None  # global cache

def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict(df: pd.DataFrame, top_n: int = 11) -> pd.DataFrame:
    """Return DataFrame of top_n players sorted by predicted probability."""
    model = _get_model()
    proba = model.predict_proba(df)[:, 1]
    out = df.copy()
    out['pred_proba_top'] = proba
    return out.sort_values('pred_proba_top', ascending=False).head(top_n)
