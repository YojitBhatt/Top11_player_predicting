import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifact"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
ENCODER_PATH = ARTIFACT_DIR / "encoder.pkl"
