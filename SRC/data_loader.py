import pandas as pd
from pathlib import Path

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load CSV dataset."""
    return pd.read_csv(path)
