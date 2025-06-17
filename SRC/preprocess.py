import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_preprocess_pipeline(df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """Return (pipeline, feature_columns)."""
    # Exclude leakage columns
    target_cols = ["top_performer", "performance_score"]
    id_cols = ["match_id", "player_name"]
    feature_cols = [c for c in df.columns if c not in target_cols + id_cols]

    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    return preprocessor, feature_cols
