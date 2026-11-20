"""Loading, validation, cleaning, splitting and preprocessing for Indian premium data."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import DEFAULT_DATA_PATH, RANDOM_STATE, REQUIRED_COLUMNS, TARGET_COLUMN, RAW_NUMERICAL_FEATURES, RAW_CATEGORICAL_FEATURES

COLUMN_RENAME = {
    "Age": "age", "Diabetes": "diabetes",
    "BloodPressureProblems": "blood_pressure_problems",
    "AnyTransplants": "any_transplants",
    "AnyChronicDiseases": "any_chronic_diseases",
    "Height": "height_cm", "Weight": "weight_kg",
    "KnownAllergies": "known_allergies",
    "HistoryOfCancerInFamily": "family_cancer_history",
    "NumberOfMajorSurgeries": "major_surgeries",
    "PremiumPrice": "premium_price",
}

def load_dataset(filepath: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Indian dataset not found: {path}. Download Medicalpremium.csv from "
            "https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction "
            "and save it as data/insurance.csv."
        )
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_RENAME)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset schema mismatch. Missing columns: {missing}. "
                         f"Expected columns: {REQUIRED_COLUMNS}")
    return df[REQUIRED_COLUMNS].copy()

def inspect_dataset(df: pd.DataFrame) -> Dict[str, object]:
    return {"shape": df.shape, "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing_values": df.isna().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "summary_stats": df.describe(include="all")}

def detect_outliers_iqr(df: pd.DataFrame, columns: List[str] | None = None) -> Dict[str, Dict[str, float]]:
    columns = columns or RAW_NUMERICAL_FEATURES + [TARGET_COLUMN]
    report = {}
    for col in columns:
        if col not in df or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        count = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        report[col] = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr), "outlier_count": count}
    return report

def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cleaned = df.rename(columns=COLUMN_RENAME).copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in cleaned.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    cleaned = cleaned[REQUIRED_COLUMNS]
    for col in RAW_NUMERICAL_FEATURES + [TARGET_COLUMN]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    for col in RAW_CATEGORICAL_FEATURES:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    report = {"rows_before": before, "rows_after": len(cleaned),
              "duplicates_removed": before - len(cleaned) - int(df[TARGET_COLUMN].isna().sum()) if TARGET_COLUMN in df else 0,
              "missing_values_after_cleaning": {k: int(v) for k, v in cleaned.isna().sum().items()},
              "outliers": detect_outliers_iqr(cleaned)}
    return cleaned, report

def split_features_target(df: pd.DataFrame):
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]

def split_train_test(X, y, test_size=0.2, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_preprocessor(numerical_features=None, categorical_features=None, scale_numeric=True):
    numerical_features = numerical_features or RAW_NUMERICAL_FEATURES
    categorical_features = categorical_features or RAW_CATEGORICAL_FEATURES
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numerical_features),
        ("cat", cat_pipe, categorical_features),
    ])
