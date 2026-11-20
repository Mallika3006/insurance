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

from src import (
    DEFAULT_DATA_PATH, RANDOM_STATE, REQUIRED_COLUMNS, TARGET_COLUMN,
    RAW_NUMERICAL_FEATURES, RAW_CATEGORICAL_FEATURES,
)

COLUMN_RENAME = {
    "Age": "age",
    "Diabetes": "diabetes",
    "BloodPressureProblems": "blood_pressure_problems",
    "AnyTransplants": "any_transplants",
    "AnyChronicDiseases": "any_chronic_diseases",
    "Height": "height_cm",
    "Weight": "weight_kg",
    "KnownAllergies": "known_allergies",
    "HistoryOfCancerInFamily": "family_cancer_history",
    "NumberOfMajorSurgeries": "major_surgeries",
    "PremiumPrice": "premium_price",
}
US_DATASET_COLUMNS = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
BINARY_COLUMNS = [
    "diabetes", "blood_pressure_problems", "any_transplants",
    "any_chronic_diseases", "known_allergies", "family_cancer_history",
]


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Accept the original Kaggle headers or already-normalised project headers."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out.rename(columns=COLUMN_RENAME)


def _validate_schema(df: pd.DataFrame, source: Path | str) -> None:
    columns = set(df.columns)
    if US_DATASET_COLUMNS.issubset(columns):
        raise ValueError(
            "Wrong dataset detected: data/insurance.csv contains the old US "
            "Medical Cost dataset (age, sex, bmi, children, smoker, region, charges). "
            "This project now requires the Indian Medical Insurance Premium dataset. "
            "Download Medicalpremium.csv from "
            "https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction, "
            "replace data/insurance.csv, and run training again."
        )
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            f"Dataset schema mismatch in {source}. Missing columns: {missing}. "
            f"Expected the Indian dataset columns: {REQUIRED_COLUMNS}. "
            "Check data/README.md and make sure you did not use the old US insurance.csv."
        )


def load_dataset(filepath: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Indian dataset not found: {path}. Download Medicalpremium.csv from "
            "https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction "
            "and save it as data/insurance.csv."
        )
    df = _normalise_column_names(pd.read_csv(path))
    _validate_schema(df, path)
    return df[REQUIRED_COLUMNS].copy()


def inspect_dataset(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "shape": tuple(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "summary_stats": df.describe(include="all").to_dict(),
    }


def detect_outliers_iqr(
    df: pd.DataFrame, columns: List[str] | None = None
) -> Dict[str, Dict[str, float]]:
    columns = columns or RAW_NUMERICAL_FEATURES + [TARGET_COLUMN]
    report = {}
    for col in columns:
        if col not in df or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        count = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        report[col] = {
            "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "outlier_count": count,
        }
    return report


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cleaned = _normalise_column_names(df)
    _validate_schema(cleaned, "provided dataframe")
    cleaned = cleaned[REQUIRED_COLUMNS].copy()

    for col in RAW_NUMERICAL_FEATURES + [TARGET_COLUMN, "major_surgeries"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    yes_no = {"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0}
    for col in BINARY_COLUMNS:
        values = cleaned[col].astype("string").str.strip().str.lower()
        values = values.replace(yes_no)
        cleaned[col] = pd.to_numeric(values, errors="coerce")

    # Impossible measurements are treated as missing and imputed inside the
    # training pipeline; target values must be present and positive.
    cleaned.loc[~cleaned["age"].between(18, 100), "age"] = float("nan")
    cleaned.loc[~cleaned["height_cm"].between(100, 230), "height_cm"] = float("nan")
    cleaned.loc[~cleaned["weight_kg"].between(20, 300), "weight_kg"] = float("nan")
    cleaned.loc[~cleaned["major_surgeries"].between(0, 20), "major_surgeries"] = float("nan")
    for col in BINARY_COLUMNS:
        cleaned.loc[~cleaned[col].isin([0, 1]), col] = float("nan")

    before = len(cleaned)
    missing_targets = int(cleaned[TARGET_COLUMN].isna().sum())
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned = cleaned[cleaned[TARGET_COLUMN] > 0]
    after_target_filter = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    report = {
        "rows_before": int(before),
        "rows_after": int(len(cleaned)),
        "missing_or_nonpositive_targets_removed": int(before - after_target_filter),
        "duplicates_removed": int(after_target_filter - len(cleaned)),
        "missing_values_after_cleaning": {
            k: int(v) for k, v in cleaned.isna().sum().items()
        },
        "outliers": detect_outliers_iqr(cleaned),
    }
    return cleaned, report


def split_features_target(df: pd.DataFrame):
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def split_train_test(X, y, test_size=0.2, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessor(
    numerical_features=None, categorical_features=None, scale_numeric=True
):
    numerical_features = numerical_features or RAW_NUMERICAL_FEATURES
    categorical_features = categorical_features or RAW_CATEGORICAL_FEATURES
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numerical_features),
        ("cat", cat_pipe, categorical_features),
    ])
