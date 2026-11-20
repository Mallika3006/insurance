"""Data loading, inspection, cleaning, and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import DEFAULT_DATA_PATH, RANDOM_STATE, REQUIRED_COLUMNS, TARGET_COLUMN

RAW_NUMERICAL_FEATURES: List[str] = ["age", "bmi", "children"]
RAW_CATEGORICAL_FEATURES: List[str] = ["sex", "smoker", "region"]


def load_dataset(filepath: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the Medical Cost Personal Dataset CSV and validate its schema."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please download 'insurance.csv' from Kaggle "
            "(https://www.kaggle.com/datasets/mirichoi0218/insurance) "
            "and place it in the 'data/' directory. See data/README.md for details."
        )

    df = pd.read_csv(path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Expected columns: {REQUIRED_COLUMNS}."
        )
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """Return a structured summary of dataset shape, types, missing values, and duplicates."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "head": df.head(5),
        "tail": df.tail(5),
        "summary_stats": df.describe(include="all"),
    }


def detect_outliers_iqr(df: pd.DataFrame, columns: List[str] | None = None) -> Dict[str, Dict[str, float]]:
    """Identify potential outliers in numerical columns using the 1.5 * IQR rule."""
    if columns is None:
        columns = ["age", "bmi", "children", "charges"]

    outlier_report: Dict[str, Dict[str, float]] = {}
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outlier_report[col] = {
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": count,
            "outlier_pct": round((count / len(df)) * 100, 2) if len(df) else 0.0,
        }
    return outlier_report


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Clean the dataset: normalize text, remove exact duplicates, handle missing values, and audit outliers.

    Note on outliers: High insurance charges and high BMI values in medical data
    represent real high-risk individuals (especially smokers) rather than sensor errors,
    so outlier records are retained and documented rather than blindly dropped.
    """
    cleaned = df.copy()
    initial_rows = len(cleaned)

    # Normalize string columns to lowercase and strip whitespace for consistent encoding
    for col in RAW_CATEGORICAL_FEATURES:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip().str.lower()

    # Check and drop rows where target variable is missing
    missing_before = cleaned.isnull().sum().to_dict()
    target_missing = int(cleaned[TARGET_COLUMN].isnull().sum())
    if target_missing > 0:
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN])

    # Handle missing feature values if any are present
    for col in RAW_NUMERICAL_FEATURES:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    for col in RAW_CATEGORICAL_FEATURES:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0])

    # Remove exact duplicate records
    duplicates_removed = int(cleaned.duplicated().sum())
    if duplicates_removed > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    outlier_report = detect_outliers_iqr(cleaned)

    cleaning_report: Dict[str, object] = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "missing_values_before": {k: int(v) for k, v in missing_before.items()},
        "target_missing_dropped": target_missing,
        "duplicates_removed": duplicates_removed,
        "outliers_retained_iqr": outlier_report,
    }
    return cleaned, cleaning_report


def split_features_target(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate input features (X) from the target variable (y)."""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into 80% training and 20% testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create a scikit-learn ColumnTransformer for numerical and categorical features.

    Fitting this transformer inside a Pipeline strictly on training data prevents
    data leakage between train and test splits.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor
