"""Feature engineering transformer and helpers for Health Insurance Premium Prediction."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_preprocessing import RAW_CATEGORICAL_FEATURES, RAW_NUMERICAL_FEATURES

ENGINEERED_NUMERICAL_FEATURES: List[str] = RAW_NUMERICAL_FEATURES + [
    "smoker_bmi_interaction",
    "is_obese_smoker",
]

ENGINEERED_CATEGORICAL_FEATURES: List[str] = RAW_CATEGORICAL_FEATURES + [
    "bmi_category",
    "age_group",
]


def categorize_bmi(bmi_series: pd.Series) -> pd.Series:
    """Map continuous BMI values to standard clinical weight categories."""
    bins = [-np.inf, 18.5, 25.0, 30.0, np.inf]
    labels = ["underweight", "normal", "overweight", "obese"]
    return pd.cut(bmi_series, bins=bins, labels=labels, right=False).astype(str)


def categorize_age(age_series: pd.Series) -> pd.Series:
    """Map continuous age values to interpretable life-stage age groups."""
    bins = [-np.inf, 30, 45, 60, np.inf]
    labels = ["young_adult", "adult", "middle_aged", "senior"]
    return pd.cut(age_series, bins=bins, labels=labels, right=False).astype(str)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed engineered features to a DataFrame copy.

    Engineered features:
    - bmi_category: Clinical BMI brackets (underweight, normal, overweight, obese).
    - age_group: Life-stage age brackets (young_adult, adult, middle_aged, senior).
    - smoker_bmi_interaction: Continuous interaction term (bmi * is_smoker) capturing
      the steep cost acceleration for smokers with higher BMI.
    - is_obese_smoker: Binary indicator (1 if smoker == 'yes' and bmi >= 30 else 0).
    """
    engineered = df.copy()
    smoker_flag = (
        engineered["smoker"].astype(str).str.strip().str.lower() == "yes"
    ).astype(int)

    engineered["bmi_category"] = categorize_bmi(engineered["bmi"])
    engineered["age_group"] = categorize_age(engineered["age"])
    engineered["smoker_bmi_interaction"] = engineered["bmi"] * smoker_flag
    engineered["is_obese_smoker"] = (
        (smoker_flag == 1) & (engineered["bmi"] >= 30.0)
    ).astype(int)
    return engineered


class InsuranceFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for creating health insurance domain features.

    Embedding this transformer inside a scikit-learn Pipeline ensures identical feature
    generation during cross-validation, testing, and live Streamlit inference.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "InsuranceFeatureEngineer":
        """Stateless transformer; no parameters are learned from data."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame by optionally appending engineered features."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("InsuranceFeatureEngineer expects a pandas DataFrame as input.")
        if not self.enabled:
            return X.copy()
        return add_engineered_features(X)


def get_feature_lists(use_engineered_features: bool = True) -> Tuple[List[str], List[str]]:
    """Return (numerical_features, categorical_features) matching the selected feature mode."""
    if use_engineered_features:
        return list(ENGINEERED_NUMERICAL_FEATURES), list(ENGINEERED_CATEGORICAL_FEATURES)
    return list(RAW_NUMERICAL_FEATURES), list(RAW_CATEGORICAL_FEATURES)
