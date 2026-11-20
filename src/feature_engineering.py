"""Indian health-profile feature engineering, kept inside the sklearn Pipeline."""
from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src import RAW_NUMERICAL_FEATURES, RAW_CATEGORICAL_FEATURES

ENGINEERED_NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ["bmi"]
ENGINEERED_CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES + ["bmi_category", "surgery_history"]

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    height_m = pd.to_numeric(out["height_cm"], errors="coerce") / 100.0
    out["bmi"] = pd.to_numeric(out["weight_kg"], errors="coerce") / height_m.pow(2)
    out["bmi_category"] = pd.cut(out["bmi"], [-float("inf"), 18.5, 25, 30, float("inf")],
                                 labels=["underweight", "normal", "overweight", "obese"]).astype(object)
    out["surgery_history"] = out["major_surgeries"].apply(
        lambda n: "unknown" if pd.isna(n) else ("none" if n == 0 else ("one" if n == 1 else "multiple"))
    )
    return out

class InsuranceFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("InsuranceFeatureEngineer expects a pandas DataFrame.")
        return add_engineered_features(X) if self.enabled else X.copy()

def get_feature_lists(use_engineered_features=True):
    if use_engineered_features:
        return list(ENGINEERED_NUMERICAL_FEATURES), list(ENGINEERED_CATEGORICAL_FEATURES)
    return list(RAW_NUMERICAL_FEATURES), list(RAW_CATEGORICAL_FEATURES)
