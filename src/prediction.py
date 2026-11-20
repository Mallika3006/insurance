"""Input validation, saved model loading, and inference helper for predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src import DEFAULT_MODEL_PATH

VALID_SEX = ("female", "male")
VALID_SMOKER = ("no", "yes")
VALID_REGIONS = ("northeast", "northwest", "southeast", "southwest")


def validate_user_input(
    age: int | float,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
) -> Dict[str, Any]:
    """Validate and normalize user inputs before inference."""
    try:
        age_val = int(age)
    except (TypeError, ValueError) as exc:
        raise ValueError("Age must be a valid integer.") from exc
    if not (18 <= age_val <= 100):
        raise ValueError("Age must be between 18 and 100 years.")

    try:
        bmi_val = float(bmi)
    except (TypeError, ValueError) as exc:
        raise ValueError("BMI must be a valid number.") from exc
    if not (10.0 <= bmi_val <= 70.0):
        raise ValueError("BMI must be between 10.0 and 70.0.")

    try:
        children_val = int(children)
    except (TypeError, ValueError) as exc:
        raise ValueError("Number of children must be a valid integer.") from exc
    if not (0 <= children_val <= 10):
        raise ValueError("Number of children must be between 0 and 10.")

    sex_val = str(sex).strip().lower()
    if sex_val not in VALID_SEX:
        raise ValueError(f"Gender must be one of {VALID_SEX}.")

    smoker_val = str(smoker).strip().lower()
    if smoker_val not in VALID_SMOKER:
        raise ValueError(f"Smoking status must be one of {VALID_SMOKER}.")

    region_val = str(region).strip().lower()
    if region_val not in VALID_REGIONS:
        raise ValueError(f"Region must be one of {VALID_REGIONS}.")

    return {
        "age": age_val,
        "sex": sex_val,
        "bmi": round(bmi_val, 2),
        "children": children_val,
        "smoker": smoker_val,
        "region": region_val,
    }


def load_trained_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    """Load the saved joblib preprocessing + regression pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{path}'.\n"
            "Run 'python -m src.model_training' after placing 'insurance.csv' in the 'data/' folder."
        )
    return joblib.load(path)


def predict_insurance_charge(
    age: int | float,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
    model=None,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """Validate inputs, run inference with the saved pipeline, and return the estimated charge."""
    cleaned_input = validate_user_input(age, sex, bmi, children, smoker, region)
    pipeline = model if model is not None else load_trained_model(model_path)

    input_df = pd.DataFrame([cleaned_input])
    raw_prediction = float(pipeline.predict(input_df)[0])
    estimated_charge = round(max(0.0, raw_prediction), 2)

    return {
        "estimated_annual_charges": estimated_charge,
        "validated_input": cleaned_input,
    }
