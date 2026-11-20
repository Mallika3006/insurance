"""Input validation and schema-safe INR inference for the Indian dataset."""
from pathlib import Path
from typing import Any, Dict
import json

import joblib
import pandas as pd

from src import DEFAULT_MODEL_PATH, RAW_NUMERICAL_FEATURES, RAW_CATEGORICAL_FEATURES

BINARY_FIELDS = (
    "diabetes", "blood_pressure_problems", "any_transplants",
    "any_chronic_diseases", "known_allergies", "family_cancer_history",
)
EXPECTED_INPUT_COLUMNS = RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES
MODEL_SCHEMA_VERSION = 2


def validate_user_input(
    age, diabetes, blood_pressure_problems, any_transplants,
    any_chronic_diseases, height_cm, weight_kg, known_allergies,
    family_cancer_history, major_surgeries,
) -> Dict[str, Any]:
    try:
        values = {
            "age": int(age),
            "height_cm": float(height_cm),
            "weight_kg": float(weight_kg),
            "major_surgeries": int(major_surgeries),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("Age, height, weight and surgery count must be valid numbers.") from exc

    checks = [
        ("age", 18, 100, "Age"),
        ("height_cm", 120, 220, "Height"),
        ("weight_kg", 25, 250, "Weight"),
        ("major_surgeries", 0, 10, "Major surgeries"),
    ]
    for key, low, high, label in checks:
        if not low <= values[key] <= high:
            unit = " cm" if key == "height_cm" else " kg" if key == "weight_kg" else ""
            raise ValueError(f"{label} must be between {low} and {high}{unit}.")

    raw = dict(zip(
        BINARY_FIELDS,
        [diabetes, blood_pressure_problems, any_transplants,
         any_chronic_diseases, known_allergies, family_cancer_history],
    ))
    for field, value in raw.items():
        normalized = str(value).strip().lower()
        if normalized in ("yes", "1", "true"):
            values[field] = 1
        elif normalized in ("no", "0", "false"):
            values[field] = 0
        else:
            raise ValueError(f"{field.replace('_', ' ').title()} must be Yes or No.")

    return values


def load_trained_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            "Indian model not found. First place the correct Indian dataset at "
            "data/insurance.csv, then run: python -m src.model_training"
        )

    metadata_path = path.parent / "model_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(
            "The saved model is from an older project version or has no schema metadata. "
            "Delete the old models/best_model.joblib and models/model_metadata.json, "
            "then retrain with the Indian dataset: python -m src.model_training"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != MODEL_SCHEMA_VERSION
        or metadata.get("input_columns") != EXPECTED_INPUT_COLUMNS
    ):
        raise RuntimeError(
            "Model schema mismatch: the saved model was not trained with the current "
            "Indian dataset features. Retrain it using: python -m src.model_training"
        )

    model = joblib.load(path)
    if not hasattr(model, "predict") or not hasattr(model, "named_steps"):
        raise RuntimeError("Invalid model file: expected the complete sklearn prediction pipeline.")
    if "feature_engineering" not in model.named_steps or "preprocessor" not in model.named_steps:
        raise RuntimeError("Saved model is missing the expected Indian feature-engineering pipeline.")
    return model


def predict_insurance_charge(
    age, diabetes, blood_pressure_problems, any_transplants,
    any_chronic_diseases, height_cm, weight_kg, known_allergies,
    family_cancer_history, major_surgeries, model=None,
    model_path: Path | str = DEFAULT_MODEL_PATH,
):
    cleaned = validate_user_input(
        age, diabetes, blood_pressure_problems, any_transplants,
        any_chronic_diseases, height_cm, weight_kg, known_allergies,
        family_cancer_history, major_surgeries,
    )
    pipeline = model if model is not None else load_trained_model(model_path)
    input_df = pd.DataFrame([cleaned], columns=EXPECTED_INPUT_COLUMNS)
    estimate = max(0.0, float(pipeline.predict(input_df)[0]))
    return {
        "estimated_annual_premium_inr": round(estimate, 2),
        "validated_input": cleaned,
    }
