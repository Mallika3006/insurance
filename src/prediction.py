"""Input validation and INR-denominated inference for the Indian dataset."""
from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd
from src import DEFAULT_MODEL_PATH

BINARY_FIELDS = ("diabetes", "blood_pressure_problems", "any_transplants",
                 "any_chronic_diseases", "known_allergies", "family_cancer_history")

def validate_user_input(age, diabetes, blood_pressure_problems, any_transplants,
                        any_chronic_diseases, height_cm, weight_kg, known_allergies,
                        family_cancer_history, major_surgeries) -> Dict[str, Any]:
    try:
        values = {"age": int(age), "height_cm": float(height_cm), "weight_kg": float(weight_kg),
                  "major_surgeries": int(major_surgeries)}
    except (TypeError, ValueError) as exc:
        raise ValueError("Age, height, weight and surgery count must be valid numbers.") from exc
    if not 18 <= values["age"] <= 100: raise ValueError("Age must be between 18 and 100.")
    if not 120 <= values["height_cm"] <= 220: raise ValueError("Height must be between 120 and 220 cm.")
    if not 25 <= values["weight_kg"] <= 250: raise ValueError("Weight must be between 25 and 250 kg.")
    if not 0 <= values["major_surgeries"] <= 10: raise ValueError("Major surgeries must be between 0 and 10.")
    raw = dict(zip(BINARY_FIELDS, [diabetes, blood_pressure_problems, any_transplants,
                                  any_chronic_diseases, known_allergies, family_cancer_history]))
    for field, value in raw.items():
        if str(value).strip().lower() in ("yes", "1", "true"): values[field] = 1
        elif str(value).strip().lower() in ("no", "0", "false"): values[field] = 0
        else: raise ValueError(f"{field.replace('_', ' ').title()} must be Yes or No.")
    return values

def load_trained_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError("Trained model not found. Run: python -m src.model_training")
    return joblib.load(path)

def predict_insurance_charge(age, diabetes, blood_pressure_problems, any_transplants,
                             any_chronic_diseases, height_cm, weight_kg, known_allergies,
                             family_cancer_history, major_surgeries, model=None,
                             model_path: Path | str = DEFAULT_MODEL_PATH):
    cleaned = validate_user_input(age, diabetes, blood_pressure_problems, any_transplants,
                                  any_chronic_diseases, height_cm, weight_kg, known_allergies,
                                  family_cancer_history, major_surgeries)
    pipeline = model if model is not None else load_trained_model(model_path)
    estimate = max(0.0, float(pipeline.predict(pd.DataFrame([cleaned]))[0]))
    return {"estimated_annual_premium_inr": round(estimate, 2), "validated_input": cleaned}
