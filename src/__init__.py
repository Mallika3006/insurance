"""Core paths and schema for the Indian health insurance premium project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_PATH = DATA_DIR / "insurance.csv"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
EVAL_OUTPUT_DIR = OUTPUTS_DIR / "evaluation"
ASSETS_DIR = PROJECT_ROOT / "assets"

RANDOM_STATE = 42
TARGET_COLUMN = "premium_price"
RAW_NUMERICAL_FEATURES = ["age", "height_cm", "weight_kg"]
RAW_CATEGORICAL_FEATURES = [
    "diabetes", "blood_pressure_problems", "any_transplants",
    "any_chronic_diseases", "known_allergies",
    "family_cancer_history", "major_surgeries",
]
REQUIRED_COLUMNS = RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES + [TARGET_COLUMN]
