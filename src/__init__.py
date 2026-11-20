"""Core source package for Health Insurance Premium Prediction."""

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
TARGET_COLUMN = "charges"
REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
