"""Model evaluation metrics, diagnostic plots, feature importance, and error analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src import EVAL_OUTPUT_DIR

sns.set_theme(style="whitegrid")


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate MAE, MSE, RMSE, and R² score for regression predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
    }


def plot_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path = EVAL_OUTPUT_DIR / "actual_vs_predicted.png",
) -> str:
    """Plot actual charges against predicted charges with an ideal 45-degree reference line."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.65, color="#2b6cb0", edgecolors="white", linewidth=0.4)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.6, label="Ideal Prediction (y = x)")
    ax.set_title(f"Actual vs. Predicted Charges — {model_name}")
    ax.set_xlabel("Actual Annual Charges ($)")
    ax.set_ylabel("Predicted Annual Charges ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path = EVAL_OUTPUT_DIR / "residual_plot.png",
) -> str:
    """Plot residuals (Actual - Predicted) vs. predicted values and residual distribution."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(y_pred, residuals, alpha=0.65, color="#3182ce", edgecolors="white", linewidth=0.4)
    axes[0].axhline(0.0, color="#e53e3e", linestyle="--", linewidth=1.4)
    axes[0].set_title(f"Residuals vs. Predicted — {model_name}")
    axes[0].set_xlabel("Predicted Charges ($)")
    axes[0].set_ylabel("Residual (Actual - Predicted) ($)")

    sns.histplot(residuals, kde=True, bins=30, color="#4299e1", ax=axes[1])
    axes[1].axvline(0.0, color="#e53e3e", linestyle="--", linewidth=1.4)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def extract_feature_importance(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Extract feature importance (tree models) or absolute standardized coefficients (linear models)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    else:
        raise ValueError(f"Model '{type(model).__name__}' does not expose feature importances or coefficients.")

    # Clean ColumnTransformer prefixes (e.g., 'num__', 'cat__') for readability
    clean_names = [name.split("__", 1)[-1] for name in feature_names]
    importance_df = pd.DataFrame({"feature": clean_names, "importance": importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False).head(top_n).reset_index(drop=True)
    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    save_path: Path = EVAL_OUTPUT_DIR / "feature_importance.png",
) -> str:
    """Plot horizontal bar chart of top feature importances."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        palette="viridis",
        ax=ax,
    )
    ax.set_title(f"Top Feature Importances — {model_name}")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path = EVAL_OUTPUT_DIR / "model_comparison.png",
) -> str:
    """Plot side-by-side RMSE and R² comparison across evaluated models."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    sns.barplot(
        data=comparison_df,
        x="Model",
        y="RMSE",
        hue="Model",
        legend=False,
        palette="Blues_d",
        ax=axes[0],
    )
    axes[0].set_title("Test RMSE by Model (Lower is Better)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RMSE ($)")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=comparison_df,
        x="Model",
        y="R2",
        hue="Model",
        legend=False,
        palette="Greens_d",
        ax=axes[1],
    )
    axes[1].set_title("Test R² Score by Model (Higher is Better)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R² Score")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def perform_error_analysis(
    X_test: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray
) -> Dict[str, object]:
    """Compute detailed error analysis metrics on the test set."""
    analysis_df = X_test.copy().reset_index(drop=True)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr
    abs_errors = np.abs(residuals)

    analysis_df["actual_charges"] = y_true_arr
    analysis_df["predicted_charges"] = np.round(y_pred_arr, 2)
    analysis_df["residual"] = np.round(residuals, 2)
    analysis_df["abs_error"] = np.round(abs_errors, 2)

    mae_by_smoker = {}
    if "smoker" in analysis_df.columns:
        mae_by_smoker = {
            str(k): round(float(v), 2)
            for k, v in analysis_df.groupby("smoker")["abs_error"].mean().items()
        }

    worst_cases = (
        analysis_df.sort_values(by="abs_error", ascending=False)
        .head(5)[["age", "sex", "bmi", "children", "smoker", "region", "actual_charges", "predicted_charges", "abs_error"]]
        .to_dict(orient="records")
    )

    return {
        "mean_residual": round(float(np.mean(residuals)), 2),
        "median_abs_error": round(float(np.median(abs_errors)), 2),
        "p90_abs_error": round(float(np.percentile(abs_errors, 90)), 2),
        "max_abs_error": round(float(np.max(abs_errors)), 2),
        "mae_by_smoker": mae_by_smoker,
        "top_5_largest_errors": worst_cases,
    }
