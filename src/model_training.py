"""End-to-end model training, feature engineering comparison, hyperparameter tuning, and saving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    EVAL_OUTPUT_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
    RAW_NUMERICAL_FEATURES,
    RAW_CATEGORICAL_FEATURES,
)
from src.data_preprocessing import (
    build_preprocessor,
    clean_dataset,
    load_dataset,
    split_features_target,
    split_train_test,
)
from src.eda import run_eda
from src.feature_engineering import InsuranceFeatureEngineer, get_feature_lists
from src.model_evaluation import (
    compute_regression_metrics,
    extract_feature_importance,
    perform_error_analysis,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
)


def get_candidate_models(random_state: int = RANDOM_STATE) -> Dict[str, object]:
    """Return the four required regression models configured with a fixed random_state."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=random_state
        ),
    }


def get_param_grid(model_name: str) -> Dict[str, list]:
    """Return a clean, focused GridSearchCV hyperparameter grid for the selected model."""
    grids: Dict[str, Dict[str, list]] = {
        "Gradient Boosting Regressor": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
        },
        "Random Forest Regressor": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 6, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Decision Tree Regressor": {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
        },
        "Linear Regression": {
            "model__fit_intercept": [True, False],
        },
    }
    return grids.get(model_name, {})


def build_model_pipeline(
    model: object, use_engineered_features: bool = True
) -> Pipeline:
    """Assemble an end-to-end Pipeline: feature engineering -> preprocessing -> regressor."""
    num_features, cat_features = get_feature_lists(use_engineered_features)
    preprocessor = build_preprocessor(
        numerical_features=num_features,
        categorical_features=cat_features,
        scale_numeric=True,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", InsuranceFeatureEngineer(enabled=use_engineered_features)),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_models_for_feature_set(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    use_engineered_features: bool,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Train and evaluate all four candidate models for a given feature set."""
    rows = []
    fitted_pipelines: Dict[str, Pipeline] = {}
    feature_label = "With Engineered Features" if use_engineered_features else "Raw Features Only"

    for name, estimator in get_candidate_models(RANDOM_STATE).items():
        pipeline = build_model_pipeline(estimator, use_engineered_features=use_engineered_features)

        # 5-fold cross-validation strictly on training data
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        cv_rmse = round(float(-cv_scores.mean()), 2)

        # Fit on training data and evaluate on untouched test data
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = compute_regression_metrics(y_test, y_pred)

        rows.append(
            {
                "Feature Set": feature_label,
                "Model": name,
                "CV_RMSE": cv_rmse,
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )
        fitted_pipelines[name] = pipeline

    results_df = pd.DataFrame(rows).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    return results_df, fitted_pipelines


def train_and_save_pipeline(
    data_path: Path | str = DEFAULT_DATA_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    run_eda_plots: bool = True,
) -> Dict[str, object]:
    """Execute the full ML workflow: load, clean, EDA, split, compare, tune, evaluate, and save."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 & 3: Load and clean dataset
    raw_df = load_dataset(data_path)
    cleaned_df, cleaning_report = clean_dataset(raw_df)

    # Step 2: Run EDA visualizations and summary statistics
    if run_eda_plots:
        run_eda(cleaned_df)

    # Step 4 & 6: Separate features/target and perform 80/20 split before any fitting
    X, y = split_features_target(cleaned_df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Step 5 & 7: Compare models with raw features vs. engineered features
    raw_results_df, raw_pipelines = evaluate_models_for_feature_set(
        X_train, X_test, y_train, y_test, use_engineered_features=False
    )
    eng_results_df, eng_pipelines = evaluate_models_for_feature_set(
        X_train, X_test, y_train, y_test, use_engineered_features=True
    )

    fe_comparison_df = pd.concat([raw_results_df, eng_results_df], ignore_index=True)
    fe_comparison_df.to_csv(OUTPUTS_DIR / "feature_engineering_comparison.csv", index=False)

    # Select whether engineered features improve best CV RMSE (no test leakage in selection)
    best_raw_cv = float(raw_results_df["CV_RMSE"].min())
    best_eng_cv = float(eng_results_df["CV_RMSE"].min())
    use_engineered = best_eng_cv <= best_raw_cv

    selected_comparison_df = eng_results_df if use_engineered else raw_results_df
    selected_pipelines = eng_pipelines if use_engineered else raw_pipelines
    selected_comparison_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    # Select best candidate model strictly using training CV RMSE
    best_candidate_row = selected_comparison_df.sort_values(by="CV_RMSE", ascending=True).iloc[0]
    best_model_name = str(best_candidate_row["Model"])
    untuned_pipeline = selected_pipelines[best_model_name]
    untuned_test_metrics = compute_regression_metrics(y_test, untuned_pipeline.predict(X_test))

    # Step 9: Hyperparameter tuning using GridSearchCV on training data only
    base_estimator = get_candidate_models(RANDOM_STATE)[best_model_name]
    tuning_pipeline = build_model_pipeline(base_estimator, use_engineered_features=use_engineered)
    param_grid = get_param_grid(best_model_name)

    grid_search = GridSearchCV(
        estimator=tuning_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    tuned_pipeline = grid_search.best_estimator_
    tuned_cv_rmse = round(float(-grid_search.best_score_), 2)

    # Evaluate final tuned model on untouched test set
    y_test_pred = tuned_pipeline.predict(X_test)
    tuned_test_metrics = compute_regression_metrics(y_test, y_test_pred)

    # Keep tuned pipeline if CV improved (or equals); otherwise keep untuned
    final_pipeline = tuned_pipeline if tuned_cv_rmse <= float(best_candidate_row["CV_RMSE"]) else untuned_pipeline
    final_pred = final_pipeline.predict(X_test)
    final_metrics = compute_regression_metrics(y_test, final_pred)

    # Step 8: Generate evaluation plots, feature importance, and error analysis
    plot_model_comparison(selected_comparison_df, EVAL_OUTPUT_DIR / "model_comparison.png")
    plot_actual_vs_predicted(y_test, final_pred, f"{best_model_name} (Tuned)")
    plot_residuals(y_test, final_pred, f"{best_model_name} (Tuned)")

    importance_df = extract_feature_importance(final_pipeline, top_n=15)
    importance_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(importance_df, f"{best_model_name} (Tuned)")

    error_report = perform_error_analysis(X_test, y_test, final_pred)
    with open(OUTPUTS_DIR / "error_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(error_report, fh, indent=2)

    tuning_comparison = {
        "best_model_name": best_model_name,
        "use_engineered_features": use_engineered,
        "best_params": grid_search.best_params_,
        "before_tuning": {
            "CV_RMSE": float(best_candidate_row["CV_RMSE"]),
            **untuned_test_metrics,
        },
        "after_tuning": {
            "CV_RMSE": tuned_cv_rmse,
            **tuned_test_metrics,
        },
        "final_selected_metrics": final_metrics,
    }
    with open(OUTPUTS_DIR / "tuning_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(tuning_comparison, fh, indent=2)

    # Step 10: Save complete preprocessing + prediction pipeline and verify reload
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)

    reloaded_pipeline = joblib.load(model_path)
    sample_row = X_test.iloc[[0]]
    pred_before = float(final_pipeline.predict(sample_row)[0])
    pred_after = float(reloaded_pipeline.predict(sample_row)[0])
    if not np.isclose(pred_before, pred_after, rtol=1e-6):
        raise RuntimeError("Reloaded model prediction does not match original model prediction.")

    metadata = {
        "schema_version": 2,
        "input_columns": RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES,
        "target_column": "premium_price",
        "currency": "INR",
        "best_model_name": best_model_name,
        "use_engineered_features": use_engineered,
        "random_state": RANDOM_STATE,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "cleaning_report": cleaning_report,
        "final_test_metrics": final_metrics,
        "reload_verification": {
            "prediction_before_save": round(pred_before, 2),
            "prediction_after_reload": round(pred_after, 2),
            "verified": True,
        },
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return {
        "model_path": str(model_path),
        "best_model_name": best_model_name,
        "use_engineered_features": use_engineered,
        "final_metrics": final_metrics,
        "tuning_comparison": tuning_comparison,
    }


if __name__ == "__main__":
    summary = train_and_save_pipeline()
    print("\nTraining complete!")
    print(f"Best Model: {summary['best_model_name']}")
    print(f"Engineered Features Used: {summary['use_engineered_features']}")
    print(f"Final Test Metrics: {summary['final_metrics']}")
    print(f"Saved Pipeline: {summary['model_path']}")
