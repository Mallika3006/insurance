"""Streamlit web application for Health Insurance Premium Prediction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import DEFAULT_MODEL_PATH, EDA_OUTPUT_DIR, EVAL_OUTPUT_DIR, MODELS_DIR, OUTPUTS_DIR
from src.prediction import (
    VALID_REGIONS,
    VALID_SEX,
    VALID_SMOKER,
    load_trained_model,
    predict_insurance_charge,
)

st.set_page_config(
    page_title="Health Insurance Premium Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_model():
    """Load the saved preprocessing and regression pipeline once per session."""
    return load_trained_model(DEFAULT_MODEL_PATH)


@st.cache_data
def load_json_if_exists(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


@st.cache_data
def load_csv_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def render_home_page() -> None:
    """Page 1: Project overview, workflow explanation, and technology stack."""
    st.title("Health Insurance Premium Prediction Using Machine Learning")
    st.markdown(
        "An educational Machine Learning regression application that estimates an individual's "
        "approximate annual health insurance charges from personal, demographic, and lifestyle inputs."
    )

    st.info(
        "**Disclaimer:** This application is an educational Machine Learning project built on the "
        "public Medical Cost Personal Dataset. Predictions are statistical estimates and **not** "
        "official insurance quotes or underwriting decisions."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("How the Model Works")
        st.markdown(
            """
            1. **Data Cleaning & Validation:** Standardizes categorical values, checks missing values,
               removes exact duplicate rows, and audits outliers.
            2. **Feature Engineering:** Adds clinical BMI categories, age groups, and a smoking-BMI
               interaction term inside a scikit-learn `Pipeline`.
            3. **Leakage-Free Preprocessing:** Fits `ColumnTransformer` (scaling numerical features and
               one-hot encoding categorical features) strictly on the 80% training split.
            4. **Model Selection & Tuning:** Compares Linear Regression, Decision Tree, Random Forest,
               and Gradient Boosting using 5-fold cross-validation, then tunes the top candidate with
               `GridSearchCV`.
            5. **Saved Pipeline Inference:** Loads the serialized `joblib` pipeline directly so
               predictions run instantly without retraining.
            """
        )

    with col2:
        st.subheader("Technologies Used")
        st.markdown(
            """
            - **Language:** Python
            - **Data Analysis:** Pandas, NumPy
            - **Visualization:** Matplotlib, Seaborn
            - **Machine Learning:** Scikit-learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`)
            - **Model Persistence:** Joblib
            - **Web Interface:** Streamlit
            - **Interactive Analysis:** Jupyter Notebook
            """
        )

    metadata = load_json_if_exists(MODELS_DIR / "model_metadata.json")
    if metadata:
        st.subheader("Active Model Status")
        m1, m2, m3, m4 = st.columns(4)
        metrics = metadata.get("final_test_metrics", {})
        m1.metric("Selected Model", metadata.get("best_model_name", "N/A"))
        m2.metric("Test R² Score", f"{metrics.get('R2', 0):.4f}")
        m3.metric("Test RMSE", f"${metrics.get('RMSE', 0):,.2f}")
        m4.metric("Test MAE", f"${metrics.get('MAE', 0):,.2f}")
    else:
        st.warning(
            "No trained model found yet. Place `insurance.csv` in the `data/` folder and run "
            "`python -m src.model_training` to train and save the model."
        )


def render_prediction_page() -> None:
    """Page 2: Interactive form to estimate annual health insurance charges."""
    st.title("Insurance Premium Prediction")
    st.markdown(
        "Enter the individual's details below and click **Predict Estimated Charges** to view the "
        "Machine Learning estimate."
    )

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            "Trained model file (`models/best_model.joblib`) was not found. "
            "Please download `insurance.csv` from Kaggle into `data/insurance.csv` and run:\n\n"
            "```bash\npython -m src.model_training\n```"
        )
        return

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
                help="Enter an adult age between 18 and 100.",
            )
            sex = st.selectbox(
                "Gender",
                options=list(VALID_SEX),
                format_func=lambda x: x.capitalize(),
            )
            bmi = st.number_input(
                "BMI (Body Mass Index)",
                min_value=10.0,
                max_value=70.0,
                value=26.5,
                step=0.1,
                format="%.1f",
                help="Typical adult BMI ranges from 18.5 to 40.0.",
            )

        with col2:
            children = st.number_input(
                "Number of Children / Dependents",
                min_value=0,
                max_value=10,
                value=1,
                step=1,
            )
            smoker = st.selectbox(
                "Smoking Status",
                options=list(VALID_SMOKER),
                format_func=lambda x: "Yes (Smoker)" if x == "yes" else "No (Non-Smoker)",
            )
            region = st.selectbox(
                "Residential Region",
                options=list(VALID_REGIONS),
                format_func=lambda x: x.capitalize(),
            )

        submitted = st.form_submit_button("Predict Estimated Charges", use_container_width=True)

    if submitted:
        try:
            model = get_model()
            result = predict_insurance_charge(
                age=age,
                sex=sex,
                bmi=bmi,
                children=children,
                smoker=smoker,
                region=region,
                model=model,
            )
            estimate = result["estimated_annual_charges"]
            validated = result["validated_input"]

            st.success(f"### Estimated Annual Insurance Charges: **${estimate:,.2f}**")
            st.caption(
                "Note: This figure is a Machine Learning estimate generated from historical dataset "
                "patterns. It is intended for educational purposes only and is not an actual "
                "insurance quote."
            )

            st.subheader("Entered Information Summary")
            summary_df = pd.DataFrame(
                [
                    {
                        "Age": validated["age"],
                        "Gender": validated["sex"].capitalize(),
                        "BMI": validated["bmi"],
                        "Children": validated["children"],
                        "Smoker": validated["smoker"].capitalize(),
                        "Region": validated["region"].capitalize(),
                    }
                ]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        except ValueError as err:
            st.error(f"Invalid input: {err}")
        except FileNotFoundError as err:
            st.error(str(err))
        except Exception as err:
            st.error(f"An unexpected error occurred during prediction: {err}")


def render_insights_page() -> None:
    """Page 3: Saved model comparison, evaluation metrics, feature importance, and EDA plots."""
    st.title("Model Insights & Exploratory Data Analysis")
    st.markdown(
        "This page loads saved evaluation results, feature importances, and EDA plots generated "
        "during model training (no retraining occurs when the app starts)."
    )

    comparison_df = load_csv_if_exists(OUTPUTS_DIR / "model_comparison.csv")
    fe_comparison_df = load_csv_if_exists(OUTPUTS_DIR / "feature_engineering_comparison.csv")
    tuning_data = load_json_if_exists(OUTPUTS_DIR / "tuning_comparison.json")
    importance_df = load_csv_if_exists(OUTPUTS_DIR / "feature_importance.csv")
    error_data = load_json_if_exists(OUTPUTS_DIR / "error_analysis.json")

    if comparison_df is None:
        st.warning(
            "Saved model results were not found in `outputs/`. Please download `insurance.csv` "
            "into `data/` and run `python -m src.model_training` to generate all metrics and plots."
        )
        return

    st.subheader("Model Comparison Results (Test Set)")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    if tuning_data:
        st.subheader("Hyperparameter Tuning Summary")
        before = tuning_data.get("before_tuning", {})
        after = tuning_data.get("after_tuning", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", tuning_data.get("best_model_name", "N/A"))
        col2.metric(
            "Test RMSE (After Tuning)",
            f"${after.get('RMSE', 0):,.2f}",
            delta=f"{after.get('RMSE', 0) - before.get('RMSE', 0):+.2f}",
            delta_color="inverse",
        )
        col3.metric(
            "Test MAE (After Tuning)",
            f"${after.get('MAE', 0):,.2f}",
            delta=f"{after.get('MAE', 0) - before.get('MAE', 0):+.2f}",
            delta_color="inverse",
        )
        col4.metric(
            "Test R² (After Tuning)",
            f"{after.get('R2', 0):.4f}",
            delta=f"{after.get('R2', 0) - before.get('R2', 0):+.4f}",
        )
        with st.expander("View Best Hyperparameters & Feature Engineering Comparison"):
            st.json(tuning_data.get("best_params", {}))
            if fe_comparison_df is not None:
                st.dataframe(fe_comparison_df, use_container_width=True, hide_index=True)

    st.subheader("Evaluation Diagnostics")
    eval_images = [
        ("Model Comparison", EVAL_OUTPUT_DIR / "model_comparison.png"),
        ("Actual vs. Predicted", EVAL_OUTPUT_DIR / "actual_vs_predicted.png"),
        ("Residual Diagnostics", EVAL_OUTPUT_DIR / "residual_plot.png"),
        ("Feature Importance", EVAL_OUTPUT_DIR / "feature_importance.png"),
    ]
    cols = st.columns(2)
    for idx, (caption, img_path) in enumerate(eval_images):
        if img_path.exists():
            cols[idx % 2].image(str(img_path), caption=caption, use_container_width=True)

    if importance_df is not None:
        with st.expander("Feature Importance Table"):
            st.dataframe(importance_df, use_container_width=True, hide_index=True)

    if error_data:
        with st.expander("Error Analysis Summary"):
            e1, e2, e3 = st.columns(3)
            e1.metric("Median Absolute Error", f"${error_data.get('median_abs_error', 0):,.2f}")
            e2.metric("90th Percentile Abs Error", f"${error_data.get('p90_abs_error', 0):,.2f}")
            e3.metric("Max Absolute Error", f"${error_data.get('max_abs_error', 0):,.2f}")
            if error_data.get("mae_by_smoker"):
                st.write("**MAE by Smoking Status:**", error_data["mae_by_smoker"])

    st.subheader("Exploratory Data Analysis Visualizations")
    eda_files = sorted(EDA_OUTPUT_DIR.glob("*.png")) if EDA_OUTPUT_DIR.exists() else []
    if eda_files:
        eda_cols = st.columns(2)
        for idx, img_path in enumerate(eda_files):
            title = img_path.stem.split("_", 1)[-1].replace("_", " ").title()
            eda_cols[idx % 2].image(str(img_path), caption=title, use_container_width=True)
    else:
        st.info("EDA plots will appear here once `python -m src.model_training` or `python -m src.eda` is run.")


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Home", "Insurance Premium Prediction", "Model Insights"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Health Insurance Premium Prediction • Educational ML Project")

    if page == "Home":
        render_home_page()
    elif page == "Insurance Premium Prediction":
        render_prediction_page()
    else:
        render_insights_page()


if __name__ == "__main__":
    main()
