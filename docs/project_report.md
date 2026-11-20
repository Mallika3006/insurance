# Project Report: Health Insurance Premium Prediction Using Machine Learning

## 1. Abstract

Health insurance pricing depends heavily on individual demographic and lifestyle risk factors. This project develops an end-to-end supervised Machine Learning regression system that estimates annual health insurance charges using the publicly available Medical Cost Personal Dataset. Following a strict, leakage-free scikit-learn `Pipeline` workflow, the project compares four regression algorithms (Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor), evaluates domain-engineered features, tunes the best candidate via 5-fold `GridSearchCV`, persists the trained pipeline with Joblib, and deploys an interactive three-page Streamlit web application.

## 2. Introduction

Accurately estimating medical costs is an essential analytical problem in healthcare analytics and actuarial modeling. While real-world insurance underwriting incorporates clinical histories, plan deductibles, and regulatory rules, tabular regression models built on public demographic and lifestyle datasets provide a clear, transparent foundation for understanding how key risk factors drive healthcare expenditure. This project demonstrates a clean, modular Python architecture suitable for both learning and portfolio presentation.

## 3. Problem Statement

Given six input attributes describing an individual—`age`, `sex`, `bmi`, `children`, `smoker`, and `region`—build a regression model that accurately predicts the continuous annual medical cost (`charges`) billed by health insurance, and provide a user-friendly web interface for real-time educational estimation.

## 4. Objectives

1. Perform comprehensive Exploratory Data Analysis (EDA) to inspect target skewness, feature relationships, and group differences.
2. Clean the dataset systematically (missing value check, duplicate removal, and IQR outlier audit) without introducing data leakage.
3. Engineer domain-relevant features (`bmi_category`, `age_group`, `smoker_bmi_interaction`, `is_obese_smoker`) and quantitatively test whether they improve validation performance over raw features.
4. Train and benchmark four classical regression algorithms under a uniform 80/20 train-test split (`random_state=42`).
5. Optimize the best-performing candidate model using cross-validated hyperparameter tuning (`GridSearchCV`) on training data only.
6. Evaluate the final model on the untouched test set using MAE, MSE, RMSE, R², residual diagnostics, feature importance, and error analysis.
7. Package the trained preprocessing and prediction pipeline with Joblib and deploy it in a local Streamlit application.

## 5. Dataset Description

- **Name:** Medical Cost Personal Dataset
- **Source:** [Kaggle — mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **File Path:** `data/insurance.csv`
- **Features:**
  - `age` (numeric): Age of primary beneficiary
  - `sex` (categorical): `female` or `male`
  - `bmi` (numeric): Body Mass Index
  - `children` (numeric): Number of dependents covered
  - `smoker` (categorical): `yes` or `no`
  - `region` (categorical): `northeast`, `northwest`, `southeast`, `southwest`
- **Target:** `charges` (numeric, continuous annual medical costs in USD)

## 6. Methodology

1. **Data Loading & Inspection (`src/data_preprocessing.py`):** Validates CSV presence and required schema, inspects data types, missing counts, duplicates, and descriptive statistics.
2. **Data Cleaning:** Normalizes categorical strings, handles any missing values, removes exact duplicate records, and documents IQR-based outliers while retaining genuine high-cost medical records.
3. **Train-Test Split:** Splits data into 80% training and 20% testing (`random_state=42`) **prior** to any scaling, encoding, feature selection, or tuning to prevent data leakage.
4. **Feature Engineering (`src/feature_engineering.py`):** Implements `InsuranceFeatureEngineer` (a stateless scikit-learn `BaseEstimator`/`TransformerMixin`) so feature creation executes identically during cross-validation, testing, and live inference.
5. **Preprocessing Pipeline:** Combines `SimpleImputer`, `StandardScaler`, and `OneHotEncoder(handle_unknown="ignore", drop="first")` inside a `ColumnTransformer`.
6. **Model Training & Tuning (`src/model_training.py`):** Evaluates all four models via 5-fold cross-validation on training data (both raw and engineered feature sets), selects the best candidate using CV RMSE, tunes hyperparameters via `GridSearchCV` on training data only, and evaluates the final pipeline once on the untouched test set.
7. **Persistence & Verification:** Saves the full `Pipeline` to `models/best_model.joblib` and confirms identical predictions before saving and after reloading.

## 7. Exploratory Data Analysis (EDA) Findings

Running `python -m src.eda` (or `python -m src.model_training`) generates eight diagnostic plots in `outputs/eda/` and exact dataset statistics in `outputs/eda_summary.json`:

1. **Charges Distribution (`01_charges_distribution.png`):** Inspects the right-skewed distribution and upper tail of annual medical charges.
2. **Age vs. Charges (`02_age_vs_charges.png`):** Visualizes how baseline medical charges rise steadily with age and separate into distinct cost bands driven by smoking status.
3. **BMI vs. Charges (`03_bmi_vs_charges.png`):** Highlights the non-linear jump in medical charges when smoking coincides with obesity (`BMI >= 30`), directly motivating the `smoker_bmi_interaction` and `is_obese_smoker` features.
4. **Smoking Status vs. Charges (`04_smoker_vs_charges.png`):** Compares cost distributions and averages between smokers and non-smokers.
5. **Gender vs. Charges (`05_gender_vs_charges.png`):** Compares male and female charge distributions to check whether gender introduces major cost separation.
6. **Region vs. Charges (`06_region_vs_charges.png`):** Evaluates regional cost differences across the four US regions.
7. **Children vs. Charges (`07_children_vs_charges.png`):** Examines how dependent count relates to average and spread of charges.
8. **Correlation Heatmap (`08_correlation_heatmap.png`):** Quantifies linear correlations among numerical variables, smoking status, and charges.

## 8. Algorithms Used

- **Linear Regression:** Models charges as a weighted linear sum of features; fast and interpretable, though limited on non-linear interactions unless engineered terms are supplied.
- **Decision Tree Regressor:** Splits data into axis-aligned regions to minimize MSE; captures non-linear thresholds naturally but can overfit if unpruned.
- **Random Forest Regressor:** Averages predictions from 200 bootstrap-sampled decision trees, substantially lowering variance compared to a single tree.
- **Gradient Boosting Regressor:** Sequentially fits shallow trees to negative gradients (residuals), typically achieving strong accuracy on structured tabular regression datasets.

## 9. Evaluation Metrics

- **MAE (Mean Absolute Error):** Average absolute dollar difference between predicted and actual charges; directly interpretable in USD.
- **MSE (Mean Squared Error):** Average squared prediction error; penalizes large outliers heavily.
- **RMSE (Root Mean Squared Error):** Square root of MSE, expressed in USD; primary selection metric during cross-validation.
- **R² Score (Coefficient of Determination):** Proportion of variance in annual charges explained by the model (1.0 indicates perfect prediction).

## 10. Results

> **Execution Integrity Notice:** No results are fabricated in this report. Because `data/insurance.csv` must be downloaded directly from Kaggle by the user, actual numeric metrics are generated dynamically when `python -m src.model_training` is executed locally.

After running training, review the generated result files:
- **Model Comparison Table:** `outputs/model_comparison.csv`
- **Raw vs. Engineered Feature Comparison:** `outputs/feature_engineering_comparison.csv`
- **Before vs. After Tuning Metrics:** `outputs/tuning_comparison.json`
- **Feature Importances:** `outputs/feature_importance.csv` and `outputs/evaluation/feature_importance.png`
- **Diagnostic Plots:** `outputs/evaluation/actual_vs_predicted.png` and `outputs/evaluation/residual_plot.png`
- **Error Analysis:** `outputs/error_analysis.json`

## 11. Conclusion

This project demonstrates a disciplined, end-to-end Machine Learning workflow for health insurance charge estimation. By combining domain-informed feature engineering, leakage-free scikit-learn pipelines, systematic multi-model benchmarking, cross-validated hyperparameter tuning, and a cached Streamlit web application, the system delivers clean engineering structure and transparent evaluation without unnecessary framework complexity.

## 12. Future Scope

- Incorporate log-transformed target modeling or quantile regression for prediction intervals.
- Add SHAP explainability for per-prediction local feature attributions in the Streamlit UI.
- Expand the dataset with richer clinical and lifestyle variables where legally permitted.
- Add automated `pytest` suites and containerized deployment for cloud hosting.
