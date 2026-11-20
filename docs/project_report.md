# Project Report: Indian Health Insurance Premium Prediction

## Abstract
This project demonstrates a supervised regression workflow for estimating an annual health insurance premium in Indian rupees using an Indian-oriented public dataset. It covers data validation, EDA, feature engineering, model comparison, cross-validation, hyperparameter tuning, evaluation, model persistence and a Streamlit interface.

## Problem Statement
Given age, health indicators, height, weight and major surgery history, estimate the continuous target `PremiumPrice` from the Medical Insurance Premium Prediction dataset.

## Dataset
- Source: [Kaggle — tejashvi14/medical-insurance-premium-prediction](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction)
- 986 rows, 11 columns (10 inputs and one target).
- Inputs: Age, Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, Height, Weight, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries.
- Target: PremiumPrice, used directly as INR. No USD conversion is performed.

## Methodology
1. Load CSV and normalize source column names.
2. Validate schema; inspect missing values, duplicates and outliers.
3. Split data 80:20 before fitting transformations.
4. Derive BMI, BMI category and surgery-history group.
5. Fit imputation, scaling and categorical encoding inside an sklearn Pipeline.
6. Compare Linear Regression, Decision Tree, Random Forest and Gradient Boosting with 5-fold cross-validation.
7. Tune the selected model using GridSearchCV on training data.
8. Evaluate on held-out data with MAE, MSE, RMSE and R².
9. Serialize the complete pipeline with Joblib and serve predictions with Streamlit.

## Limitations and Responsible Use
The dataset has only 986 records and its label-generation and collection methodology should be reviewed at the source. It may not represent current Indian insurer underwriting, policy terms, sum insured, location-based pricing or all customer populations. The output is an educational estimate, not an insurer quote or a real underwriting decision tool.
