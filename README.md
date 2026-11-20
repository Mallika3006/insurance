# Health Insurance Premium Prediction Using Machine Learning

An end-to-end, portfolio-ready Machine Learning regression project that estimates an individual's approximate annual health insurance charges from personal, demographic, and lifestyle attributes, paired with a clean three-page Streamlit web application.

> **Educational Disclaimer:** This project is built for educational and portfolio demonstration purposes using the publicly available Medical Cost Personal Dataset. It is **not** an official insurance underwriting or quotation engine.

---

## 1. Project Overview

Health insurance costs vary widely across individuals based on age, body mass index (BMI), smoking habits, dependents, and region. This project implements a complete, reproducible Machine Learning workflow—from Exploratory Data Analysis (EDA) and leakage-free preprocessing pipelines to model comparison, hyperparameter tuning, serialization with Joblib, and local deployment via Streamlit.

## 2. Problem Statement

Given demographic and lifestyle attributes for an individual (`age`, `sex`, `bmi`, `children`, `smoker`, `region`), predict the continuous target variable `charges` (annual medical costs billed by health insurance) as accurately as possible while generalizing cleanly to unseen data.

## 3. Project Objectives

- Perform structured Exploratory Data Analysis (EDA) across all input features and the target variable.
- Clean the dataset and prevent data leakage using scikit-learn `ColumnTransformer` and `Pipeline`.
- Evaluate domain-informed engineered features (BMI categories, age groups, and smoking-BMI interaction) against raw features.
- Train and benchmark four regression algorithms: Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor.
- Tune the best candidate model using `GridSearchCV` with 5-fold cross-validation on training data only.
- Save the complete preprocessing + prediction pipeline with `joblib` and verify reload consistency.
- Serve predictions and saved model insights through a clean local Streamlit web application.

## 4. Dataset Description

- **Dataset Name:** Medical Cost Personal Datasets
- **Source:** [Kaggle — mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **File Name:** `insurance.csv` (to be placed in `data/insurance.csv`)
- **Problem Type:** Supervised Regression (tabular data)

## 5. Features and Target Variable

| Column | Type | Description |
| :--- | :--- | :--- |
| `age` | Numerical (int) | Age of primary beneficiary |
| `sex` | Categorical | Gender (`female`, `male`) |
| `bmi` | Numerical (float) | Body Mass Index (kg/m²) |
| `children` | Numerical (int) | Number of children / dependents covered |
| `smoker` | Categorical | Smoking status (`yes`, `no`) |
| `region` | Categorical | US residential region (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges` | Numerical (float) | **Target Variable** — Annual medical costs billed by health insurance ($) |

## 6. Technologies Used

- **Python** — Core programming language
- **Pandas & NumPy** — Data manipulation and numerical computing
- **Matplotlib & Seaborn** — EDA and model diagnostic visualizations
- **Scikit-learn** — Preprocessing (`ColumnTransformer`, `Pipeline`), regression models, cross-validation, and `GridSearchCV`
- **Joblib** — Saving and loading the trained pipeline
- **Streamlit** — Interactive multi-page web application
- **Jupyter Notebook** — Step-by-step interactive experimentation and reporting

## 7. Machine Learning Algorithms

1. **Linear Regression:** Fits a linear combination of encoded and scaled features to minimize squared residuals. Serves as an interpretable baseline.
2. **Decision Tree Regressor:** Recursively partitions the feature space using variance-reducing splits, capturing non-linear interactions such as smoking × BMI.
3. **Random Forest Regressor:** An ensemble of decorrelated decision trees trained on bootstrap samples, averaging predictions to reduce variance and overfitting.
4. **Gradient Boosting Regressor:** Builds shallow trees sequentially, where each new tree learns to correct the residual errors of the previous ensemble.

## 8. Project Architecture

```text
data/insurance.csv
       │
       ▼
src/data_preprocessing.py  ──► Data loading, schema validation, deduplication, 80/20 split
       │
       ├──► src/eda.py  ──► outputs/eda/*.png + outputs/eda_summary.json
       │
       ▼
src/feature_engineering.py ──► InsuranceFeatureEngineer (BMI category, age group, smoker×BMI)
       │
       ▼
src/model_training.py      ──► Pipeline + 4-model CV comparison + GridSearchCV tuning
       │
       ├──► src/model_evaluation.py ──► MAE, MSE, RMSE, R², diagnostic plots, error analysis
       ├──► models/best_model.joblib + models/model_metadata.json
       └──► outputs/model_comparison.csv, tuning_comparison.json, feature_importance.csv
       │
       ▼
src/prediction.py ◄── app.py (Streamlit 3-page web application)
```

## 9. Folder Structure

```text
Health_Insurance_Premium_Prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── README.md
│   └── insurance.csv                # Place downloaded Kaggle dataset here
├── notebooks/
│   └── health_insurance_prediction.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── prediction.py
├── models/
│   └── .gitkeep                     # best_model.joblib & model_metadata.json saved here
├── outputs/
│   └── .gitkeep                     # EDA plots, evaluation plots, and CSV/JSON metrics saved here
├── assets/
│   └── .gitkeep                     # Store app screenshots here
└── docs/
    ├── project_report.md
    └── interview_questions.md
```

## 10. Installation Instructions (Windows Command Prompt)

Ensure **Python 3.9–3.12** is installed on your machine. Open **Command Prompt (`cmd`)** or **PowerShell** inside the extracted `Health_Insurance_Premium_Prediction` folder:

```cmd
:: 1. Create a virtual environment
python -m venv venv

:: 2. Activate the virtual environment (Windows Command Prompt)
venv\Scripts\activate

:: (If using PowerShell instead of Command Prompt, run: .\venv\Scripts\Activate.ps1)

:: 3. Upgrade pip and install all required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 11. Dataset Setup Instructions

Because this project does not bundle or fabricate dataset files, download the real dataset from Kaggle before training:

1. Visit [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance).
2. Download and unzip the archive to obtain `insurance.csv`.
3. Place `insurance.csv` inside the `data\` folder:
   ```text
   Health_Insurance_Premium_Prediction\data\insurance.csv
   ```
4. Verify the file exists:
   ```cmd
   dir data\insurance.csv
   ```

See `data/README.md` for optional Kaggle CLI download instructions.

## 12. How to Train the Model

With the virtual environment activated and `data\insurance.csv` in place, run the end-to-end training script:

```cmd
python -m src.model_training
```

This command will:
- Load and clean `data/insurance.csv`.
- Generate all 8 EDA plots in `outputs/eda/` and `outputs/eda_summary.json`.
- Split data 80/20 (`random_state=42`) and compare raw vs. engineered features across all 4 models.
- Tune the top candidate model with 5-fold `GridSearchCV` on training data only.
- Evaluate on the untouched test set, generate diagnostic plots in `outputs/evaluation/`, and save all metrics to `outputs/`.
- Save the final pipeline to `models/best_model.joblib` and verify predictions after reload.

Alternatively, you can run every step interactively in Jupyter Notebook:

```cmd
jupyter notebook notebooks\health_insurance_prediction.ipynb
```

## 13. How to Run the Streamlit Application

After training completes (creating `models\best_model.joblib` and `outputs\`), launch the Streamlit app:

```cmd
streamlit run app.py
```

Open the local URL displayed in your terminal (typically `http://localhost:8501`).

## 14. Screenshots

Capture screenshots of your running Streamlit app and save them in `assets/` to replace the placeholders below:

- **Home Page:** `![Home Page](assets/screenshot_home.png)`
- **Prediction Page:** `![Prediction Page](assets/screenshot_prediction.png)`
- **Model Insights Page:** `![Model Insights Page](assets/screenshot_insights.png)`

## 15. Model Evaluation Results

> **Note on Actual Execution Results:** Model metrics are **not hardcoded or fabricated**. After running `python -m src.model_training` on the real Kaggle `data/insurance.csv` dataset, actual test metrics are automatically saved to:
> - `outputs/model_comparison.csv` — Side-by-side MAE, MSE, RMSE, and R² for all four models
> - `outputs/feature_engineering_comparison.csv` — Raw vs. engineered feature set comparison
> - `outputs/tuning_comparison.json` — Before vs. after `GridSearchCV` tuning metrics and best hyperparameters
> - `outputs/error_analysis.json` — Residual statistics and largest test-set prediction errors
> - `models/model_metadata.json` — Final selected model, sample counts, and reload verification check

Once you execute training locally, copy your actual run values from `outputs/model_comparison.csv` into the table below for your portfolio README:

| Model | CV RMSE | Test MAE ($) | Test MSE | Test RMSE ($) | Test R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Linear Regression | *Run training* | *Run training* | *Run training* | *Run training* | *Run training* |
| Decision Tree Regressor | *Run training* | *Run training* | *Run training* | *Run training* | *Run training* |
| Random Forest Regressor | *Run training* | *Run training* | *Run training* | *Run training* | *Run training* |
| Gradient Boosting Regressor | *Run training* | *Run training* | *Run training* | *Run training* | *Run training* |

## 16. Future Improvements

- Explore target log-transformation (`np.log1p(charges)`) with `TransformedTargetRegressor` to further stabilize right-skewed residuals.
- Add prediction intervals (e.g., quantile regression with Gradient Boosting) to show a low/high estimated range rather than only a point estimate.
- Incorporate additional health factors (pre-existing conditions, exercise frequency, blood pressure) if richer authorized datasets become available.
- Add automated unit tests with `pytest` and CI workflows for regression testing.
- Containerize the Streamlit application with Docker for cloud deployment.
