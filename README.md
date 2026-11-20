# Indian Health Insurance Premium Prediction

An end-to-end supervised Machine Learning regression project that estimates an annual health insurance premium in **Indian rupees (₹)** using age, health history, body measurements and surgery history.

> **Important:** This is an educational model trained on a small public dataset. It is not an official insurer quote and must not be used for underwriting or medical/financial decisions.

## Dataset — read this before training

- **Dataset:** Medical Insurance Premium Prediction
- **Source:** [Kaggle dataset](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction)
- **Size reported by source:** 986 records and 11 columns.
- **Target:** `PremiumPrice`, used directly as INR.
- **No currency conversion:** the old US Medical Cost dataset is not compatible with this project.

Download `Medicalpremium.csv`, rename it to `insurance.csv`, and put it in `data/insurance.csv`. Follow [data/README.md](data/README.md). The application deliberately rejects the old US CSV rather than silently creating fake or irrelevant features.

| Input | Meaning |
|---|---|
| Age | Applicant age in years |
| Diabetes | Diabetes indicator (0/1) |
| BloodPressureProblems | Blood pressure problems (0/1) |
| AnyTransplants | Organ transplant history (0/1) |
| AnyChronicDiseases | Chronic disease history (0/1) |
| Height | Height in centimetres |
| Weight | Weight in kilograms |
| KnownAllergies | Known allergies (0/1) |
| HistoryOfCancerInFamily | Family cancer history (0/1) |
| NumberOfMajorSurgeries | Number of major surgeries |
| **PremiumPrice** | **Target premium (₹)** |

## ML workflow

1. Validate and normalize the original Kaggle column names.
2. Clean invalid values, inspect missing values, duplicates and outliers.
3. Split data into train/test sets (80:20).
4. Engineer BMI, BMI category and surgery-history category.
5. Fit imputation, scaling and one-hot encoding inside an sklearn Pipeline.
6. Compare Linear Regression, Decision Tree, Random Forest and Gradient Boosting using 5-fold CV.
7. Tune the selected model with GridSearchCV on training data only.
8. Evaluate with MAE, MSE, RMSE and R².
9. Save and reload the complete pipeline with schema metadata.
10. Use Streamlit to collect applicant inputs and display an INR estimate.

## Run on Windows

```cmd
git pull origin main
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the Indian CSV as described above, then run:

```cmd
python -m src.model_training
streamlit run app.py
```

If you previously ran the US version locally, replace `data/insurance.csv` with the Indian CSV and delete the old `models/best_model.joblib` and `models/model_metadata.json` before retraining. The new training run recreates both files.

## Limitations

The dataset is small and its premium-label methodology may not represent current Indian insurer underwriting, plan benefits, sum insured, location pricing or all customer populations. Treat the output only as an educational ML estimate.
