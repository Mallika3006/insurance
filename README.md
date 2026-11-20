# Indian Health Insurance Premium Prediction

An end-to-end supervised Machine Learning regression project that estimates an annual health insurance premium in **Indian rupees (₹)** using age, health history, body measurements and surgery history.

> **Important:** This is an educational model trained on a small public dataset. It is not an official insurer quote and must not be used for underwriting or medical/financial decisions.

## Application features

- **BMI calculator:** Calculates BMI automatically when height or weight is changed and shows a general adult BMI range.
- **Premium prediction:** Uses the existing trained scikit-learn model to estimate an annual premium in INR.
- **Health details summary:** Displays the health-history selections entered by the user. It does not diagnose conditions or provide a medical risk assessment.

## Dataset — read this before training

- **Dataset:** Medical Insurance Premium Prediction
- **Source:** [Kaggle dataset](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction)
- **Size reported by source:** 986 records and 11 columns.
- **Target:** `PremiumPrice`, used directly as INR.
- **No currency conversion:** the old US Medical Cost dataset is not compatible with this project.

The Indian dataset is included at `data/insurance.csv`, so a fresh clone does not require a separate Kaggle download. The application deliberately rejects the old US CSV rather than silently creating fake or irrelevant features.

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
10. Use Streamlit to calculate BMI, summarize entered health details and display an INR estimate.

## Run on Windows

```cmd
git pull origin main
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the application:

```cmd
streamlit run app.py
```

On the first visit to **Predict Premium**, the application automatically trains the model from the bundled Indian dataset and saves the pipeline and metadata in `models/`. This first training run may take a few minutes. Later launches reuse the saved model. To retrain manually after deleting the model, run `python -m src.model_training`.

If you previously ran the US version locally, pull the latest changes. The incompatible US model artifacts are not included; the app will train a fresh Indian model automatically.

## Limitations

The dataset is small and its premium-label methodology may not represent current Indian insurer underwriting, plan benefits, sum insured, location pricing or all customer populations. BMI is a general screening measure and the health details summary only repeats user-entered information. Treat all outputs only as educational information.
