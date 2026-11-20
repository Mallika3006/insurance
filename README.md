# Indian Health Insurance Premium Prediction

An end-to-end supervised Machine Learning regression project that estimates an annual health insurance premium in **Indian rupees (₹)** from an applicant's age, health history, body measurements and surgery history.

> **Important:** This is an educational model trained on a small public Indian-oriented dataset. It is not an official quote, does not represent every Indian insurer or policy, and must not be used for underwriting or medical/financial decisions.

## Dataset
- **Dataset:** Medical Insurance Premium Prediction
- **Source:** [Kaggle — tejashvi14/medical-insurance-premium-prediction](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction)
- **Size:** 986 records, 11 columns (10 inputs + target)
- **Target:** `PremiumPrice`, used directly as rupees.
- **No currency conversion:** the old US Medical Cost Personal dataset and USD-to-INR multiplier have been removed.

Download the CSV from Kaggle, rename it to `insurance.csv`, and place it in `data/insurance.csv`. See [data/README.md](data/README.md).

| Input | Description |
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
| **PremiumPrice** | **Target premium amount (₹)** |

## Workflow
1. Load and validate the Indian dataset; normalize original column names.
2. Inspect missing values, duplicates, summary statistics and outliers.
3. Perform EDA for premium distribution, age, height, weight and correlations.
4. Engineer BMI, BMI category and surgery-history category.
5. Split into train/test sets (80:20, random state 42).
6. Compare Linear Regression, Decision Tree, Random Forest and Gradient Boosting using 5-fold CV.
7. Tune the selected model with GridSearchCV using training data only.
8. Evaluate using MAE, MSE, RMSE and R².
9. Save the full sklearn pipeline with Joblib.
10. Use Streamlit to accept Indian health-profile inputs and display an INR estimate.

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib and Streamlit.

## Run on Windows
```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the dataset as described above, then train and launch:
```cmd
python -m src.model_training
streamlit run app.py
```

## Evaluation
Metrics are generated from your downloaded dataset; no accuracy or test score is hardcoded. Review `outputs/model_comparison.csv`, `outputs/tuning_comparison.json` and `outputs/error_analysis.json`.

## Limitations
The dataset is small (986 rows), has limited health/demographic fields and may not represent current products, all regions, age groups or insurer pricing. Its premium-label methodology should be checked against the Kaggle source. This demonstrates a regression workflow, not actual market pricing. No USD values or exchange-rate assumptions are used.
