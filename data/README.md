# Indian Health Insurance Dataset

## Important: use the correct CSV

The repository previously contained the US Medical Cost dataset. **Do not use that file** for this Indian project. The old dataset has `sex`, `bmi`, `children`, `smoker`, `region` and `charges`; those columns do not belong to the Indian dataset.

1. Download **Medical Insurance Premium Prediction** from https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction
2. Extract the CSV (commonly named `Medicalpremium.csv`).
3. Rename it to `insurance.csv`.
4. Place it at `data/insurance.csv` in this repository.

The CSV must contain these 11 columns:

`Age, Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, Height, Weight, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries, PremiumPrice`.

The model uses `PremiumPrice` directly as the dataset's INR-denominated target. No USD conversion or exchange-rate multiplier is applied.

After replacing the CSV, retrain the model with `python -m src.model_training`. Do not reuse a model file trained on the old US dataset. The dataset is intended for education and portfolio demonstration; review its source and licence before redistribution.
