# Indian Health Insurance Dataset

1. Download the **Medical Insurance Premium Prediction** dataset from Kaggle: https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction
2. Extract the CSV file (commonly named `Medicalpremium.csv`).
3. Rename it to `insurance.csv` and place it in this folder: `data/insurance.csv`.

Expected schema: Age, Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, Height, Weight, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries, PremiumPrice. The dataset is described as 986 records and 11 columns.

The target `PremiumPrice` is used directly as an INR-denominated amount; there is no USD-to-INR conversion or hardcoded exchange rate. Do not redistribute the dataset unless its licence permits it. The model is an educational estimate, not an insurer quote.
