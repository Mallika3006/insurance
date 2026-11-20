# Dataset Setup Instructions

This project uses the publicly available **Medical Cost Personal Dataset** from Kaggle.
To keep the repository clean and avoid fabricating data, `insurance.csv` is not bundled and must be downloaded directly from Kaggle.

## Dataset Details

- **Source:** [Medical Cost Personal Datasets (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Expected File Location:** `data/insurance.csv`
- **Expected Columns:**
  - `age` — Age of primary beneficiary (integer)
  - `sex` — Insurance contractor gender (`female`, `male`)
  - `bmi` — Body Mass Index (float)
  - `children` — Number of children / dependents covered by health insurance (integer)
  - `smoker` — Smoking status (`yes`, `no`)
  - `region` — Beneficiary's residential area in the US (`northeast`, `northwest`, `southeast`, `southwest`)
  - `charges` — Individual medical costs billed by health insurance (target variable, float)

## Option 1: Download Manually from Kaggle (Recommended for Beginners)

1. Open [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance) in your browser.
2. Sign in to your Kaggle account and click **Download**.
3. Extract the downloaded archive and locate `insurance.csv`.
4. Copy `insurance.csv` into this folder so the path is:
   ```text
   Health_Insurance_Premium_Prediction/data/insurance.csv
   ```

## Option 2: Download Using Kaggle CLI (Windows Command Prompt)

If you have the Kaggle CLI configured (`kaggle.json` placed in `%USERPROFILE%\.kaggle\`):

```cmd
pip install kaggle
kaggle datasets download -d mirichoi0218/insurance -p data --unzip
```

## Verify Placement

Once placed, verify the file exists before training:

```cmd
dir data\insurance.csv
```
