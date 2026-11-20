"""Dataset-aware EDA plots for Indian health insurance premiums."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src import DEFAULT_DATA_PATH, EDA_OUTPUT_DIR, OUTPUTS_DIR, TARGET_COLUMN
from src.data_preprocessing import clean_dataset, load_dataset

def run_eda(df: pd.DataFrame, output_dir: Path = EDA_OUTPUT_DIR,
            summary_path: Path = OUTPUTS_DIR / "eda_summary.json"):
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes(include="number")
    plots = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], kde=True, ax=ax)
    ax.set(title="Distribution of Annual Premium (INR)", xlabel="Annual premium (₹)")
    fig.tight_layout(); p=output_dir/"01_premium_distribution.png"; fig.savefig(p,dpi=140); plt.close(fig)
    plots["premium_distribution"] = str(p)
    for col in ["age", "height_cm", "weight_kg"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x=col, y=TARGET_COLUMN, ax=ax)
        ax.set_ylabel("Annual premium (₹)"); ax.set_title(f"{col.replace('_',' ').title()} vs Premium")
        fig.tight_layout(); p=output_dir/f"02_{col}_vs_premium.png"; fig.savefig(p,dpi=140); plt.close(fig)
        plots[f"{col}_vs_premium"] = str(p)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Numeric Feature Correlations")
    fig.tight_layout(); p=output_dir/"06_correlation_heatmap.png"; fig.savefig(p,dpi=140); plt.close(fig)
    plots["correlation_heatmap"] = str(p)
    summary = {"row_count": int(len(df)), "premium_inr": {
        "mean": float(df[TARGET_COLUMN].mean()), "median": float(df[TARGET_COLUMN].median()),
        "min": float(df[TARGET_COLUMN].min()), "max": float(df[TARGET_COLUMN].max())},
        "missing_values": df.isna().sum().to_dict(),
        "correlations_with_premium": numeric.corr()[TARGET_COLUMN].to_dict()}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"plot_paths": plots, "summary": summary}

if __name__ == "__main__":
    data, _ = clean_dataset(load_dataset(DEFAULT_DATA_PATH))
    run_eda(data)
