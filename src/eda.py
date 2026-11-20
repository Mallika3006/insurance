"""Exploratory Data Analysis (EDA) visualization and summary statistics module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import DEFAULT_DATA_PATH, EDA_OUTPUT_DIR, OUTPUTS_DIR
from src.data_preprocessing import clean_dataset, load_dataset

sns.set_theme(style="whitegrid", palette="Set2")


def _save_fig(fig: plt.Figure, save_path: Path) -> str:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_charges_distribution(df: pd.DataFrame, output_dir: Path = EDA_OUTPUT_DIR) -> str:
    """Plot histogram with KDE and box plot for insurance charges."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.histplot(df["charges"], kde=True, bins=35, color="#2b6cb0", ax=axes[0])
    axes[0].set_title("Distribution of Annual Insurance Charges")
    axes[0].set_xlabel("Charges ($)")
    axes[0].set_ylabel("Count")

    sns.boxplot(x=df["charges"], color="#63b3ed", ax=axes[1])
    axes[1].set_title("Box Plot of Annual Insurance Charges")
    axes[1].set_xlabel("Charges ($)")
    return _save_fig(fig, output_dir / "01_charges_distribution.png")


def plot_age_vs_charges(df: pd.DataFrame, output_dir: Path = EDA_OUTPUT_DIR) -> str:
    """Plot scatter plot of age versus charges colored by smoking status."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.scatterplot(
        data=df,
        x="age",
        y="charges",
        hue="smoker",
        alpha=0.75,
        palette={"no": "#2b6cb0", "yes": "#e53e3e"},
        ax=ax,
    )
    ax.set_title("Age vs. Insurance Charges (Grouped by Smoking Status)")
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Annual Charges ($)")
    return _save_fig(fig, output_dir / "02_age_vs_charges.png")


def plot_bmi_vs_charges(df: pd.DataFrame, output_dir: Path = EDA_OUTPUT_DIR) -> str:
    """Plot scatter plot of BMI versus charges colored by smoking status."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.scatterplot(
        data=df,
        x="bmi",
        y="charges",
        hue="smoker",
        alpha=0.75,
        palette={"no": "#2b6cb0", "yes": "#e53e3e"},
        ax=ax,
    )
    ax.axvline(30.0, color="#4a5568", linestyle="--", linewidth=1.2, label="Obesity Threshold (BMI = 30)")
    ax.set_title("BMI vs. Insurance Charges (Grouped by Smoking Status)")
    ax.set_xlabel("Body Mass Index (BMI)")
    ax.set_ylabel("Annual Charges ($)")
    ax.legend()
    return _save_fig(fig, output_dir / "03_bmi_vs_charges.png")


def plot_smoker_vs_charges(df: pd.DataFrame, output_dir: Path = EDA_OUTPUT_DIR) -> str:
    """Plot box plot and average charges bar chart by smoking status."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    sns.boxplot(
        data=df,
        x="smoker",
        y="charges",
        hue="smoker",
        legend=False,
        palette={"no": "#2b6cb0", "yes": "#e53e3e"},
        ax=axes[0],
    )
    axes[0].set_title("Insurance Charges by Smoking Status")
    axes[0].set_xlabel("Smoker")
    axes[0].set_ylabel("Annual Charges ($)")

    mean_by_smoker = df.groupby("smoker", as_index=False)["charges"].mean()
    sns.barplot(
        data=mean_by_smoker,
        x="smoker",
        y="charges",
        hue="smoker",
        legend=False,
        palette={"no": "#2b6cb0", "yes": "#e53e3e"},
        ax=axes[1],
    )
    axes[1].set_title("Mean Insurance Charges by Smoking Status")
    axes[1].set_xlabel("Smoker")
    axes[1].set_ylabel("Mean Annual Charges ($)")
    return _save_fig(fig, output_dir / "04_smoker_vs_charges.png")


def plot_gender_vs_charges(df: pd.DataFrame, output_dir: Path = EDA_OUTPUT_DIR) -> str:
    """Plot box plot and average charges bar chart by gender."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    sns.boxplot(data=df, x="sex", y="charges", hue="sex", legend=False, ax=axes[0])
    axes[0].set_title("Insurance Charges by Gender")
    axes[0].set_xlabel("Gender")
    axes[0].set_ylabel("Annual Charges ($)")

    mean_by_sex = df.groupby("sex", as_index=False)["charges"].mean()
    sns.barplot(data=mean_by_sex, x="sex", y="charges", hue="sex", legend=False, ax=axes[1])
    axes[1].set_title("Mean Insurance Charges by Gender")
    axes[1].set_xlabel("Gender")
    axes[1].set_ylabel("Mean Annual Charges ($)")
    return _save_fig(fig, output_dir / "05_gender_vs_charges.png")


def plot_region_vs_charges(df: pd.DataFrame, output_dir: Path = EDA_OUTPUT_DIR) -> str:
    """Plot box plot and average charges bar chart by residential region."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sns.boxplot(data=df, x="region", y="charges", hue="region", legend=False, ax=axes[0])
    axes[0].set_title("Insurance Charges by Region")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("Annual Charges ($)")

    mean_by_region = df.groupby("region", as_index=False)["charges"].mean()
    sns.barplot(data=mean_by_region, x="region", y="charges", hue="region", legend=False, ax=axes[1])
    axes[1].set_title("Mean Insurance Charges by Region")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Mean Annual Charges ($)")
    return _save_fig(fig, output_dir / "06_region_vs_charges.png")


def plot_children_vs_charges(df: pd.DataFrame, output_dir: Path = EDA_OUTPUT_DIR) -> str:
    """Plot box plot and average charges bar chart by number of dependents."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sns.boxplot(data=df, x="children", y="charges", hue="children", legend=False, ax=axes[0])
    axes[0].set_title("Insurance Charges by Number of Children")
    axes[0].set_xlabel("Number of Children")
    axes[0].set_ylabel("Annual Charges ($)")

    mean_by_children = df.groupby("children", as_index=False)["charges"].mean()
    sns.barplot(data=mean_by_children, x="children", y="charges", hue="children", legend=False, ax=axes[1])
    axes[1].set_title("Mean Insurance Charges by Number of Children")
    axes[1].set_xlabel("Number of Children")
    axes[1].set_ylabel("Mean Annual Charges ($)")
    return _save_fig(fig, output_dir / "07_children_vs_charges.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path = EDA_OUTPUT_DIR) -> str:
    """Plot correlation heatmap across numerical columns and binary smoker indicator."""
    corr_df = df[["age", "bmi", "children", "charges"]].copy()
    corr_df["smoker_binary"] = (df["smoker"].str.lower() == "yes").astype(int)
    corr_matrix = corr_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    return _save_fig(fig, output_dir / "08_correlation_heatmap.png")


def compute_eda_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Compute real descriptive statistics from the dataset for reporting and Streamlit display."""
    corr_df = df[["age", "bmi", "children", "charges"]].copy()
    corr_df["smoker_binary"] = (df["smoker"].str.lower() == "yes").astype(int)
    correlations = corr_df.corr(numeric_only=True)["charges"].round(4).to_dict()

    return {
        "row_count": int(len(df)),
        "charges_stats": {
            "mean": round(float(df["charges"].mean()), 2),
            "median": round(float(df["charges"].median()), 2),
            "std": round(float(df["charges"].std()), 2),
            "min": round(float(df["charges"].min()), 2),
            "max": round(float(df["charges"].max()), 2),
        },
        "mean_charges_by_smoker": {
            str(k): round(float(v), 2)
            for k, v in df.groupby("smoker")["charges"].mean().items()
        },
        "mean_charges_by_sex": {
            str(k): round(float(v), 2)
            for k, v in df.groupby("sex")["charges"].mean().items()
        },
        "mean_charges_by_region": {
            str(k): round(float(v), 2)
            for k, v in df.groupby("region")["charges"].mean().items()
        },
        "mean_charges_by_children": {
            str(int(k)): round(float(v), 2)
            for k, v in df.groupby("children")["charges"].mean().items()
        },
        "correlations_with_charges": correlations,
    }


def run_eda(
    df: pd.DataFrame,
    output_dir: Path = EDA_OUTPUT_DIR,
    summary_path: Path = OUTPUTS_DIR / "eda_summary.json",
) -> Dict[str, object]:
    """Generate all EDA plots and save a JSON summary of computed statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        "charges_distribution": plot_charges_distribution(df, output_dir),
        "age_vs_charges": plot_age_vs_charges(df, output_dir),
        "bmi_vs_charges": plot_bmi_vs_charges(df, output_dir),
        "smoker_vs_charges": plot_smoker_vs_charges(df, output_dir),
        "gender_vs_charges": plot_gender_vs_charges(df, output_dir),
        "region_vs_charges": plot_region_vs_charges(df, output_dir),
        "children_vs_charges": plot_children_vs_charges(df, output_dir),
        "correlation_heatmap": plot_correlation_heatmap(df, output_dir),
    }
    summary = compute_eda_summary(df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return {"plot_paths": plot_paths, "summary": summary}


if __name__ == "__main__":
    raw_df = load_dataset(DEFAULT_DATA_PATH)
    cleaned_df, _ = clean_dataset(raw_df)
    results = run_eda(cleaned_df)
    print(f"EDA completed. Generated {len(results['plot_paths'])} plots in '{EDA_OUTPUT_DIR}'.")
