"""Regression metrics and diagnostic plots; all monetary values are INR."""
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from src import EVAL_OUTPUT_DIR

def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mse=float(mean_squared_error(y_true,y_pred))
    return {"MAE":round(float(mean_absolute_error(y_true,y_pred)),2),
            "MSE":round(mse,2),"RMSE":round(float(np.sqrt(mse)),2),
            "R2":round(float(r2_score(y_true,y_pred)),4)}

def _save(fig, path):
    path.parent.mkdir(parents=True,exist_ok=True); fig.tight_layout()
    fig.savefig(path,dpi=140,bbox_inches="tight"); plt.close(fig); return str(path)

def plot_actual_vs_predicted(y_true,y_pred,model_name,save_path=EVAL_OUTPUT_DIR/"actual_vs_predicted.png"):
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(y_true,y_pred,alpha=.65)
    lo=min(float(np.min(y_true)),float(np.min(y_pred))); hi=max(float(np.max(y_true)),float(np.max(y_pred)))
    ax.plot([lo,hi],[lo,hi],"r--"); ax.set(title=f"Actual vs Predicted — {model_name}",xlabel="Actual premium (₹)",ylabel="Predicted premium (₹)")
    return _save(fig,save_path)

def plot_residuals(y_true,y_pred,model_name,save_path=EVAL_OUTPUT_DIR/"residual_plot.png"):
    residual=np.asarray(y_true)-np.asarray(y_pred); fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].scatter(y_pred,residual,alpha=.6); axes[0].axhline(0,color="red",linestyle="--")
    axes[0].set(xlabel="Predicted premium (₹)",ylabel="Residual (₹)",title=f"Residuals — {model_name}")
    sns.histplot(residual,kde=True,ax=axes[1]); axes[1].set_xlabel("Residual (₹)")
    return _save(fig,save_path)

def extract_feature_importance(pipeline: Pipeline,top_n=15):
    pre=pipeline.named_steps["preprocessor"]; model=pipeline.named_steps["model"]
    names=[n.split("__",1)[-1] for n in pre.get_feature_names_out()]
    vals=np.asarray(model.feature_importances_ if hasattr(model,"feature_importances_") else np.abs(model.coef_).ravel())
    return pd.DataFrame({"feature":names,"importance":vals}).sort_values("importance",ascending=False).head(top_n).reset_index(drop=True)

def plot_feature_importance(df,model_name,save_path=EVAL_OUTPUT_DIR/"feature_importance.png"):
    fig,ax=plt.subplots(figsize=(8,5)); sns.barplot(data=df,x="importance",y="feature",ax=ax)
    ax.set_title(f"Feature Importance — {model_name}"); return _save(fig,save_path)

def plot_model_comparison(df,save_path=EVAL_OUTPUT_DIR/"model_comparison.png"):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    sns.barplot(data=df,x="Model",y="RMSE",ax=axes[0]); axes[0].set_ylabel("RMSE (₹)"); axes[0].tick_params(axis="x",rotation=20)
    sns.barplot(data=df,x="Model",y="R2",ax=axes[1]); axes[1].tick_params(axis="x",rotation=20)
    return _save(fig,save_path)

def perform_error_analysis(X_test,y_true,y_pred):
    errors=np.abs(np.asarray(y_true)-np.asarray(y_pred))
    return {"mean_residual":float(np.mean(np.asarray(y_true)-np.asarray(y_pred))),
            "median_abs_error":float(np.median(errors)),"p90_abs_error":float(np.percentile(errors,90)),
            "max_abs_error":float(np.max(errors))}
