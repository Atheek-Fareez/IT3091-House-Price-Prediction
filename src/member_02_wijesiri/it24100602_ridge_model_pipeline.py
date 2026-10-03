# -*- coding: utf-8 -*-
"""IT24100602_Ridge_Model_Pipeline
Member 2: Wijesiri S.P.R.H. (IT24100602)
Group 2026-AI-08K · IT3091 Machine Learning Project
Model 02: Regularized Linear Model (Ridge Regression) Pipeline & Feature Analysis
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

# Resolve project root and import Member 3 preprocessor
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "src/member_03_wazni_ahamed/preprocessing.py").is_file()
             or (p / "src/member_03/preprocessing.py").is_file()), Path("."))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.member_03_wazni_ahamed.preprocessing import load_ames_data, build_preprocessor, get_feature_names
except ModuleNotFoundError:
    from src.member_03.preprocessing import load_ames_data, build_preprocessor, get_feature_names

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print(f"[INIT] Repository root found at: {ROOT}")


def main():
    # 1. Load Data
    train_df, test_df = load_ames_data(str(ROOT))
    print(f"[DATA] Train: {train_df.shape[0]} rows, Test: {test_df.shape[0]} rows")

    X = train_df.drop(columns=["Id", "SalePrice"]).copy()
    y = train_df["SalePrice"].astype(float).copy()
    y_log = np.log1p(y)

    RANDOM_STATE = 42
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 2. Academic Baseline Benchmark (SLIIT Required)
    baseline = DummyRegressor(strategy="mean")
    b_log_preds = cross_val_predict(baseline, X, y_log, cv=cv)
    b_dollar_preds = np.expm1(b_log_preds)
    b_rmse = float(np.sqrt(mean_squared_error(y, b_dollar_preds)))
    b_mae = float(mean_absolute_error(y, b_dollar_preds))
    b_r2 = float(r2_score(y, b_dollar_preds))
    print(f"[BASELINE] RMSE: ${b_rmse:,.2f} | MAE: ${b_mae:,.2f} | R²: {b_r2:.4f}")

    # 3. Ridge Regression Regularization Grid Tuning
    alpha_values = [0.1, 1.0, 10.0, 50.0, 100.0]
    results = []

    print("[TRAIN] Evaluating Ridge across alpha regularization values...")
    for alpha in alpha_values:
        pipe = Pipeline([
            ("prep", build_preprocessor(
                configuration="scaled",
                log_numeric=True,
                sparse_output=False
            )),
            ("reg", Ridge(alpha=alpha, random_state=RANDOM_STATE))
        ])

        y_log_pred = cross_val_predict(pipe, X, y_log, cv=cv)
        y_pred = np.expm1(y_log_pred)

        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))
        r2_dollar = float(r2_score(y, y_pred))

        cv_scores = cross_validate(
            pipe, X, y_log, cv=cv, scoring="r2", return_train_score=True
        )
        train_r2 = float(cv_scores["train_score"].mean())
        val_r2 = float(cv_scores["test_score"].mean())
        overfit_gap = float(train_r2 - val_r2)

        results.append({
            "Alpha": alpha,
            "RMSE": rmse,
            "MAE": mae,
            "Dollar R2": r2_dollar,
            "Train R2": train_r2,
            "Validation R2": val_r2,
            "Overfit Gap": overfit_gap
        })

    results_df = pd.DataFrame(results)
    print("\n[GRID RESULTS]")
    print(results_df.to_string(index=False))

    # Select best alpha (alpha=0.1 as per Member 2 notebook)
    best_alpha = 0.1
    best_row = results_df[results_df["Alpha"] == best_alpha].iloc[0]

    # 4. Train Final Pipeline on Full Data
    final_ridge_pipeline = Pipeline([
        ("prep", build_preprocessor(
            configuration="scaled",
            log_numeric=True,
            sparse_output=False
        )),
        ("reg", Ridge(alpha=best_alpha, random_state=RANDOM_STATE))
    ])
    final_ridge_pipeline.fit(X, y_log)
    print(f"\n[FINAL FIT] Ridge fitted with alpha={best_alpha}")

    # Extract Out-of-fold predictions for residuals
    final_log_preds = cross_val_predict(final_ridge_pipeline, X, y_log, cv=cv)
    final_dollar_preds = np.expm1(final_log_preds)

    ridge_rmse = float(best_row["RMSE"])
    ridge_mae = float(best_row["MAE"])
    ridge_dollar_r2 = float(best_row["Dollar R2"])
    train_r2 = float(best_row["Train R2"])
    val_r2 = float(best_row["Validation R2"])
    overfit_gap = float(best_row["Overfit Gap"])

    print("\n" + "=" * 55)
    print("MEMBER 2 (WIJESIRI) RIDGE REGRESSION FINAL RESULTS")
    print("=" * 55)
    print(f"CV RMSE:          ${ridge_rmse:,.2f}")
    print(f"CV MAE:           ${ridge_mae:,.2f}")
    print(f"Dollar R²:        {ridge_dollar_r2:.4f}")
    print(f"Log Val R²:       {val_r2:.4f}")
    print(f"Log Train R²:     {train_r2:.4f}")
    print(f"Overfitting Gap:  {overfit_gap:.4f}")
    print("=" * 55)

    # 5. Extract Feature Names and Ridge Coefficients (Secondary Lens)
    feature_names = get_feature_names(final_ridge_pipeline.named_steps["prep"])
    coefficients = final_ridge_pipeline.named_steps["reg"].coef_

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })
    coef_df["AbsCoefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values(by="AbsCoefficient", ascending=False)

    top_positive = coef_df[coef_df["Coefficient"] > 0].sort_values(by="Coefficient", ascending=False).head(8)
    top_negative = coef_df[coef_df["Coefficient"] < 0].sort_values(by="Coefficient", ascending=True).head(8)
    top_impact = pd.concat([top_positive.head(5), top_negative.head(5)]).sort_values("Coefficient")

    # 6. Save Artifacts
    out_dir = ROOT / "notebooks/Member_02_Wijesiri"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = ROOT / "reports/Member_02_Wijesiri"
    rep_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "reports/model_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Save .joblib model
    joblib.dump(final_ridge_pipeline, out_dir / "ridge_model.joblib")
    print(f"[OK] Saved model to: {out_dir / 'ridge_model.joblib'}")

    # Save Top Coefficients Plot
    plt.figure(figsize=(10, 6))
    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in top_impact["Coefficient"]]
    plt.barh(top_impact["Feature"], top_impact["Coefficient"], color=colors)
    plt.xlabel("Ridge Coefficient (Impact on Log Sale Price)", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.title("Ames Housing: Top Positive & Negative Ridge Coefficients (Member 2)", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "member2_top_features.png", dpi=300)
    plt.savefig(rep_dir / "member2_top_features.png", dpi=300)
    plt.close()

    # Save Residual Plot
    residuals = y - final_dollar_preds
    plt.figure(figsize=(10, 5))
    plt.scatter(final_dollar_preds, residuals, alpha=0.4, color="#3498db")
    plt.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    plt.title("Residual Error Analysis — Ridge Regression (Predicted vs Error)", fontsize=13, weight="bold")
    plt.xlabel("Predicted Sale Price ($)", fontsize=11)
    plt.ylabel("Prediction Error ($)", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "member2_residuals.png", dpi=300)
    plt.savefig(rep_dir / "member2_residuals.png", dpi=300)
    plt.close()

    # Save Metrics JSON
    metrics_record = {
        "member": "Member 2 - Wijesiri S.P.R.H. (IT24100602)",
        "model_name": "Ridge Regression",
        "best_alpha": float(best_alpha),
        "baseline_benchmark": {
            "rmse": b_rmse,
            "mae": b_mae,
            "r2": b_r2
        },
        "ridge_performance": {
            "rmse": ridge_rmse,
            "mae": ridge_mae,
            "dollar_r2": ridge_dollar_r2,
            "train_r2": train_r2,
            "validation_r2": val_r2,
            "overfit_gap": overfit_gap
        },
        "top_positive_features": top_positive["Feature"].head(5).tolist(),
        "top_negative_features": top_negative["Feature"].head(5).tolist()
    }

    with open(results_dir / "member2_ridge_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=4)
    with open(rep_dir / "member2_ridge_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=4)
    print(f"[OK] Saved metrics to: {results_dir / 'member2_ridge_metrics.json'}")

    # 7. Generate Kaggle Test Set Predictions
    X_test = test_df.drop(columns=["Id"]).copy()
    test_dollar_preds = np.expm1(final_ridge_pipeline.predict(X_test))
    sub_df = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": test_dollar_preds
    })
    sub_df.to_csv(rep_dir / "submission_ridge.csv", index=False)
    print(f"[OK] Saved Kaggle submission to: {rep_dir / 'submission_ridge.csv'}")
    print("\nFirst 5 Predicted Houses in Kaggle Test Set:")
    print(sub_df.head())


if __name__ == "__main__":
    main()
