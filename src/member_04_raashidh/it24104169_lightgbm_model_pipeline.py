# -*- coding: utf-8 -*-
"""IT24104169_LightGBM_Model_Pipeline
Member 4: Raashidh M.R.A. (IT24104169)
Group 2026-AI-08K · IT3091 Machine Learning Project
Model 03: LightGBM Regressor Pipeline & Feature Analysis
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
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import joblib

# Resolve project root and import Member 3 preprocessor
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "src/member_03/preprocessing.py").is_file()), Path("."))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.member_03.preprocessing import load_ames_data, build_preprocessor

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

    # 2. Academic Baseline Benchmark
    baseline = DummyRegressor(strategy="mean")
    b_log_preds = cross_val_predict(baseline, X, y_log, cv=cv)
    b_dollar_preds = np.expm1(b_log_preds)
    b_rmse = float(np.sqrt(mean_squared_error(y, b_dollar_preds)))
    b_mae = float(mean_absolute_error(y, b_dollar_preds))
    b_r2 = float(r2_score(y, b_dollar_preds))
    print(f"[BASELINE] RMSE: ${b_rmse:,.2f} | MAE: ${b_mae:,.2f} | R2: {b_r2:.4f}")

    # 3. Build & Tune LightGBM Pipeline
    preprocessor = build_preprocessor(configuration="unscaled", log_numeric=False, sparse_output=False)
    lgbm_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ))
    ])

    param_grid = {
        "regressor__n_estimators": [200, 300],
        "regressor__learning_rate": [0.03, 0.05],
        "regressor__num_leaves": [20, 31, 40],
        "regressor__max_depth": [4, 6, -1],
        "regressor__subsample": [0.8, 1.0],
        "regressor__colsample_bytree": [0.7, 0.8]
    }

    print("[TRAIN] Tuning LightGBM with 5-Fold Cross-Validation...")
    grid_search = GridSearchCV(
        estimator=lgbm_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X, y_log)
    best_pipe = grid_search.best_estimator_
    print(f"[TUNING] Best Parameters: {grid_search.best_params_}")

    # 4. Out-of-fold Evaluation
    lgbm_log_preds = cross_val_predict(best_pipe, X, y_log, cv=cv)
    lgbm_dollar_preds = np.expm1(lgbm_log_preds)

    lgbm_rmse = float(np.sqrt(mean_squared_error(y, lgbm_dollar_preds)))
    lgbm_mae = float(mean_absolute_error(y, lgbm_dollar_preds))
    lgbm_r2 = float(r2_score(y, lgbm_dollar_preds))

    cv_scores = cross_validate(best_pipe, X, y_log, cv=cv, return_train_score=True, scoring="r2")
    train_r2 = float(cv_scores["train_score"].mean())
    val_r2 = float(cv_scores["test_score"].mean())
    overfit_gap = float(train_r2 - val_r2)

    print("\n" + "=" * 55)
    print("MEMBER 4 (RAASHIDH) LIGHTGBM FINAL RESULTS")
    print("=" * 55)
    print(f"CV RMSE:      ${lgbm_rmse:,.2f}")
    print(f"CV MAE:       ${lgbm_mae:,.2f}")
    print(f"CV R2:        {lgbm_r2:.4f}")
    print(f"Train R2:     {train_r2:.4f}")
    print(f"Val R2:       {val_r2:.4f}")
    print(f"Overfit Gap:  {overfit_gap:.4f}")
    print("=" * 55)

    # 5. Fit full model & Feature Importance
    best_pipe.fit(X, y_log)
    encoder_step = best_pipe.named_steps["preprocessor"].named_steps["encode"]
    feature_names = encoder_step.get_feature_names_out()
    importances = best_pipe.named_steps["regressor"].feature_importances_

    feature_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    feature_imp_df["Clean_Name"] = (
        feature_imp_df["Feature"]
        .str.replace("numeric__", "")
        .str.replace("nominal__", "")
        .str.replace("indicators__", "")
    )

    # 6. Save Artifacts
    out_dir = ROOT / "notebooks/Member_04_Raashidh"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = ROOT / "reports/Member_04_Raashidh"
    rep_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "reports/model_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Save .joblib
    joblib.dump(best_pipe, out_dir / "lightgbm_model.joblib")
    print(f"[OK] Saved model to: {out_dir / 'lightgbm_model.joblib'}")

    # Save plots
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_imp_df, x="Importance", y="Clean_Name", palette="mako")
    plt.title("Ames Housing: Top 15 Features Driving Property Prices (LightGBM)", fontsize=14, weight="bold")
    plt.xlabel("Importance Score (Number of splits)")
    plt.ylabel("House Feature")
    plt.tight_layout()
    plt.savefig(fig_dir / "member4_top_features.png", dpi=300)
    plt.savefig(rep_dir / "member4_top_features.png", dpi=300)
    plt.close()

    residuals = y - lgbm_dollar_preds
    plt.figure(figsize=(10, 5))
    plt.scatter(lgbm_dollar_preds, residuals, alpha=0.4, color="teal")
    plt.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    plt.title("Residual Error Analysis — LightGBM (Predicted vs Error)", fontsize=13, weight="bold")
    plt.xlabel("Predicted Sale Price ($)")
    plt.ylabel("Prediction Error ($)")
    plt.tight_layout()
    plt.savefig(fig_dir / "member4_residuals.png", dpi=300)
    plt.savefig(rep_dir / "member4_residuals.png", dpi=300)
    plt.close()

    # Save Metrics JSON
    metrics_record = {
        "member": "Member 4 - Raashidh M.R.A. (IT24104169)",
        "model_name": "LightGBM Regressor",
        "baseline_benchmark": {
            "rmse": b_rmse,
            "mae": b_mae,
            "r2": b_r2
        },
        "lgbm_performance": {
            "rmse": lgbm_rmse,
            "mae": lgbm_mae,
            "r2": lgbm_r2,
            "train_r2": train_r2,
            "overfit_gap": overfit_gap
        },
        "best_parameters": grid_search.best_params_,
        "top_5_features": feature_imp_df["Clean_Name"].head(5).tolist()
    }

    with open(results_dir / "member4_lgbm_metrics.json", "w") as f:
        json.dump(metrics_record, f, indent=4)
    with open(rep_dir / "member4_lgbm_metrics.json", "w") as f:
        json.dump(metrics_record, f, indent=4)
    print(f"[OK] Saved metrics to: {results_dir / 'member4_lgbm_metrics.json'}")

    # Generate Kaggle Submission CSV
    X_test = test_df.drop(columns=["Id"]).copy()
    test_dollar_preds = np.expm1(best_pipe.predict(X_test))
    sub_df = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": test_dollar_preds
    })
    sub_df.to_csv(rep_dir / "submission_lgbm.csv", index=False)
    print(f"[OK] Saved Kaggle submission to: {rep_dir / 'submission_lgbm.csv'}")


if __name__ == "__main__":
    main()
