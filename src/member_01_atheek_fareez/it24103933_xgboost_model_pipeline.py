"""
IT3091 Machine Learning Project
Member 1: Atheek M. F (IT24103933)
Model: XGBoost Regressor Pipeline with 5-Fold Cross-Validation & Hyperparameter Tuning
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn tools for model evaluation
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Model algorithm & file saver
from xgboost import XGBRegressor
import joblib

# Resolve project root dynamically
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Backward-compatibility aliasing for Member 3
try:
    import src.member_03_wazni_ahamed as m3_pkg
    import src.member_03_wazni_ahamed.preprocessing as m3_prep
    sys.modules["src.member_03"] = m3_pkg
    sys.modules["src.member_03.preprocessing"] = m3_prep
except Exception:
    pass

from src.member_03.preprocessing import load_ames_data, build_preprocessor


def main():
    print("=" * 60)
    print("[INIT] IT3091 MEMBER 1 (ATHEEK) XGBOOST PIPELINE")
    print(f"[INIT] Repository root: {ROOT}")
    print("=" * 60)

    # 1. Load Data
    train_df, test_df = load_ames_data(ROOT)
    print(f"[DATA] Train: {train_df.shape[0]} rows, {train_df.shape[1]} cols")
    print(f"[DATA] Test:  {test_df.shape[0]} rows, {test_df.shape[1]} cols")

    X = train_df.drop(columns=["Id", "SalePrice"]).copy()
    y = train_df["SalePrice"].astype(float).copy()
    y_log = np.log1p(y)

    RANDOM_STATE = 42
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 2. Academic Baseline (Median Guess)
    baseline_model = DummyRegressor(strategy="median")
    baseline_log_preds = cross_val_predict(baseline_model, X, y_log, cv=cv)
    baseline_dollar_preds = np.expm1(baseline_log_preds)

    baseline_rmse = float(np.sqrt(mean_squared_error(y, baseline_dollar_preds)))
    baseline_mae = float(mean_absolute_error(y, baseline_dollar_preds))
    baseline_r2 = float(r2_score(y, baseline_dollar_preds))

    print("\n--- BASELINE BENCHMARK ---")
    print(f"Baseline RMSE: ${baseline_rmse:,.2f}")
    print(f"Baseline MAE:  ${baseline_mae:,.2f}")
    print(f"Baseline R2:   {baseline_r2:.4f}")

    # 3. Build Pipeline & Hyperparameter Tuning
    preprocessor = build_preprocessor(configuration="unscaled", log_numeric=False, sparse_output=False)
    xgb_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    param_grid = {
        "regressor__n_estimators": [200, 300],
        "regressor__learning_rate": [0.03, 0.05],
        "regressor__max_depth": [3, 4, 5],
        "regressor__subsample": [0.8, 1.0],
        "regressor__colsample_bytree": [0.7, 0.8]
    }

    print("\n[TUNING] Running GridSearchCV with 5-fold CV...")
    grid_search = GridSearchCV(
        estimator=xgb_pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X, y_log)

    best_xgb_pipeline = grid_search.best_estimator_
    print(f"[OK] Best CV Log-RMSE: {-grid_search.best_score_:.4f}")
    print(f"[OK] Best Params: {grid_search.best_params_}")

    # 4. Out-of-Fold Evaluation
    xgb_log_preds = cross_val_predict(best_xgb_pipeline, X, y_log, cv=cv, n_jobs=-1)
    xgb_dollar_preds = np.expm1(xgb_log_preds)

    xgb_rmse = float(np.sqrt(mean_squared_error(y, xgb_dollar_preds)))
    xgb_mae = float(mean_absolute_error(y, xgb_dollar_preds))
    xgb_r2 = float(r2_score(y, xgb_dollar_preds))

    cv_scores = cross_validate(
        best_xgb_pipeline, X, y_log,
        cv=cv,
        scoring="r2",
        return_train_score=True,
        n_jobs=-1
    )
    train_r2 = float(np.mean(cv_scores["train_score"]))
    val_r2 = float(np.mean(cv_scores["test_score"]))
    overfit_gap = float(train_r2 - val_r2)

    print("\n--- FINAL XGBOOST PERFORMANCE ---")
    print(f"XGBoost RMSE:    ${xgb_rmse:,.2f}")
    print(f"XGBoost MAE:     ${xgb_mae:,.2f}")
    print(f"XGBoost CV R2:   {xgb_r2:.4f}")
    print(f"Train R2:        {train_r2:.4f}")
    print(f"Validation R2:   {val_r2:.4f}")
    print(f"Overfitting Gap: {overfit_gap:.4f}")

    # 5. Extract Feature Importance
    raw_feature_names = best_xgb_pipeline.named_steps["preprocessor"].get_feature_names_out()
    clean_feature_names = [f.split("__")[-1] for f in raw_feature_names]
    importances = best_xgb_pipeline.named_steps["regressor"].feature_importances_

    feature_imp_df = pd.DataFrame({
        "Feature": raw_feature_names,
        "Clean_Name": clean_feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # 6. Save Plots
    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    m1_rep_dir = ROOT / "reports/Member_01_Atheek_Fareez"
    m1_rep_dir.mkdir(parents=True, exist_ok=True)

    # Top Features Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_imp_df.head(15), x="Importance", y="Clean_Name", palette="Blues_r")
    plt.title("Ames Housing: Top 15 Features Driving Property Prices (XGBoost)", fontsize=14, weight="bold")
    plt.xlabel("Importance Score (Gain / Weight)")
    plt.ylabel("House Feature")
    plt.tight_layout()
    plt.savefig(fig_dir / "member1_top_features.png", dpi=300)
    plt.savefig(m1_rep_dir / "member1_top_features.png", dpi=300)
    plt.close()

    # Residuals Plot
    residuals = y - xgb_dollar_preds
    plt.figure(figsize=(10, 5))
    plt.scatter(xgb_dollar_preds, residuals, alpha=0.4, color="royalblue")
    plt.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    plt.title("XGBoost Residual Error Analysis (Predicted Price vs Residual Error)", fontsize=13, weight="bold")
    plt.xlabel("Predicted Sale Price ($)")
    plt.ylabel("Prediction Error ($: Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(fig_dir / "member1_residuals.png", dpi=300)
    plt.savefig(m1_rep_dir / "member1_residuals.png", dpi=300)
    plt.close()
    print("[OK] Saved plots to reports/figures/ and reports/Member_01_Atheek_Fareez/")

    # 7. Save Model Artifacts
    out_dir = ROOT / "notebooks/Member_01_Atheek_Fareez"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_xgb_pipeline, out_dir / "xgboost_best_model.joblib")
    print(f"[OK] Saved model to: {out_dir / 'xgboost_best_model.joblib'}")

    # 8. Save Metrics
    metrics_record = {
        "member": "Member 1 - Atheek M. F (IT24103933)",
        "model_name": "XGBoost Regressor",
        "baseline_benchmark": {
            "rmse": baseline_rmse,
            "mae": baseline_mae,
            "r2": baseline_r2
        },
        "xgboost_performance": {
            "rmse": xgb_rmse,
            "mae": xgb_mae,
            "r2": xgb_r2,
            "train_r2": train_r2,
            "overfit_gap": overfit_gap
        },
        "best_parameters": grid_search.best_params_,
        "top_5_features": feature_imp_df["Clean_Name"].head(5).tolist()
    }

    results_dir = ROOT / "reports/model_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "member1_xgb_metrics.json", "w") as f:
        json.dump(metrics_record, f, indent=4)
    with open(m1_rep_dir / "member1_xgb_metrics.json", "w") as f:
        json.dump(metrics_record, f, indent=4)
    print(f"[OK] Saved metrics to: {results_dir / 'member1_xgb_metrics.json'}")

    # 9. Generate Kaggle Test Set Predictions
    X_test = test_df.drop(columns=["Id"]).copy()
    test_dollar_predictions = np.expm1(best_xgb_pipeline.predict(X_test))
    sub_df = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": test_dollar_predictions
    })
    sub_df.to_csv(m1_rep_dir / "submission_xgb.csv", index=False)
    print(f"[OK] Saved Kaggle submission to: {m1_rep_dir / 'submission_xgb.csv'}")


if __name__ == "__main__":
    main()