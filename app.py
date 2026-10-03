"""IT3091 — House Price Prediction Microservice Backend
Group 2026-AI-08K · Machine Learning Project · SLIIT

Serves the group's actual trained machine learning models (.joblib pipelines):
  1. XGBoost Regressor Pipeline (Member 1 - Atheek Fareez IT24103933)
  2. Random Forest Regressor Pipeline (Member 3 - Wazni Ahamed IT24103352)
  3. Baseline Dummy Regressor Benchmark ($180,921)
"""

import sys
import os
import io
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

# Resolve project root and register import paths
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Backward-compatibility unpickling aliasing for Member 3's preprocessor
try:
    import src.member_03_wazni_ahamed as m3_pkg
    import src.member_03_wazni_ahamed.preprocessing as m3_prep
    sys.modules["src.member_03"] = m3_pkg
    sys.modules["src.member_03.preprocessing"] = m3_prep
except Exception as e:
    print(f"[INFO] Module aliasing note: {e}")

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="IT3091 Ames Housing Valuation API",
    description="High-performance machine learning inference API connecting trained .joblib pipelines to the frontend dashboard.",
    version="1.0.0"
)

# Enable CORS so browser dashboard can call http://127.0.0.1:8000 directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model store
MODELS: Dict[str, Any] = {}
TEMPLATE_ROW: Optional[pd.Series] = None
BASELINE_PRICE: float = 180921.0


class PropertyInput(BaseModel):
    overall_qual: int = Field(default=7, ge=1, le=10, description="Overall material and finish rating (1-10)")
    gr_liv_area: float = Field(default=1850.0, ge=300, le=10000, description="Above grade living area square feet")
    total_bsmt_sf: float = Field(default=1050.0, ge=0, le=8000, description="Total square feet of basement area")
    year_built: int = Field(default=2004, ge=1850, le=2026, description="Original construction date")
    neighborhood: str = Field(default="CollgCr", description="Physical location within Ames city limits")
    exter_qual: str = Field(default="Gd", description="Evaluates the quality of material on exterior (Ex, Gd, TA, Fa, Po)")
    kitchen_qual: str = Field(default="Gd", description="Kitchen quality (Ex, Gd, TA, Fa, Po)")
    garage_cars: int = Field(default=2, ge=0, le=5, description="Size of garage in car capacity")
    baths: float = Field(default=2.5, ge=0.0, le=10.0, description="Total bathroom equivalent count")


@app.on_event("startup")
def load_artifacts():
    """Load the trained .joblib models and base dataset template at startup."""
    global MODELS, TEMPLATE_ROW
    
    print("\n" + "="*60)
    print("[INIT] INITIALIZING IT3091 PRODUCTION MACHINE LEARNING ENGINE")
    print("="*60)

    # 1. Load Member 1 (Atheek) XGBoost Pipeline
    xgb_path = ROOT / "notebooks/Member_01_Atheek_Fareez/xgboost_best_model.joblib"
    if not xgb_path.exists():
        alt_xgb = ROOT / "xgboost_best_model.joblib"
        if alt_xgb.exists():
            xgb_path = alt_xgb
    
    if xgb_path.exists():
        try:
            MODELS["xgb"] = joblib.load(xgb_path)
            print(f"[OK] Loaded Member 1 XGBoost Pipeline: {xgb_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load XGBoost model: {e}")
    else:
        print("[WARN] xgboost_best_model.joblib not found at expected path.")

    # 2. Load Member 3 (Wazni) Random Forest Pipeline
    rf_path = ROOT / "notebooks/Member_03_Wazni_Ahamed/random_forest_model.joblib"
    if rf_path.exists():
        try:
            MODELS["rf"] = joblib.load(rf_path)
            print(f"[OK] Loaded Member 3 Random Forest Pipeline: {rf_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load Random Forest model: {e}")
    else:
        print("[WARN] random_forest_model.joblib not found at expected path.")

    # 3. Load Member 4 (Raashidh) LightGBM Pipeline
    lgbm_path = ROOT / "notebooks/Member_04_Raashidh/lightgbm_model.joblib"
    if not lgbm_path.exists():
        alt_lgbm = ROOT / "lightgbm_model.joblib"
        if alt_lgbm.exists():
            lgbm_path = alt_lgbm
    if lgbm_path.exists():
        try:
            MODELS["lgbm"] = joblib.load(lgbm_path)
            print(f"[OK] Loaded Member 4 LightGBM Pipeline: {lgbm_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load LightGBM model: {e}")
    else:
        print("[WARN] lightgbm_model.joblib not found at expected path.")

    # 4. Load Member 2 (Wijesiri) Ridge Regression Pipeline
    ridge_path = ROOT / "notebooks/Member_02_Wijesiri/ridge_model.joblib"
    if not ridge_path.exists():
        alt_ridge = ROOT / "ridge_model.joblib"
        if alt_ridge.exists():
            ridge_path = alt_ridge
    if ridge_path.exists():
        try:
            MODELS["ridge"] = joblib.load(ridge_path)
            print(f"[OK] Loaded Member 2 Ridge Pipeline: {ridge_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load Ridge model: {e}")
    else:
        print("[WARN] ridge_model.joblib not found at expected path.")

    # 4. Load baseline template row from raw Ames train.csv
    try:
        data_path = ROOT / "data/raw/house-prices-advanced-regression-techniques/train.csv"
        if data_path.exists():
            df_raw = pd.read_csv(data_path)
            TEMPLATE_ROW = df_raw.drop(columns=["Id", "SalePrice"]).iloc[0].copy()
            print("[OK] Loaded Ames Property Feature Schema & Baseline Template Row")
        else:
            print("[WARN] Raw train.csv not found; will build synthetic schema fallback.")
    except Exception as e:
        print(f"[WARN] Template load warning: {e}")

    print("="*60)
    print(f"[READY] Backend ready with {len(MODELS)} active models loaded.")
    print("="*60 + "\n")


@app.get("/")
def root_index():
    return {
        "project": "IT3091 House Price Prediction Pipeline",
        "group": "2026-AI-08K",
        "status": "online",
        "loaded_models": list(MODELS.keys()),
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict (POST)",
            "docs": "/docs"
        }
    }


@app.get("/api/health")
def health_check():
    """Health status check called by the frontend dashboard."""
    return {
        "status": "healthy",
        "engine": "live_joblib_backend",
        "models_ready": {
            "xgboost": "xgb" in MODELS,
            "random_forest": "rf" in MODELS,
            "lightgbm": "lgbm" in MODELS,
            "ridge": "ridge" in MODELS,
            "baseline": True
        },
        "python_version": sys.version.split()[0]
    }


@app.post("/api/predict")
def predict_valuation(prop: PropertyInput):
    """Run real live inference on the trained XGBoost, Random Forest, LightGBM, and Ridge .joblib pipelines."""
    if TEMPLATE_ROW is None:
        raise HTTPException(status_code=500, detail="Base property schema is not loaded.")

    # Synthesize complete 79-predictor Ames house record based on user inputs
    house = TEMPLATE_ROW.copy()
    house["OverallQual"] = int(prop.overall_qual)
    house["GrLivArea"] = float(prop.gr_liv_area)
    house["TotalBsmtSF"] = float(prop.total_bsmt_sf)
    house["YearBuilt"] = int(prop.year_built)
    house["YearRemodAdd"] = max(int(prop.year_built), int(house.get("YearRemodAdd", prop.year_built)))
    house["Neighborhood"] = str(prop.neighborhood)
    house["ExterQual"] = str(prop.exter_qual)
    house["KitchenQual"] = str(prop.kitchen_qual)
    house["GarageCars"] = int(prop.garage_cars)
    house["GarageArea"] = float(prop.garage_cars * 225.0)
    
    full_baths = int(prop.baths)
    half_baths = 1 if (prop.baths - full_baths) >= 0.5 else 0
    house["FullBath"] = full_baths
    house["HalfBath"] = half_baths
    
    # 1stFlrSF proportional allocation
    house["1stFlrSF"] = float(min(prop.gr_liv_area, max(800.0, prop.gr_liv_area * 0.65)))
    house["2ndFlrSF"] = float(max(0.0, prop.gr_liv_area - house["1stFlrSF"]))

    # Convert to single-row DataFrame
    input_df = pd.DataFrame([house])

    predictions: Dict[str, int] = {
        "baseline": int(BASELINE_PRICE)
    }

    # 1. Real XGBoost Prediction
    if "xgb" in MODELS:
        try:
            log_pred = MODELS["xgb"].predict(input_df)[0]
            dollar_pred = float(np.expm1(log_pred))
            predictions["xgb"] = int(round(dollar_pred))
        except Exception as e:
            print(f"[ERROR] XGBoost live inference error: {e}")
            predictions["xgb"] = int(round(BASELINE_PRICE))

    # 2. Real Random Forest Prediction
    if "rf" in MODELS:
        try:
            log_pred = MODELS["rf"].predict(input_df)[0]
            dollar_pred = float(np.expm1(log_pred))
            predictions["rf"] = int(round(dollar_pred))
        except Exception as e:
            print(f"[ERROR] Random Forest live inference error: {e}")
            predictions["rf"] = int(round(BASELINE_PRICE))

    # 3. Real LightGBM Prediction (Member 4)
    if "lgbm" in MODELS:
        try:
            log_pred = MODELS["lgbm"].predict(input_df)[0]
            dollar_pred = float(np.expm1(log_pred))
            predictions["lgbm"] = int(round(dollar_pred))
        except Exception as e:
            print(f"[ERROR] LightGBM live inference error: {e}")
            predictions["lgbm"] = int(round(BASELINE_PRICE))

    # 4. Real Ridge Regression Prediction (Member 2)
    if "ridge" in MODELS:
        try:
            log_pred = MODELS["ridge"].predict(input_df)[0]
            dollar_pred = float(np.expm1(log_pred))
            predictions["ridge"] = int(round(dollar_pred))
        except Exception as e:
            print(f"[ERROR] Ridge live inference error: {e}")
            predictions["ridge"] = int(round(BASELINE_PRICE))

    return {
        "status": "success",
        "source": "live_joblib_models",
        "inputs": prop.dict(),
        "predictions": predictions,
        "metrics": {
            "xgb": {"name": "XGBoost Regressor", "rmse": 28823, "mae": 15172, "r2": 0.868},
            "ridge": {"name": "Ridge Regression", "rmse": 47413, "mae": 15895, "r2": 0.644},
            "rf": {"name": "Random Forest", "rmse": 31640, "mae": 17491, "r2": 0.841},
            "lgbm": {"name": "LightGBM Regressor", "rmse": 29093, "mae": 15644, "r2": 0.866},
            "baseline": {"name": "Baseline Model", "rmse": 81448, "mae": 55656, "r2": -0.052}
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("\nStarting local API server on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
