# 🚀 Project Startup & Execution Guide
## IT3091 — Machine Learning Project · Group 2026-AI-08K
### Ames Iowa Residential Property Valuation Platform

---

## 👥 Group Members & Allocations

| Member | Student ID | Core Assignment Scope | Production Deliverable |
| :--- | :---: | :--- | :--- |
| **Atheek Fareez** | `IT24103933` | Model 01 & Hyperparameter Tuning | `xgboost_best_model.joblib` (Champion Model) |
| **Wijesiri H.P.N.K.** | `IT24100602` | Exploratory Data Analysis & Quality | Statistical Distributions & Outlier Checks |
| **Wazni Ahamed** | `IT24103352` | Preprocessing Pipeline & Model 02 | `random_forest_model.joblib` & Transform Pipeline |
| **Raashidh M.R.A.** | `IT24104169` | Model 03 Evaluation & Documentation | Upcoming Model Evaluation & Project Reporting |

---

## ⚡ Quick Start (Under 1 Minute)

### Step 1: Start the Python Backend (`app.py`)
Choose **either** Method A or Method B:

* **Method A (1-Click Shortcut - Windows):**  
  Double-click **`run_api.bat`** in the project root folder.

* **Method B (Terminal / Command Line):**  
  Open your terminal in the project root folder and run:
  ```powershell
  # Using the project's pre-configured virtual environment:
  .\.venv\Scripts\python.exe app.py
  ```
  *(Or on macOS / Linux: `./.venv/bin/python app.py`)*

You will see:
```text
============================================================
[INIT] INITIALIZING IT3091 PRODUCTION MACHINE LEARNING ENGINE
============================================================
[OK] Loaded Member 1 XGBoost Pipeline: xgboost_best_model.joblib
[OK] Loaded Member 3 Random Forest Pipeline: random_forest_model.joblib
[OK] Loaded Ames Property Feature Schema & Baseline Template Row
============================================================
[READY] Backend ready with 2 active models loaded.
============================================================
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

### Step 2: Open the Frontend Dashboard
Navigate to `docs/dashboard/` and double-click:
👉 **`docs/dashboard/index.html`**

*(Or right-click `index.html` ➔ "Open with" ➔ Google Chrome / Microsoft Edge / Brave / Firefox).*

---

### Step 3: Verify Live Model Connection
1. In your browser, look at the **top-right header**. You will see:
   `🟢 API: Live .joblib Connected`
2. Click the 8th tab: **`🏡 Valuation Simulator`**.
3. The status badge will display:
   `🟢 Live Python .joblib API Connected (Port 8000)`
4. Move any slider (*Overall Quality, Living Area, Year Built, etc.*) or click preset buttons (*Starter Home, Suburban Family, Luxury Estate*).
5. Watch the dashboard execute **real-time live inference** across Member 1's XGBoost pipeline, Member 3's Random Forest pipeline, and the baseline benchmark!

---

## 🛠️ Fresh Installation & Environment Setup (For Other Group Members)

If a team member clones this repository onto a new laptop, follow these steps to set up the environment from scratch:

### 1. Ensure Python is Installed
Python **3.10, 3.11, or 3.12** is required (Python 3.11 recommended).  
Verify in your terminal:
```powershell
python --version
```

### 2. Create the Virtual Environment
In the repository root folder:
```powershell
python -m venv .venv
```

### 3. Activate the Environment
* **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an execution policy warning, run once: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

* **Windows (Command Prompt):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 4. Install Required Libraries
Install the pinned production backend requirements:
```powershell
pip install -r requirements-api.txt
```
*(Packages installed: `fastapi`, `uvicorn`, `scikit-learn==1.6.1`, `xgboost`, `pandas`, `numpy`, `joblib`, `pydantic`).*

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│               FRONTEND USER INTERFACE                  │
│               docs/dashboard/index.html                │
│                                                        │
│  - 9 Interactive Tabs (EDA, Prep, Metrics, Simulator)  │
│  - Pure HTML5 / CSS3 (Nexora Velvet Crimson Theme)     │
│  - Chart.js Visualizations                             │
│  - Auto-polling Backend Health Check every 4 seconds   │
└───────────────────────────┬────────────────────────────┘
                            │
              HTTP POST /api/predict (JSON)
              HTTP GET  /api/health
                            │
┌───────────────────────────▼────────────────────────────┐
│                BACKEND MICROSERVICE                    │
│                       app.py                           │
│              FastAPI · Port 8000 · Uvicorn             │
├────────────────────────────────────────────────────────┤
│  1. Ingests 9 Core Property Inputs                     │
│  2. Populates 79 Ames Predictor Record                 │
│  3. Member 03 Preprocessing Pipeline Transformation    │
│  4. Member 01 XGBoost Inference (model.predict)        │
│  5. Member 03 Random Forest Inference                  │
│  6. Inverse Log Transform: np.expm1(log_price)         │
│  7. Returns JSON with Dollar Values & Benchmark Deltas │
└────────────────────────────────────────────────────────┘
```

---

## 🛡️ Dual-Mode Resilience (Frontend Fallback)

The frontend dashboard is designed to **never break or show errors**:
* **Online Mode (`app.py` running):**  
  Badge shows `🟢 Live Python .joblib API Connected`.  
  Predictions come directly from the serialized Python binary models with the label `[Live .joblib Output]`.
* **Offline Mode (`app.py` stopped):**  
  Badge shows `⚡ Client Simulator · (Run python app.py for Live .joblib API)`.  
  The dashboard automatically falls back to an embedded client calculation engine, allowing team members and evaluators to explore all 9 tabs even on machines without Python installed!

---

## 🔍 Interactive API Documentation (Swagger UI)

When `app.py` is running, you can test the REST endpoints directly in your browser:
* **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Alternative Redoc UI:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* **Health Endpoint:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 📌 Guide for Team Members (Extending the Project)

### For Member 4 (Adding Model 03)
When Member 4 completes their model:
1. Export the fitted pipeline from notebook/script:
   ```python
   import joblib
   joblib.dump(pipeline, "notebooks/Member_04_Raashidh/model_03.joblib")
   ```
2. Open `app.py` and register the model in `load_artifacts()`:
   ```python
   MODELS["m4"] = joblib.load(ROOT / "notebooks/Member_04_Raashidh/model_03.joblib")
   ```
3. Add the prediction call inside `predict_valuation()` in `app.py`.
4. The dashboard will automatically reflect Model 4 across the multi-model comparison table!

---

## ❓ Troubleshooting & FAQs

### Q1: The badge says "Offline (Client Simulator)". How do I make it green?
**Answer:** Start the backend server by double-clicking `run_api.bat` or executing `.\.venv\Scripts\python.exe app.py`. Keep that terminal window open while using the browser dashboard.

### Q2: Port 8000 is already in use by another application.
**Answer:** In `app.py`, change line 236 from `port=8000` to `port=8001`, and in `docs/dashboard/index.html` change `8000` to `8001`.

### Q3: PowerShell gives an execution policy error when activating `.venv`.
**Answer:** Run this command once in PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Q4: Can I run this without an internet connection?
**Answer:** Yes! The backend server runs completely offline on your `localhost` (`127.0.0.1`). If you don't have internet access for the CDN icons/fonts, the core calculations and layout still operate normally.
