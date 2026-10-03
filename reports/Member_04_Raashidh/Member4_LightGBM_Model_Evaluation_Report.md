# IT3091 – MACHINE LEARNING (GROUP ASSIGNMENT)

## Member 4 Individual Technical Report
### Model 03: LightGBM Regressor Pipeline, Model Comparison Leadership & Valuation Analysis

**Group Identifier:** 2026-AI-08K  
**Project Title:** Ames Residential Property Valuation & Comparative Machine Learning  
**Target Variable:** `SalePrice` (Log-transformed: $\ln(1 + \text{SalePrice})$)  

| Student Name | Student ID | Group Role | Assigned Responsibilities |
| :--- | :--- | :--- | :--- |
| **Raashidh M.R.A.** | **IT24104169** | Member 4 | Model 03: LightGBM Regressor + Model Comparison Co-Lead + Valuation Feature Analysis |

---

## 1. Executive Summary & Role Allocation

Within Group 2026-AI-08K's collaborative machine learning workflow, Member 4 was entrusted with three core engineering and analytical responsibilities:
1. **Model Architecture 03 (LightGBM):** Design, cross-validate, and optimize a high-efficiency **Light Gradient Boosting Machine (LightGBM)** regressor for residential valuation.
2. **Model Comparison Leadership:** Co-lead the comparative cross-validation evaluation alongside Member 1 (Lead), synthesising results across the Academic Baseline, Ridge Regression (Member 2), Random Forest (Member 3), LightGBM (Member 4), and XGBoost (Member 1).
3. **Secondary Valuation Decision Lens:** Analyze tree split frequencies to isolate the Top 15 physical and geographical property attributes determining market value in Ames, Iowa.

All modeling adhered strictly to the team's unified evaluation contract: 5-fold cross-validation (`random_state=42`), log-scale training ($\ln(1 + \text{SalePrice})$), and leak-free nesting of Member 3's preprocessor (`configuration='unscaled', log_numeric=False`).

---

## 2. Mathematical Foundation & Algorithmic Advantages of LightGBM

LightGBM is a high-performance gradient boosting framework based on decision tree algorithms, distinguished by two fundamental innovations:

### 2.1 Leaf-Wise (Best-First) Tree Growth vs Level-Wise Growth
Traditional gradient boosting algorithms (such as standard implementations of XGBoost) grow trees level-wise (depth-first), splitting all nodes at a given depth simultaneously. In contrast, LightGBM chooses the leaf that minimizes the global loss function:
$$\Delta \mathcal{L} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
Where $G_L, G_R$ and $H_L, H_R$ represent the first and second-order gradient statistics of the left and right splits. By growing leaf-wise, LightGBM achieves lower loss with fewer splits, accelerating convergence.

### 2.2 Histogram-Based Feature Binning
Rather than sorting continuous feature values at every split ($O(n \cdot p)$ complexity), LightGBM discretizes continuous variables into $K$ discrete bins (typically $K=255$). This reduces split-finding memory consumption and computation to $O(K \cdot p)$, enabling near-instantaneous training across 5 folds.

### 2.3 Preprocessing Configuration Rationale
Tree-based partition algorithms determine splits strictly based on ordinal order:
$$\{x_j \le \theta\} \quad \text{vs} \quad \{x_j > \theta\}$$
Monotonic transformations (such as z-score standardization) do not alter the ordering of feature values. Therefore, Member 4 utilized `build_preprocessor(configuration='unscaled', log_numeric=False)`, avoiding unnecessary matrix floating-point computations while preserving native feature scale interpretability.

---

## 3. Academic Baseline Benchmark (SLIIT Rubric Compliance)

To verify that supervised learning delivers demonstrable predictive utility over naive estimation, Member 4 evaluated an Academic Baseline (`DummyRegressor(strategy='mean')`) across the 5 cross-validation folds:

| Metric | Academic Baseline (Mean Guess) | Tuned LightGBM Model | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **CV RMSE ($)** | **$80,701.61** | **$29,092.78** | **63.95% Error Reduction** 📉 |
| **CV MAE ($)** | **$55,655.37** | **$15,643.71** | **71.89% Error Reduction** 📉 |
| **Dollar $R^2$** | **-0.0327** | **0.8658** | **+0.8985 Explanatory Shift** 📈 |
| **Log Validation $R^2$** | -0.0084 | **0.8911** | High-precision log fit |

The benchmark establishes that a naive guess incurs an average dollar error of **$55,655.37**. LightGBM shrinks this error to **$15,643.71**, demonstrating genuine machine learning efficacy.

---

## 4. Hyperparameter Optimization & Tuning Strategy

Hyperparameter tuning was conducted using `GridSearchCV` nested inside 5-fold cross-validation. The search explored the trade-off between model capacity (`num_leaves`), learning rate, and tree depth:

| Hyperparameter | Search Space | Optimal Selected Value | Engineering Rationale |
| :--- | :--- | :---: | :--- |
| `n_estimators` | `[200, 300]` | **300** | Sufficient ensemble rounds for gradual gradient descent |
| `learning_rate` | `[0.03, 0.05]` | **0.03** | Smaller shrinkage step size prevents overshooting optima |
| `num_leaves` | `[20, 31, 40]` | **31** | Constrains tree complexity to avoid overfitting small subsets |
| `max_depth` | `[4, 6, -1]` | **6** | Caps leaf-wise growth depth, curbing extreme branch paths |
| `colsample_bytree` | `[0.7, 0.8]` | **0.7** | Feature subsampling decorrelates individual trees |

### Overfitting & Generalization Analysis
- **Log Training $R^2$:** 0.9855
- **Log Validation $R^2$:** 0.8911
- **Overfitting Gap:** **0.0945** (Under 0.10, indicating balanced variance without pathological memorization).

---

## 5. Secondary Valuation Decision Lens: Physical & Locational Drivers

By extracting the `feature_importances_` (split counts) from the fitted LightGBM pipeline, Member 4 answered the group's secondary research objective: *Which architectural and environmental features drive property values in Ames, Iowa?*

### Top 10 Valuation Drivers Identified by LightGBM
1. **Above Ground Living Area (`GrLivArea`):** 1,248 splits. The single most influential continuous price predictor.
2. **Overall Material and Finish Quality (`OverallQual`):** 982 splits. High-grade craftsmanship commands exponential premiums.
3. **Total Basement Square Footage (`TotalBsmtSF`):** 741 splits. Sub-grade usable volume significantly augments property utility.
4. **Lot Area (`LotArea`):** 689 splits. Parcel dimensions provide baseline land value.
5. **Year Built (`YearBuilt`) & Year Remodeled (`YearRemodAdd`):** Structural vintage reflects modern insulation, wiring, and reduced deferred maintenance.
6. **1st Floor Square Footage (`1stFlrSF`):** Primary ground-level footprint area.
7. **Garage Area (`GarageArea`) & Garage Cars (`GarageCars`):** Suburban vehicular accommodation convenience.
8. **Neighborhood Categorization:** Premium micro-markets (`Northridge Heights`, `Stone Brook`, `Crawford`) exhibit distinct positive split thresholds.

---

## 6. Official Master Model Comparison Table (All 4 Members)

As comparative evaluation co-lead, Member 4 produced the master cross-validation benchmark table across all 4 student models and the academic baseline on identical 5 folds:

| Model Architecture | Team Member & ID | CV RMSE ($) | CV MAE ($) | Dollar $R^2$ | Log Val $R^2$ | Overfit Gap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Academic Baseline** | University Benchmark | $81,448.16 | $55,655.52 | -0.0519 | -0.0084 | 0.0047 |
| **Ridge Regression** | Wijesiri S.P.R.H. (IT24100602) | $47,412.74 | $15,894.80 | 0.6436 | 0.8703 | 0.0754 |
| **Random Forest** | Ahamed M.A.W. (IT24103352) | $31,640.11 | $17,490.80 | 0.8413 | 0.8659 | 0.1165 |
| **LightGBM Regressor** | **Raashidh M.R.A. (IT24104169)** | **$29,092.78** | **$15,643.71** | **0.8658** | **0.8911** | **0.0945** |
| **XGBoost Regressor** | Atheek M. F. (IT24103933) 🏆 | **$28,822.61** | **$15,171.92** | **0.8683** | **0.8948** | **0.0896** |

### Synthesis & Production Recommendation
1. **Top Predictive Accuracy:** XGBoost and LightGBM form the elite predictive tier, separated by less than $270 RMSE.
2. **Speed & Scalability Advantage:** LightGBM completed 5-fold cross-validation in **12.4 seconds**, compared to 36.8 seconds for XGBoost and 42.1 seconds for Random Forest. For large-scale real-time retraining pipelines, LightGBM is the superior industrial choice.
3. **Ensemble Diversity:** In the FastAPI microservice (`app.py`), all four models are actively loaded, allowing users to contrast linear interpretability (Ridge) with bagging robustness (Random Forest) and boosting precision (XGBoost/LightGBM).

---

## 7. Deliverables & Repository Verification

All assigned artifacts for Member 4 have been generated, validated, and placed in their respective project folders:

| Artifact | Repository Location | Status |
| :--- | :--- | :---: |
| **LightGBM Training Notebook** | `notebooks/Member_04_Raashidh/Member4_LightGBM.ipynb` | ✅ Executed & Verified |
| **Master Comparison Notebook** | `notebooks/Member_04_Raashidh/Final_Model_Comparison_and_Selection.ipynb` | ✅ Executed & Verified |
| **Python Pipeline Script** | `src/member_04_raashidh/it24104169_lightgbm_model_pipeline.py` | ✅ Verified CLI script |
| **Serialized Model** | `notebooks/Member_04_Raashidh/lightgbm_model.joblib` | ✅ Active in FastAPI |
| **Metrics Record** | `reports/model_results/member4_lgbm_metrics.json` | ✅ Exported JSON |
| **Master Comparison CSV** | `reports/model_results/master_model_comparison_table.csv` | ✅ Consolidated Benchmark |
| **Kaggle Test Predictions** | `reports/Member_04_Raashidh/submission_lgbm.csv` | ✅ 1,459 Unseen Predictions |
| **Feature Importance Chart** | `reports/figures/member4_top_features.png` | ✅ High-res visualization |
| **Residual Error Chart** | `reports/figures/member4_residuals.png` | ✅ Error dispersion plot |
| **Master Comparison Chart** | `reports/figures/master_model_comparison.png` | ✅ Multi-model benchmark chart |
