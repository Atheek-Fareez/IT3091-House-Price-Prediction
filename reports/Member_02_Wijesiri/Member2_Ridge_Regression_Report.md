# IT3091 – MACHINE LEARNING (GROUP ASSIGNMENT)

## Member 2 Individual Technical Report
### Model 02: Regularized Linear Model (Ridge Regression) Pipeline & Valuation Analysis

**Group Identifier:** 2026-AI-08K  
**Project Title:** Ames Residential Property Valuation & Comparative Machine Learning  
**Target Variable:** `SalePrice` (Log-transformed: $\ln(1 + \text{SalePrice})$)  

| Student Name | Student ID | Group Role | Assigned Model |
| :--- | :--- | :--- | :--- |
| **Wijesiri S.P.R.H.** | **IT24100602** | Member 2 | Model 02: Ridge Regression (L2 Regularized Linear Model) |

---

## 1. Executive Summary & Objective

In this phase of the IT3091 Machine Learning project, Member 2 was assigned the formulation, cross-validation, hyperparameter tuning, and interpretability analysis of **Model 02: Ridge Regression** ($L_2$-penalized linear regression). 

While tree-based ensemble methods (such as Member 3's Random Forest, Member 4's LightGBM, and Member 1's XGBoost) excel at capturing non-linear interactions, **Ridge Regression serves as the foundational parametric regularized benchmark** for the team. Its primary advantages are:
1. **Mathematical Interpretability:** Coefficients provide direct, linear insight into directional price drivers (elasticities and unit premiums).
2. **Multicollinearity Control:** Ames housing records exhibit substantial feature collinearity (e.g., `GrLivArea` vs `TotRmsAbvGrd`, `GarageCars` vs `GarageArea`). The $L_2$ quadratic penalty shrinks correlated weights proportionally without arbitrary zeroing, maintaining numeric stability.
3. **Generalization Guarantee:** By penalizing large weight vectors, Ridge mitigates the severe overfitting that ordinary least squares (OLS) experiences in high-dimensional feature spaces (100+ one-hot encoded categories).

All experiments were executed under the group's unified evaluation contract: 5-fold cross-validation (`random_state=42`) with strict leakage-free encapsulation of Member 3's preprocessing pipeline (`configuration='scaled', log_numeric=True`).

---

## 2. Mathematical Formulation & Architecture

The standard Ordinary Least Squares (OLS) objective minimizes the residual sum of squares:
$$\min_{\mathbf{w}} \sum_{i=1}^n \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2$$

When features are correlated or high-dimensional, the matrix $\mathbf{X}^T \mathbf{X}$ becomes ill-conditioned, leading to exploding coefficient variances. Ridge Regression addresses this by introducing an $L_2$ regularization penalty:
$$\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2n} \sum_{i=1}^n \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2 + \frac{\alpha}{2} \|\mathbf{w}\|_2^2 = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \frac{\alpha}{2} \sum_{j=1}^p w_j^2$$

The closed-form analytical solution is:
$$\hat{\mathbf{w}}_{\text{Ridge}} = \left( \mathbf{X}^T \mathbf{X} + n\alpha \mathbf{I} \right)^{-1} \mathbf{X}^T \mathbf{y}$$

### Preprocessing and Scaling Rationale
Because the $L_2$ penalty adds $\alpha \sum w_j^2$, features with larger numeric scales would be unfairly penalized more heavily than features with small numerical ranges. Therefore, Member 2's pipeline explicitly requires:
- **`configuration='scaled'`:** Standardizes all continuous features ($\mu=0, \sigma=1$) using `StandardScaler` fitted strictly on each training fold.
- **`log_numeric=True`:** Applies `np.log1p` to skewed continuous area variables (`LotArea`, `1stFlrSF`, `GrLivArea`) to compress long tails and linearize relationships with log-sale price.
- **`sparse_output=False`:** Delivers dense arrays for compatibility with Ridge solvers.

---

## 3. Academic Baseline Benchmark (SLIIT Rubric Alignment)

In strict adherence to the SLIIT assessment criteria, Member 2 first established the **Academic Baseline Benchmark** using a `DummyRegressor(strategy='mean')`. This models a naive appraiser who simply predicts the training sample mean price for every house in Ames:

| Metric | Academic Baseline (Mean Guess) | Member 2 Target |
| :--- | :--- | :--- |
| **CV RMSE ($)** | **$80,701.61** | Substantial reduction (< $50,000) |
| **CV MAE ($)** | **$55,655.37** | Substantial reduction (< $20,000) |
| **Dollar $R^2$** | **-0.0327** | Positive explanatory power (> 0.60) |

The baseline benchmark demonstrates that an uninformed guess produces an average dollar error of **$55,655.37**, confirming the necessity of supervised machine learning modeling.

---

## 4. Hyperparameter Tuning & Cross-Validation Results

The regularization parameter $\alpha$ controls the balance between model complexity and bias. A systematic grid of $\alpha \in [0.1, 1.0, 10.0, 50.0, 100.0]$ was evaluated across identical 5-fold cross-validation partitions:

| Regularization $\alpha$ | CV RMSE ($) | CV MAE ($) | Dollar $R^2$ | Train $R^2$ (Log) | Val $R^2$ (Log) | Overfitting Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.1 (Selected)** | **$47,412.74** | **$15,894.80** | **0.6436** | **0.9457** | **0.8703** | **0.0754** |
| 1.0 | $47,936.20 | $16,027.68 | 0.6356 | 0.9414 | 0.8753 | 0.0662 |
| 10.0 | $49,966.05 | $16,101.71 | 0.6041 | 0.9320 | 0.8752 | 0.0568 |
| 50.0 | $51,322.19 | $16,460.36 | 0.5824 | 0.9197 | 0.8725 | 0.0472 |
| 100.0 | $51,342.64 | $16,764.61 | 0.5820 | 0.9126 | 0.8703 | 0.0423 |

### Tuning Interpretation & Selection
1. **Best Dollar Error:** $\alpha = 0.1$ achieved the lowest out-of-fold RMSE (**$47,412.74**) and lowest MAE (**$15,894.80**), while maximizing Dollar $R^2$ at **0.6436**.
2. **Log-Space Generalization:** On the log scale, validation $R^2$ remained exceptionally steady between **0.8703 and 0.8753** across all $\alpha$ choices, demonstrating that the underlying linear fit is stable.
3. **Overfitting Analysis:** The gap between training $R^2$ (0.9457) and validation $R^2$ (0.8703) is **0.0754**. While higher $\alpha$ values further shrink the gap (to 0.0423 at $\alpha=100$), they introduce excess bias that penalizes dollar-scale accuracy. Thus, **$\alpha = 0.1$** was chosen as the optimal operational parameter.

---

## 5. Secondary Decision Lens: Linear Valuation Price Drivers

A primary motivation for deploying Ridge Regression is linear coefficient transparency. In log-linear models ($\ln y = \mathbf{w}^T \mathbf{x}$), a unit increase in a standardized feature approximately equates to a proportional percentage change in property value.

### Top Positive Price Drivers (Premiums)
1. **Above Ground Living Area (`GrLivArea`):** Standardized coefficient $+0.114$. Living area remains the primary continuous physical determinant of residential appraisal value.
2. **Overall Quality (`OverallQual`):** Standardized coefficient $+0.098$. Finish craftsmanship and building grade yield substantial market premiums.
3. **Neighborhood - Stone Brook (`StoneBr`) & Northridge Heights (`NridgHt`):** Neighborhood indicator coefficients $+0.062$ and $+0.058$. Affirms the real estate adage that location commands premium capitalization.
4. **Year Built (`YearBuilt`):** Standardized coefficient $+0.054$. Modern structural construction age reduces expected maintenance depreciation.
5. **Total Basement Square Footage (`TotalBsmtSF`):** Standardized coefficient $+0.049$. Usable sub-grade foundation space contributes significantly to valuation.

### Top Negative Price Drivers (Depreciations)
1. **Zoning - Commercial (`C (all)`):** Negative coefficient $-0.082$. Residential dwellings adjacent to or zoned for commercial usage face steep appraisal discounts.
2. **Proximity to Adverse Conditions (`Artery`):** Negative coefficient $-0.038$. Heavy arterial street noise and traffic impede residential appeal.
3. **Property Age / Functional Obsolescence:** Older structures without recent renovations suffer noticeable negative coefficient dampening.

---

## 6. Diagnostic Evaluation & Limitations Analysis

### Why Dollar RMSE ($47,412) vs MAE ($15,895)?
A critical finding in Member 2's evaluation is the disparity between MAE ($15,894.80) and RMSE ($47,412.74).
- **The Exponential Inversion Effect:** In log-linear regression, predictions are generated in log space and transformed back via $\hat{y} = e^{\hat{z}} - 1$. For typical homes (\$100,000–\$300,000), predictions are tightly clustered with median errors below 9%.
- **Extreme Luxury Residuals:** A small number of atypical luxury estates (e.g., homes selling above \$600,000 or unusual farm lots) receive slightly elevated log predictions that, when exponentiated, yield large residual discrepancies. Because RMSE squares individual errors, these few outliers heavily weight the aggregate metric, whereas the robust MAE accurately reflects typical day-to-day valuation accuracy.

---

## 7. Artifacts Summary & Deliverables

All required pipeline deliverables for Member 2 have been produced, verified, and committed to their designated repository locations:

| Deliverable | File Path | Status | Verification Detail |
| :--- | :--- | :---: | :--- |
| **Jupyter Notebook** | `notebooks/Member_02_Wijesiri/Member2_Ridge_Regression.ipynb` | ✅ Verified | 8-step executable notebook with outputs & embedded plots |
| **Pipeline Script** | `src/member_02_wijesiri/it24100602_ridge_model_pipeline.py` | ✅ Verified | Python CLI script with automated CV & evaluation |
| **Serialized Model** | `notebooks/Member_02_Wijesiri/ridge_model.joblib` | ✅ Verified | Serialized `Pipeline` ready for FastAPI microservice |
| **Metrics Record** | `reports/model_results/member2_ridge_metrics.json` | ✅ Verified | JSON record with CV RMSE, MAE, R², and Overfit Gap |
| **Kaggle Submission** | `reports/Member_02_Wijesiri/submission_ridge.csv` | ✅ Verified | 1,459 unseen test predictions formatted for Kaggle |
| **Feature Plot** | `reports/figures/member2_top_features.png` | ✅ Verified | 300 DPI horizontal bar chart of top Ridge coefficients |
| **Residual Plot** | `reports/figures/member2_residuals.png` | ✅ Verified | 300 DPI scatter plot of predicted prices vs dollar residuals |
