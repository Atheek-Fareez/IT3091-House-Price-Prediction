# IT3091 – MACHINE LEARNING (INDIVIDUAL REPORT)

## Section 6: Personal Learning Journey & Contribution Report
### Student: Raashidh M.R.A. · IT24104169 (Member 4)

**Module:** IT3091 Machine Learning  
**Academic Year:** 3rd Year, 1st Semester  
**Group:** 2026-AI-08K  
**Project:** Ames House Price Prediction & Real Estate Valuation Platform  

---

## 1. Individual Project Contributions

Throughout the life cycle of this machine learning project, my core individual contributions spanned both specialized model development and overarching team coordination:

1. **Model Architecture 03 (LightGBM Development):**  
   I designed, tuned, and validated the **Light Gradient Boosting Machine (LightGBM)** pipeline (`Member4_LightGBM.ipynb`). By implementing histogram-based feature quantization and leaf-wise tree growth, I achieved a cross-validated RMSE of **$29,092.78** and an $R^2$ of **0.8658**, demonstrating that gradient boosting can achieve state-of-the-art predictive accuracy with superior computational speed.

2. **Master Model Comparison & Evaluation Leadership:**  
   In collaboration with Member 1, I co-led the comparative evaluation phase (`Final_Model_Comparison_and_Selection.ipynb`). I formulated the standardized 5-fold cross-validation race that compared the Academic Baseline, Member 2's Ridge Regression, Member 3's Random Forest, my LightGBM, and Member 1's XGBoost on identical data splits. This produced the group's official benchmark table (`master_model_comparison_table.csv`).

3. **Secondary Valuation Decision Lens Analysis:**  
   I extracted split-frequency feature importances from LightGBM to identify the top 15 residential value drivers in Ames, Iowa, providing actionable domain insights for property buyers, sellers, and mortgage lenders.

4. **Microservice & Dashboard Integration:**  
   I serialized the final trained pipeline into `lightgbm_model.joblib` and collaborated on its integration into our FastAPI microservice and interactive valuation dashboard.

---

## 2. Technical Skills Acquired

1. **Modern Gradient Boosting Frameworks:**  
   Deepened my practical mastery of LightGBM's hyperparameters (`num_leaves`, `max_depth`, `colsample_bytree`, `learning_rate`), contrasting its leaf-wise tree growth against traditional level-wise algorithms.
2. **Leakage-Free ML Pipelines:**  
   Learned to encapsulate multi-stage preprocessors (imputation, one-hot encoding, target transformations) directly within Scikit-Learn pipelines, ensuring zero test or validation leakage across cross-validation folds.
3. **Parametric vs. Non-Parametric Modeling:**  
   Gained concrete appreciation for when to select linear models (Ridge for coefficient interpretability) versus tree ensembles (LightGBM/XGBoost for complex non-linear interactions).
4. **End-to-End MLOps Artifact Management:**  
   Acquired experience in serializing pipelines with `joblib`, managing JSON metric schemas, and serving multi-model REST APIs with FastAPI.

---

## 3. Challenges Encountered & Resolutions

| Challenge Encountered | Technical Root Cause | Resolution Strategy |
| :--- | :--- | :--- |
| **High Initial Residuals on Mansions** | Linear scaling struggled with long right-tail property prices ($600,000+). | Applied `np.log1p(SalePrice)` before training, stabilizing variance and transforming predictions back with `np.expm1()`. |
| **Leaf-Wise Tree Overfitting Risk** | LightGBM's unconstrained best-first search can overfit small localized leaf clusters. | Capped `max_depth=6`, tuned `num_leaves=31`, and used `colsample_bytree=0.7` to constrain model capacity and maintain an overfitting gap < 0.10. |
| **Cross-Environment Path Resolution** | Notebooks failed when moving between local VS Code and Google Colab environments. | Developed a dynamic root-detection routine using `pathlib.Path` and fallback module imports (`src.member_03_wazni_ahamed` and `src.member_03`). |

---

## 4. Personal Reflection & Future Outlook

This project was a transformative learning experience. Beyond simply calling `.fit()` and `.predict()`, I gained deep insight into the mathematical principles of gradient boosting, the critical necessity of leakage prevention, and the value of clear collaborative code modularity. 

In future machine learning projects, I plan to explore Bayesian optimization for automated hyperparameter tuning (e.g., Optuna) and implement automated CI/CD model regression testing pipelines.

---
**Student Signature:** *Raashidh M.R.A.*  
**Date:** October 3, 2026
