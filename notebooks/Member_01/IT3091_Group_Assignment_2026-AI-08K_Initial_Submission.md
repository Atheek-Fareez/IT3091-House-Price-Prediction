# Sri Lanka Institute of Information Technology

## Initial Report Submission

**<Group Assignment - 2026-AI-08K>**

Y3S1 - Kandy Uni

| Student Name | Student ID |
|---|---|
| Atheek M. F | IT24103933 |
| Wijesiri S.P.R.H | IT24100602 |
| Ahamed M.A. W | IT24103352 |
| Raashidh M. R | IT24104191 |

**IT3091 - Machine Learning [2026/JUL]**

B.Sc. (Hons) in Information Technology

---

# House Price Prediction and Valuation Feature Analysis

*A Machine Learning Project Report on the Ames Housing / House Prices Dataset*

- **Dataset** - House Prices - Advanced Regression Techniques (Ames Housing)
- **Data Source** - Kaggle competition dataset
- **Business Track** - Real Estate Analytics (Property Pricing & Valuation)
- **Primary Lens** - Price Prediction
- **Optional Secondary Lens** - Valuation Feature Analysis
- **Group** - 2026-AI-08K

---

## 1. The Track

### Dataset

This project uses the Ames Housing / House Prices dataset supplied through the Kaggle "House Prices - Advanced Regression Techniques" competition. The provided training file contains 1,460 property records and 81 columns, including the target variable SalePrice. The provided test file contains 1,459 property records and 80 predictor columns. Each row represents one residential property sale, while the columns describe zoning, lot characteristics, neighbourhood, building style, quality and condition ratings, construction dates, basement and garage information, living area, rooms, amenities, and sale details.

### Stakeholder

The primary stakeholder is a real estate company or property valuation team that needs to estimate a reasonable selling price for a residential property. Agents and valuers regularly compare properties that differ in location, size, quality, age, condition, garage capacity, basement features and other characteristics. A data-driven prediction model can support this judgement by using historical sales patterns consistently across many property characteristics.

### Decision Need

The business decision is to determine a defensible expected sale price for a property before it is listed, valued or compared with other properties. The project therefore asks: given the available characteristics of a house, what sale price should be expected? The predicted value is intended as decision support rather than a replacement for professional valuation, because market conditions and local knowledge may contain information not captured in the dataset.

### Unit of Analysis

The unit of analysis is one residential property record. Every training row describes one house and its characteristics, with SalePrice representing the observed sale price for that property. This unit aligns naturally with the intended use of the model: generate one estimated price for one property based on its recorded features.

### Initial Data Quality Observations

A preliminary inspection of the training data found no fully duplicated rows. However, several variables contain missing values. The largest counts occur in PoolQC (1,453 missing), MiscFeature (1,406), Alley (1,369), Fence (1,179), MasVnrType (872) and FireplaceQu (690). Some of these missing values may represent the absence of a feature rather than an unknown measurement, so they must be interpreted carefully before preprocessing.

---

## 2. Lenses with Reasons

The dataset can support several real-estate questions, including price prediction, valuation feature analysis and property segmentation. For this project, one primary lens is selected and one closely related secondary lens is retained so that the work remains focused.

### Primary Lens - Price Prediction

Price Prediction is the primary lens. The project will estimate SalePrice from the available property characteristics. This directly answers the stakeholder's decision need because the output can support listing, comparison and valuation decisions. It is also technically appropriate because SalePrice is a continuous numeric target, making the task a supervised regression problem.

### Optional Secondary Lens - Valuation Feature Analysis

Valuation Feature Analysis is used as a secondary lens because it strengthens the main prediction task rather than creating a separate unrelated pipeline. After models are trained, feature importance or other interpretation methods can be used to identify which property characteristics contribute most strongly to predicted value. For example, variables such as overall quality, living area, neighbourhood, garage capacity, construction year and basement area can be investigated as potential value drivers, but final conclusions will be based on the project evidence rather than assumptions.

### Lens Considered and Set Aside - Property Segment Analysis

Property Segment Analysis is a valid possible lens, but it is not included in the main project because it would introduce a different objective, such as clustering houses into groups. That would require different methods and evaluation criteria and could weaken the clarity of the primary pricing decision. It is therefore treated as a possible future extension rather than a second modelling pipeline.

---

## 3. Proposed Task / Output

### Exact Task

The exact machine-learning task is supervised regression. The model will learn from the training records where SalePrice is known and will predict a numeric sale price for unseen property records. The project will begin with a simple baseline and then compare at least three suitable alternative regression methods under a consistent validation strategy.

### Exact Output

The primary output is a predicted sale price in dollars for each property. The secondary output is an evidence-based explanation of the property features that have the strongest influence on the selected model or on observed valuation patterns.

### Why This Task Fits the Business Track

The target, unit of analysis and stakeholder decision are directly aligned: one property record is used to produce one estimated sale price. The result can help a real estate company compare a proposed price with data-driven evidence, while valuation feature analysis can explain the major characteristics associated with price differences. The final recommendation will also describe limitations so that the model is used as decision support rather than treated as an exact market valuation.

### Planned Evaluation

Because the target is continuous, model evaluation will use regression metrics. The group plans to compare models using measures such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) and R-squared (R²), together with validation results and error analysis. The exact metric used as the main selection criterion will be justified during modelling.

---

## 4. Workflow

1. **Business Problem** - Support better property/pricing decisions
2. **Data Understanding** - Explore Ames Housing data; 1460 training records, 81 variables
3. **EDA & Data Quality** - SalePrice distribution, missing values, outliers, relationships
4. **Preprocessing** - Missing values, encoding, scaling, leakage checks
5. **Feature Engineering** - Create derived property features and feature selection
6. **Train / Validation** - Train/test split or cross-validation using training data
7. **Model Training** - Build baseline + at least 3 regression alternatives
8. **Evaluation** - RMSE / MAE / R² + model comparison
9. **Recommendation** - Best model, valuation factors, limitations, business value

The workflow begins with the real-estate pricing decision and then moves through data understanding and exploratory analysis before any preprocessing decisions are finalized. EDA will identify the SalePrice distribution, relationships between predictors and the target, missing-value patterns, possible outliers and unusual feature behaviour.

Preprocessing and feature engineering will then be designed in response to those findings. Missing values that indicate "no feature" must be distinguished from genuinely unknown values. Categorical variables will require suitable encoding, while scaling or transformations will be used only when appropriate for the selected methods. Data leakage will be avoided by fitting learned preprocessing steps only on training data.

A baseline regression model and at least three alternative regression methods will be trained and compared using a consistent validation strategy. The selected model will then be interpreted for valuation factors and translated into a practical recommendation, together with limitations and stakeholder value.

---

## 5. Core Responsibility of Each Member

| Member | Core responsibility | Main coding / analysis area | Main deliverable |
|---|---|---|---|
| Member 1 | Business framing & task definition | Stakeholder, decision need, unit of analysis, primary/secondary lens rationale, exact regression task and expected output, contribution to decision log | Clear problem-framing section and justified project direction |
| Member 2 | Data understanding, EDA & data quality | Load and inspect train/test data, variable types, SalePrice distribution, missing-value analysis, duplicates, outliers, relationships/correlations, categorical comparisons, EDA insight log | Evidence-based understanding of the raw data and its quality/patterns |
| Member 3 | Preprocessing & feature engineering | Missing-value treatment, categorical encoding, transformations/scaling where needed, leakage-safe pipeline, feature engineering/selection, preprocessing log | Reproducible modelling-ready feature pipeline |
| Member 4 | Model development, evaluation & recommendation | Baseline + at least three regression alternatives, validation, RMSE/MAE/R² comparison, model selection, interpretation, limitations and stakeholder recommendation | Compared models, selected approach and final business recommendation |

### Member 1 - Business Framing and Task Definition (IT24103933 Atheek M.F)

Define who will use the solution, the decision should support, why Price Prediction is the primary lens, why Valuation Feature Analysis strengthens it, the unit of analysis, the exact regression task and the expected outputs. Maintain the corresponding problem-framing and decision-log evidence.

### Member 2 - Data Understanding, EDA and Data Quality (IT24100602 Wijesiri S.P.R.H)

Load and inspect the Ames Housing files, describe what the rows and variables represent, classify numerical and categorical variables, analyse SalePrice, identify missing values and duplicates, investigate possible outliers, study important predictor-target relationships, and record the main EDA insights. This member identifies data issues and patterns; the final preprocessing treatment of those issues belongs mainly to Member 3.

### Member 3 - Preprocessing and Feature Engineering (IT24103352 Ahamed M.A.W)

Use the EDA findings to design justified missing-value handling, categorical encoding, transformations or scaling where required, and leakage-safe feature preparation. Create or select useful features and document the reasoning behind preprocessing decisions in a reproducible pipeline.

### Member 4 - Model Development, Evaluation and Recommendation (IT24104191 Raashidh M.R)

Build a sensible baseline and at least three alternative regression methods, apply an appropriate validation strategy, compare models with regression metrics, select and interpret the strongest approach, and translate the evidence into a practical real-estate recommendation with limitations and stakeholder value.

---

## 6. Dataset Snapshot and Planned Stakeholder Value

### Training and Test Files

| File | Records | Columns | Target availability |
|---|---|---|---|
| train.csv | 1,460 | 81 | SalePrice included |
| test.csv | 1,459 | 80 | SalePrice not included |

### SalePrice Snapshot

| Statistic | SalePrice (USD) |
|---|---|
| Minimum | $34,900 |
| 25th percentile | $129,975 |
| Median | $163,000 |
| Mean | $180,921 |
| 75th percentile | $214,000 |
| Maximum | $755,000 |

### Important Data-Quality Areas for Investigation

- Very high missingness in PoolQC, MiscFeature, Alley and Fence.
- Substantial missingness in MasVnrType and FireplaceQu.
- LotFrontage and several garage/basement variables also contain missing values.
- No fully duplicated training rows were found in the initial inspection.
- Missing values must be interpreted semantically because some fields may mean that a feature such as a pool, alley, fence, fireplace, garage or basement is absent.
- Potential extreme values in SalePrice, lot size and living-area variables should be investigated during EDA rather than removed automatically.

### Planned Recommendation and Stakeholder Value

The final recommendation will identify the regression approach that gives the strongest validated performance while remaining suitable for the stakeholder's decision. The selected model can provide a consistent reference estimate for property pricing, and feature analysis can help explain why similar-looking houses may receive different valuations. The final report will also state limitations such as the historical and geographic scope of the Ames data, possible market changes over time, missing information and the fact that model predictions should support rather than replace professional judgement.

---

## Appendix A - Variable Groups for Initial Analysis

The following grouping is intended to guide EDA and does not predetermine the final preprocessing method.

| Variable group | Examples |
|---|---|
| Target | SalePrice |
| Location / zoning | MSZoning, Neighborhood, Condition1, Condition2 |
| Lot / land | LotFrontage, LotArea, LotShape, LandContour, LotConfig, LandSlope |
| Building type / style | MSSubClass, BldgType, HouseStyle, RoofStyle, RoofMatl |
| Quality / condition | OverallQual, OverallCond, ExterQual, ExterCond, HeatingQC, KitchenQual |
| Age / dates | YearBuilt, YearRemodAdd, GarageYrBlt, MoSold, YrSold |
| Area / size | TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GrLivArea, GarageArea, WoodDeckSF, OpenPorchSF |
| Rooms / capacity | FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageCars |
| Basement / garage | BsmtQual, BsmtExposure, BsmtFinType1, GarageType, GarageFinish, GarageQual |
| Sale information | SaleType, SaleCondition |
