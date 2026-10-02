# IT3091 – MACHINE LEARNING
## Member 2 Report
### Data Understanding, EDA & Data Quality

**Group Assignment – 2026-AI-08K**
**House Price Prediction and Valuation Feature Analysis**

| Student Name | Student ID | Core Responsibility |
|---|---|---|
| Wijesiri S.P.R.H | IT24100602 | Data Understanding, EDA & Data Quality |

---

## 1. Objective of Member 2 Work

The purpose of this section is to understand the Ames Housing training dataset, identify important patterns related to the target variable SalePrice, and assess data-quality issues that may affect later preprocessing and model development. The analysis focuses on dataset structure, variable types, missing values, duplicates, target distribution, numerical and categorical relationships, potential outliers, low-variation features, and internal consistency.

---

## 2. Dataset Understanding

The training dataset contains 1,460 property records and 81 columns. Each row represents one residential property observation. SalePrice is the target variable to be predicted, while the remaining variables describe property characteristics such as size, quality, location, age, basement, garage, and sale conditions.

| Item | Result |
|---|---|
| Number of rows | 1,460 |
| Number of columns | 81 |
| Numerical-type columns | 38 |
| Categorical/object columns | 43 |
| Target variable | SalePrice |
| Duplicate rows | 0 |

Pandas data types alone do not fully describe the meaning of each feature. Some variables stored as numbers represent categories or ordered ratings; for example, MSSubClass is a building-class code and OverallQual is an ordinal quality rating.

---

## 3. Missing-Value Analysis

Several features contain missing values. The highest missingness occurs in features representing optional property characteristics. These missing values should be interpreted carefully because some may represent the absence of a feature rather than unknown or incorrectly recorded data.

| Variable | Missing Count | Missing % |
|---|---|---|
| PoolQC | 1453 | 99.5% |
| MiscFeature | 1406 | 96.3% |
| Alley | 1369 | 93.8% |
| Fence | 1179 | 80.8% |
| MasVnrType | 872 | 59.7% |
| FireplaceQu | 690 | 47.3% |
| LotFrontage | 259 | 17.7% |
| GarageType | 81 | 5.5% |
| GarageYrBlt | 81 | 5.5% |
| GarageFinish | 81 | 5.5% |
| GarageQual | 81 | 5.5% |
| GarageCond | 81 | 5.5% |
| BsmtExposure | 38 | 2.6% |
| BsmtFinType2 | 38 | 2.6% |
| BsmtQual | 37 | 2.5% |
| BsmtCond | 37 | 2.5% |
| BsmtFinType1 | 37 | 2.5% |
| MasVnrArea | 8 | 0.5% |
| Electrical | 1 | 0.1% |

---

## 4. Target Variable: SalePrice

SalePrice contains no missing values. Prices range from $34,900 to $755,000. The mean sale price is approximately $180,921, while the median is $163,000. Because the mean is higher than the median and the histogram has a long upper tail, the target distribution is right-skewed.

**[Figure: Distribution of SalePrice — histogram showing frequency of sale prices, right-skewed with a peak around $150,000–$200,000 and a long tail extending to $755,000]**

---

## 5. Numerical Relationship Analysis

Pearson correlation was used as an exploratory measure for numerical variables. Correlation is interpreted as association rather than causation.

| Variable | Correlation with SalePrice |
|---|---|
| OverallQual | 0.791 |
| GrLivArea | 0.709 |
| GarageCars | 0.640 |
| GarageArea | 0.623 |
| TotalBsmtSF | 0.614 |
| 1stFlrSF | 0.606 |
| FullBath | 0.561 |
| TotRmsAbvGrd | 0.534 |
| YearBuilt | 0.523 |
| YearRemodAdd | 0.507 |

OverallQual has the strongest positive numerical relationship with SalePrice (r = 0.791), followed by GrLivArea (r = 0.709). GarageCars, GarageArea, TotalBsmtSF and 1stFlrSF also show notable positive relationships, suggesting that overall quality, usable living space, garage capacity, basement area and floor area are important variables to investigate during modelling.

**[Figure: GrLivArea vs SalePrice — scatter plot showing a clear positive relationship between above-ground living area and SalePrice]**

The GrLivArea scatter plot shows a clear positive relationship between above-ground living area and SalePrice. A small number of very large properties do not follow the general pattern and were flagged for outlier investigation.

**[Figure: SalePrice by Overall Quality — boxplot showing SalePrice distribution across Overall Quality Ratings 1–10]**

The OverallQual boxplot shows that median SalePrice generally rises strongly as the quality rating increases. Price variability also becomes larger among higher-quality homes.

---

## 6. Categorical Relationship: Neighborhood

**[Figure: Median SalePrice by Neighborhood — bar chart ranking neighborhoods from MeadowV (lowest) to NridgHt (highest)]**

Median sale prices differ substantially across neighborhoods. NridgHt, NoRidge and StoneBr are among the higher-median-price neighborhoods, while MeadowV, IDOTRR and BrDale are among the lower-priced neighborhoods. This indicates that Neighborhood is likely to be an important categorical predictor.

---

## 7. Outlier Analysis

Boxplots show high-value SalePrice observations and unusually large GrLivArea values. These observations were flagged rather than removed, because removal is a preprocessing decision that should be justified later.

**[Figure: SalePrice Outlier Check and GrLivArea Outlier Check — side-by-side boxplots showing upper outliers in both variables]**

| GrLivArea (sq ft) | SalePrice ($) |
|---|---|
| 5,642 | 160,000 |
| 4,676 | 184,750 |
| 4,476 | 745,000 |
| 4,316 | 755,000 |
| 3,627 | 625,000 |
| 3,608 | 475,000 |
| 3,493 | 295,000 |
| 3,447 | 381,000 |
| 3,395 | 200,000 |
| 3,279 | 538,000 |

Two particularly unusual observations have GrLivArea values of 5,642 and 4,676 square feet but SalePrice values of only $160,000 and $184,750. These cases appear inconsistent with the overall positive relationship and should be investigated during preprocessing rather than automatically deleted.

---

## 8. Low-Variation / Dominant Features

Some features are dominated by a single value. Such features may contribute limited discriminatory information and should be reviewed during feature selection rather than removed automatically.

| Variable | Dominant Value Share |
|---|---|
| Street | 99.59% |
| Utilities | 99.93% |
| Condition2 | 98.97% |
| RoofMatl | 98.22% |
| Heating | 97.81% |
| LowQualFinSF | 98.22% |
| KitchenAbvGr | 95.34% |
| 3SsnPorch | 98.36% |
| PoolArea | 99.52% |
| PoolQC | 99.52% |
| MiscFeature | 96.30% |
| MiscVal | 96.44% |

For example, Street is Pave for approximately 99.59% of properties and Utilities is AllPub for approximately 99.93%. PoolQC and MiscFeature also appear highly dominant partly because missing values reflect the absence of those optional property features.

---

## 9. Data Quality and Internal Consistency

The dataset was checked for duplicate rows, structural missingness and consistency among related property features. No duplicate rows were identified.

| Garage field | Missing count |
|---|---|
| GarageType | 81 |
| GarageYrBlt | 81 |
| GarageFinish | 81 |
| GarageCars | 0 |
| GarageArea | 0 |
| GarageQual | 81 |
| GarageCond | 81 |

| Basement field | Missing count |
|---|---|
| BsmtQual | 37 |
| BsmtCond | 37 |
| BsmtExposure | 38 |
| BsmtFinType1 | 37 |
| BsmtFinType2 | 38 |
| BsmtFinSF1 | 0 |
| BsmtFinSF2 | 0 |
| BsmtUnfSF | 0 |
| TotalBsmtSF | 0 |

Where GarageType is missing, the corresponding GarageCars and GarageArea values are zero in the checked records, indicating that the property does not have a garage. Similarly, properties with missing BsmtQual have zero values in the basement square-footage fields, indicating the absence of a basement. Therefore, many missing garage and basement categorical values are structural absence rather than random missing data.

---

## 10. EDA Insight Log

| Finding | Evidence | Possible Impact |
|---|---|---|
| SalePrice is right-skewed | Histogram and summary statistics | A target transformation may be considered later during modelling. |
| OverallQual strongly relates to SalePrice | r = 0.791 and boxplot | Likely an important predictive feature. |
| GrLivArea strongly relates to SalePrice | r = 0.709 and scatter plot | Important size-related feature; also contains unusual observations. |
| Neighborhood prices differ substantially | Median-price bar chart | Location is likely an important categorical predictor. |
| Several columns contain missing values | Missing-value table | Missingness requires feature-aware preprocessing. |
| No duplicate rows found | Duplicate check = 0 | No duplicate-row removal is required. |
| Extreme GrLivArea observations exist | Boxplot and top-area records | Outliers should be investigated before modelling. |
| Some features are highly dominated by one value | Street, Utilities and other features | Review during feature selection. |
| Garage/basement missingness is structurally meaningful | Zero area/capacity with missing category fields | Treat feature absence separately from unknown missingness. |

---

## 11. Data Dictionary Summary

| Variable | Meaning | Type | Role |
|---|---|---|---|
| SalePrice | Property sale price in dollars | Numeric | Target |
| OverallQual | Overall material and finish quality | Ordinal | Predictor |
| GrLivArea | Above-ground living area in square feet | Numeric | Predictor |
| GarageCars | Garage capacity in number of cars | Numeric | Predictor |
| GarageArea | Garage area in square feet | Numeric | Predictor |
| TotalBsmtSF | Total basement area in square feet | Numeric | Predictor |
| 1stFlrSF | First-floor area in square feet | Numeric | Predictor |
| FullBath | Number of full bathrooms above grade | Numeric | Predictor |
| TotRmsAbvGrd | Total rooms above grade excluding bathrooms | Numeric | Predictor |
| YearBuilt | Original construction year | Year | Predictor |
| Neighborhood | Physical location within Ames city limits | Categorical | Predictor |
| LotArea | Lot size in square feet | Numeric | Predictor |
| YearRemodAdd | Year of remodel or addition | Year | Predictor |
| Fireplaces | Number of fireplaces | Numeric | Predictor |
| BsmtFinSF1 | Type 1 finished basement area in square feet | Numeric | Predictor |

The full Ames Housing dataset contains 81 variables. The table above summarizes the target and selected important variables highlighted during EDA. The complete official variable descriptions can be retained in the project appendix or supporting documentation.

---

## 12. Handover to Member 3

The findings from this analysis should guide the preprocessing and feature-engineering stage. Member 3 should pay particular attention to structural missingness, categorical encoding, possible target transformation, outlier handling, highly dominant variables, and leakage-safe preprocessing. Missing values should not be imputed uniformly without considering the semantic meaning of each feature.
