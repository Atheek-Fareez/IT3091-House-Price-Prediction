# IT3091 – MACHINE LEARNING

## Member 3 Report

### Preprocessing & Feature Engineering

**Group Assignment – 2026-AI-08K**  
**House Price Prediction and Valuation Feature Analysis**  
**Dataset:** Ames Housing / Kaggle House Prices

| Student Name | Student ID | Core Responsibility |
|---|---|---|
| Ahamed M.A.W | IT24103352 | Preprocessing & Feature Engineering |

## 1. Objective of Member 3 Work

Member 3 converted raw Ames Housing records into consistent numerical inputs for supervised regression. Missingness, encoding, imputation, transformations, feature engineering and outlier decisions followed the supplied dictionary and Member 2's evidence, with leakage prevention built into the implementation.

The deliverables comprise reusable preprocessing components, feature documentation, a decision log and direct processed-data exports. Model training, predictive evaluation, model selection and business recommendations were not performed in this stage; these remain Member 4 responsibilities under the group project plan.

## 2. Handover from Member 2

Member 2 supplied distribution, relationship and data-quality evidence. The relevant findings informed preprocessing without repeating the full exploratory analysis.

| Member 2 finding | Member 3 response |
|---|---|
| `SalePrice` is right-skewed. | Preserve the original-dollar target; leave target-transformation comparison to Member 4. |
| Missing garage and basement descriptions often accompany zero capacity or area. | Establish explicit structural-absence rules and check exceptions. |
| Quality, living area, garage capacity and neighborhood show meaningful associations with price. | Preserve property information through appropriate numerical and categorical representations. |
| Some very large houses have unexpectedly low observed prices. | Investigate the identified records and retain them without target-dependent filtering. |
| Several variables are dominated by one value. | Review rarity without treating it as sufficient evidence for deletion. |
| Missingness is concentrated in optional features; no duplicate training rows were found. | Interpret gaps by feature meaning and avoid unnecessary row removal. |

These decisions continue the scope established in the [Member 1 initial submission](../Member_01/IT3091_Group_Assignment_2026-AI-08K_Initial_Submission.md) and the [Member 2 report](../Member_02/Member2_Data_Understanding_EDA_Data_Quality_Report.pdf), particularly its missingness, outlier and handover sections.

## 3. Raw Data and Preprocessing Audit

| Input | Records | Columns | Target availability |
|---|---:|---:|---|
| `train.csv` | 1,460 | 81 | `SalePrice` present |
| `test.csv` | 1,459 | 80 | `SalePrice` absent |

Removing the training target leaves the same raw predictor column names and order as the test file. IDs are unique within each file and do not overlap. `Id` is retained for alignment but excluded from prediction. The default property-only input contains 74 columns after separating the identifier and target and excluding five availability-restricted variables.

CSV loading uses `keep_default_na=False` with empty cells designated as missing. Literal `NA` and `None` therefore remain available for semantic interpretation. This matters for `MasVnrType`: the training file contains 864 literal `None` entries and eight `NA` entries. Default parsing combined these into the 872 missing entries reported upstream. Member 3 refines that interpretation: `None` is the documented no-veneer category, whereas `NA` indicates unknown material information in this field.

Test-only gaps in zoning, exterior materials, kitchen quality, functionality and garage/basement measurements required compatibility checks. Test records were inspected semantically but did not supply learned imputation statistics.

| Audit example | Preprocessing implication |
|---|---|
| Test IDs 2127 and 2577 specify detached garages but lack other garage information. | Existing garages require unknown-value treatment, not automatic zero filling. |
| Training ID 333 has missing second basement finish type despite 479 square feet of second finished area; ID 949 lacks exposure information despite a basement. | A missing basement descriptor does not establish absence. |
| Test IDs 2421, 2504 and 2600 have positive pool area but missing quality. | Preserve pool presence and represent quality as unknown. |
| Test ID 2593 records `GarageYrBlt=2207`. | Flag an invalid year without guessing a corrected date. |
| Five training and two test records combine masonry type `None` with positive area. | Preserve the measurement and flag the contradictory description. |

Cross-file predictor matches (training/test IDs 194/2866 and 830/2714) were retained: identical recorded attributes do not prove duplicate properties. No prices were transferred.

## 4. Missing-Value Treatment

### 4.1 Structural absence and genuine unknown values

Structural absence means an amenity does not exist; an unknown value means its information is unestablished. The [supplied data dictionary](../../data/raw/house-prices-advanced-regression-techniques/data_description.txt) and related measurements were used together to separate these situations. For example, 81 training properties have supported garage absence and 37 have supported basement absence.

| Feature group | Selected treatment and justification |
|---|---|
| Garage descriptions, area and capacity | Assign `NoGarage` and zero quantities only when zero area and capacity agree with absent descriptors. Existing garages retain unknown descriptors or receive applicable training medians for missing measurements. |
| Basement descriptions and quantities | Assign `NoBasement` and zero quantities when zero total area has no conflicting presence evidence. Missing descriptions of an existing basement remain unknown. |
| Fireplace and pool quality | Use explicit absence states when zero count or area supports absence. Positive capacity with missing quality preserves presence and an unknown-quality indicator. |
| Alley and fence | Interpret literal `NA` as documented absence. A genuinely blank value remains `Unknown`. |
| Miscellaneous feature | Interpret literal `NA` as no miscellaneous feature. The positive `MiscVal` contradiction at test ID 2550 remains audit-only because that value is excluded from the prediction inputs. |
| Lot frontage | Impute the training-fold median and retain `LotFrontageUnknown`; the dictionary provides no structural-absence interpretation. There are 259 training and 227 test gaps. |
| Masonry type and area | Preserve literal `None` when consistent with absence. Positive area contradicting `None` produces an unknown material category and a conflict flag. Unknown applicable area receives a training median. |
| Garage construction year | Unknown or invalid applicable years receive a training median. A confirmed absent garage also receives a neutral numerical year placeholder, distinguished by `HasGarage=0`; year zero is not treated as a construction date. |

`ApplicableMedianImputer` estimates medians from the rows supplied to its fitting method. For amenity-related fields, confirmed applicable records provide the observations; structurally absent properties do not define an existing amenity's typical measurement. Unknown ordinal descriptions also retain indicators when a median rank is supplied. Median placeholders may be fractional and must not be interpreted as observed counts or ratings.

### 4.2 Recoverable and uncertain information

Basement area is reconstructed deterministically when the observed components identify a missing total, or an observed total and the other components identify one nonnegative missing component. This uses information from the same record rather than population statistics. If a total remains missing, it is subsequently derived from the filled components.

Test ID 2189 has zero basement areas and missing bathroom counts; supported absence justifies zero bathrooms. Test ID 2121 has an entirely missing basement block. Its presence remains uncertain, with `HasBasementUnknown=1`, while applicable medians provide a numerical representation. A zero placeholder in a presence field therefore means confirmed absence only when its corresponding unknown flag is zero.

Contradictions are not silently overwritten. Observed masonry areas are retained, and invalid years outside the explicit 1800–2100 bounds become unknown. A remodel year preceding construction is flagged and made unknown. Actual sale-year inconsistencies remain audit evidence because sale timing is excluded from prediction. If a fitting subset has no applicable observations for imputation, the implementation emits a warning and uses its documented fallback: zero, or 1900 for a year.

Deterministic rules use property inputs only; learned replacements are estimated within the relevant training fit.

[FIGURE PLACEHOLDER — Figure 1: Semantic Missing-Value Treatment Summary]

Figure 1. Distinction between structural absence, genuine unknown values and inconsistent observations used in the Member 3 preprocessing strategy.

(Source locator in the previously completed notebook: Section 9, Visual 1, “Semantic treatment: absence is not the same as unknown.” The current notebook requires restoration before transfer; see Section 11.)

## 5. Semantic Feature Types and Encoding

Feature roles were assigned from their meanings rather than inferred exclusively from storage types. In particular, `MSSubClass` is converted to categorical strings because its dwelling-class codes do not represent numerical distances.

| Semantic group | Examples | Implemented representation |
|---|---|---|
| Numerical measurements | `LotArea`, `GrLivArea`, `GarageArea` | Numerical values with justified missingness handling; optional transformation and scaling. |
| Counts and capacity | Bathroom counts, `Fireplaces`, `GarageCars` | Numerical quantities; structural zeros retained. |
| Nominal categories | `Neighborhood`, `MSSubClass`, `Fence`, basement finish types | One-hot encoding without arbitrary ranking. |
| Ordinal features | `OverallQual`, quality descriptions, exposure and finish | Existing numerical ratings or explicit documented category orders. |
| Binary information | `CentralAir`, presence and uncertainty flags | Binary coding; indicators remain unscaled. |
| Years | `YearBuilt`, `YearRemodAdd`, `GarageYrBlt` | Numerical representation with documented plausibility checks. |
| Identifier, target and audit-only fields | `Id`, `SalePrice`, transaction-related variables | Kept outside the predictor matrix according to their roles. |

Nominal variables use `OneHotEncoder(handle_unknown='ignore')`, fitted on training categories only. Unseen categories produce an all-zero block rather than an invented rank. This accommodates `MSSubClass=150`, which occurs in test data but not training. Unknown nominal information is represented through the explicit `Unknown` state and selected indicators.

Ordinal mappings include `Po < Fa < TA < Gd < Ex` for applicable quality fields and `Unf < RFn < Fin` for garage finish. Shape and slope orders describe physical characteristics, not expected prices. Structural ordinal absence uses rank zero; genuinely unknown levels retain an unknown indicator when filled. These scores do not imply equal monetary differences. `CentralAir` maps `N` to zero and `Y` to one, with a separate unknown flag where necessary.

## 6. Feature Engineering

Nine property features are added: five presence flags during semantic cleaning and four aggregates after imputation, preserving their relationships with prepared components.

| Feature | Definition and purpose |
|---|---|
| `HasGarage` | Supported existence from garage descriptions, capacity and area. |
| `HasBasement` | Supported existence from basement descriptions and quantities; uncertain status remains flagged. |
| `HasFireplace` | Fireplace presence from count and available quality evidence. |
| `HasPool` | Pool presence from area and available quality evidence. |
| `HasMasonryVeneer` | Positive area or a recorded veneer material supports presence. |
| `TotalIndoorAreaSF` | `GrLivArea + TotalBsmtSF`; includes unfinished basement space and is not exclusively finished living area. |
| `FinishedBasementSF` | `BsmtFinSF1 + BsmtFinSF2`; combines the two finished-area components. |
| `BathroomEquivalent` | `FullBath + BsmtFullBath + 0.5 × (HalfBath + BsmtHalfBath)`; an explicit capacity heuristic. |
| `TotalPorchSF` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch`; excludes decks. |

The schema also defines 52 missingness/consistency indicators, including `LotFrontageUnknown`, `PoolQCUnknown`, `MasonryConflict` and `InvalidYear`. Definitions remain fixed even when constant within a fit, supporting later missingness patterns.

No feature uses `SalePrice`, neighborhood target averages or transaction-derived age. Half-bath weighting is a declared representation choice, not an estimated price effect.

## 7. Outlier and Low-Variation Decisions

Member 2 identified training IDs 524 and 1299 as unusual large properties. Their above-ground living areas are 4,676 and 5,642 square feet, with observed prices of $184,750 and $160,000 respectively. Both records describe new, partial sales. This contextual evidence does not establish a recording error, so both were retained. No automatic area threshold, target-dependent deletion or price-based filtering was applied.

Dominant features were also retained unless separately excluded by prediction availability. `Utilities` records `AllPub` for 1,459 of 1,460 training properties; other reviewed variables include `Street`, `Condition2`, `RoofMatl`, `Heating`, `LowQualFinSF`, `KitchenAbvGr`, `3SsnPorch` and pool-related fields. Rarity alone does not demonstrate that an attribute is irrelevant. No target-correlation cutoff or model-based feature selection was performed.

## 8. Prediction-Time Feature Availability

Member 1's pre-listing objective motivates a conservative property-only contract. Fields with unsuitable roles or unestablished prediction-time availability are excluded.

| Excluded field | Reason |
|---|---|
| `Id` | Record alignment only; not a property characteristic. |
| `SalePrice` | Target to be predicted; held separately in original dollars. |
| `SaleType` | Actual transaction type may not be known before listing. |
| `SaleCondition` | Actual transaction circumstances may not be established at prediction time. |
| `YrSold` | Actual sale year is not an independently supplied valuation date. |
| `MoSold` | Actual sale month is not established under the pre-listing contract. |
| `MiscVal` | Prediction-time provenance of the recorded monetary value is unclear. |

Exclusion does not imply statistical uselessness. These fields remain in the raw evidence for auditing but cannot influence cleaning indirectly. Consequently, `YrSold` does not repair property dates, and `MiscVal` does not resolve miscellaneous-feature contradictions inside the predictor pipeline.

## 9. Numerical Transformation and Scaling Strategy

Two configurations share the same semantic rules, imputation and engineered features. The default unscaled configuration retains numerical magnitudes. The scaled configuration applies `StandardScaler` to the numerical and ordinal-score branches, including counts and years, while leaving one-hot columns and binary indicators unscaled. This provides alternative representations for Member 4 without selecting a model.

An optional `log_numeric` setting applies `log1p` to a fixed list: `LotArea`, `LotFrontage`, `GrLivArea`, `MasVnrArea` and `TotalIndoorAreaSF`. The option is disabled by default and in the exported CSVs. It is motivated by skewed size measurements and preserves zero; it was not selected using predictive results.

`SalePrice` remains unchanged in dollars. Whether to use `log1p(SalePrice)` and how to evaluate or invert that transformation are downstream modelling decisions.

## 10. Leakage-Safe Reusable Preprocessing Pipeline

The implementation in [preprocessing.py](../../src/member_03/preprocessing.py) exposes loading, semantic cleaning, applicable imputation, feature engineering and a scikit-learn-compatible preprocessing factory. Its logical flow is:

**Raw data → property input contract → semantic cleaning → training-only imputation → feature engineering → encoding and optional scaling → model input.**

Ordinal mapping occurs during semantic cleaning. One-hot encoding and optional numerical scaling are separate branches of the final `ColumnTransformer`, rather than scaling every encoded column.

The notebook explains decisions and verification; the module supplies reusable code. Unlike exploratory analysis, preprocessing must repeatedly apply identical rules to training folds, held-out rows and later inputs. Member 4 obtains a fresh unfitted component with:

```python
from src.member_03 import build_preprocessor
preprocessor = build_preprocessor(configuration='unscaled')
```

Imputation medians, categorical vocabularies and scaling statistics must be fitted inside each training fold. Validation and Kaggle test records use the fitted transformation without updating it. This separation prevents information from held-out rows entering learned preprocessing.

[FIGURE PLACEHOLDER — Figure 2: Leakage-Safe Member 3 Preprocessing Pipeline]

Figure 2. Member 3 preprocessing workflow showing training-fold fitting of learned preprocessing steps before Member 4 model training.

(Source locator in the previously completed notebook: Section 15, Visual 2, “Reusable preprocessing: fit within each training fold.” Transfer requires restoration of the executed notebook.)

## 11. Preprocessing Verification

The recorded correctness split contains 1,168 fitting rows and 292 held-out rows, using `random_state=42`. It is a temporary preprocessing check, not the group's final model-evaluation split. Both configurations produce 297 transformed features for this fit; the full-training export has a different fitted vocabulary and produces 300.

**Evidence availability at the report audit:** the current Member 3 notebook file contains one empty code cell and no saved outputs. The complete notebook cannot therefore be certified from that file. The retained module, decision log, schema, manifest and CSVs were inspected, and the checks below were independently repeated in memory during report preparation. No technical files were changed. The previously completed notebook must be restored before its figures and executed cells can accompany the final submission.

| Preprocessing check | Verified result |
|---|---|
| Predictor exclusions | Changing target, identifier and excluded audit fields did not change transformed features. |
| Row alignment | Reversing input rows reversed output rows consistently; exported IDs matched the raw files. |
| Finite numerical output | Held-out, Kaggle test and exported predictor values contained no missing or infinite values. |
| Unseen categories | A previously unseen neighborhood transformed successfully without refitting. |
| Both configurations | Unscaled and scaled preprocessing completed with compatible feature counts for the same fit. |
| Determinism | Repeated transformation returned identical values. |
| Training-only learned state | Medians, categorical vocabularies, scaler means and variances matched the fitting subset; transformation left fitted statistics unchanged. |
| Target independence | Supplying altered target values during preprocessing fitting did not change the resulting features. |
| Export reproducibility | A fresh unscaled fit on all raw training predictors exactly reproduced both saved processed CSVs. |

These checks establish preprocessing correctness within the tested cases, not predictive performance. No regression estimator was fitted and no model-performance metric was calculated.

[FIGURE PLACEHOLDER — Figure 3: Before and After Preprocessing Completeness]

Figure 3. Selected feature completeness before and after preprocessing while preserving explicit uncertainty indicators.

(Source locator in the previously completed notebook: Section 16, Visual 3, “Before / after preprocessing: complete inputs, visible uncertainty.” The figure uses Kaggle test rows transformed by the verification-fit preprocessor; transfer requires the restored notebook.)

## 12. Processed Dataset Handover

For the requested direct handover, a separate unscaled preprocessor was fitted on all raw training predictors, with logarithms disabled. Its fitted state transformed training and test records; neither test rows nor target values supplied learned parameters.

| Exported file | Rows | Columns | Composition |
|---|---:|---:|---|
| `train_preprocessed.csv` | 1,460 | 302 | `Id`, 300 transformed predictors, then `SalePrice`. |
| `test_preprocessed.csv` | 1,459 | 301 | `Id` and the same 300 predictors; no target. |

Saved files have identical predictor names/order, finite values, aligned IDs and unchanged training `SalePrice`. They reside under the gitignored `data/processed/member_03/` directory with the README, manifest and feature summary.

These CSVs support inspection, handover and final-fit/demo use. They must not replace fold-wise preprocessing for cross-validation: their fitted transformation has already used every labelled row, including rows that a later split would designate for validation. For validation-safe comparison, Member 4 should supply raw predictors to a model pipeline containing a fresh `build_preprocessor()` instance.

## 13. Preprocessing Decision Log and Feature Schema

The [preprocessing decision log](preprocessing_feature_decision_log.csv) contains 25 decisions covering issues, supporting evidence, selected treatments, alternatives, rationale, leakage implications and status. It records both retained information and deliberate exclusions, allowing implementation choices to be reviewed independently of model results.

The [feature schema](feature_schema.csv) contains 142 original and derived entries. It documents meaning, raw and semantic types, structural-absence rules, missing-value strategies, encoding, scaling, prediction availability and default use. It includes excluded source fields and derived indicators; it is not a list of the 300 expanded output columns. The export manifest supplies the actual transformed column names, file shapes and checksums.

## 14. Member 4 Handover

| Handover component | Purpose and current status |
|---|---|
| Processed training/test CSVs | Direct inspection and final-fit/demo inputs, with the full-training fitting limitation stated. |
| Unfitted preprocessing factory | Reusable component for leakage-safe training and cross-validation. |
| Unscaled/scaled configurations | Shared semantic treatment with model-appropriate numerical representation options. |
| Feature schema and decision log | Traceable definitions, assumptions and decision evidence. |
| Manifest, feature summary and README | Shapes, versions, fingerprints, fitting provenance and usage policy. |
| Explanatory notebook and figures | Intended assignment evidence; the current empty notebook must be restored before submission. |

Member 4 remains responsible for a baseline regression model, at least three alternatives, validation strategy, RMSE/MAE/R² evaluation, optional target-transformation comparison, model selection, interpretation and recommendations. None of these activities is presented here as a Member 3 result. 

## 15. Limitations and Remaining Assumptions

Structural interpretation depends on the Ames dictionary and consistency between recorded fields. Completely missing basement information cannot establish true absence. Imputation provides usable numerical inputs while uncertainty indicators preserve the distinction from observed measurements. Neutral year placeholders likewise do not establish actual construction dates.

Ordinal scores encode an assumed spacing between ordered categories. Some medians can be fractional, and broad year bounds do not detect every chronological error. Sale-context contradictions remain audit-only under the conservative prediction-input contract, which also limits the information available to subsequent models.

Source variables and aggregates retain exact mathematical dependencies, including floor-area and basement totals. Full one-hot blocks can also introduce dependencies. The representation is not guaranteed to have full column rank; Member 4 must account for estimator-specific requirements without performing selection outside the training folds.

Member 2's exploration used all labelled records, so a later holdout cannot be described as entirely untouched by exploratory inspection. Separately, restoration of the explanatory notebook remains necessary for complete submission evidence. Neither limitation changes the independently verified contents of the existing processed CSVs.

## 16. Member 3 Contribution Summary

Member 3 implemented evidence-based property preprocessing that distinguishes structural absence, unknown information and inconsistent records. The contribution combines interpretable feature engineering, explicit prediction-time exclusions, training-only learned transformations and documented preprocessing checks. The processed CSVs provide the group's requested direct handover, while the reusable factory supports Member 4's fold-wise workflow. No claim about model accuracy or business performance follows from these preprocessing results.

### AI-Assisted Development Transparency

AI-assisted tools supported implementation, review and documentation. Project decisions, verification of evidence and final responsibility remain with the student and group. The report distinguishes verified implementation and export evidence from the current notebook availability gap.
