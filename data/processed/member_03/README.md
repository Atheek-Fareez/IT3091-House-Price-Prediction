# Member 3 processed-data handover

Group 2026-AI-08K | Ahamed M.A.W | IT24103352

## Full-training CSV exports

These full-training preprocessed CSV files are provided because the group requires
a direct processed-data handover. They are suitable for inspection, handover,
final-fit/demo use, but should NOT be used as input for cross-validation because
preprocessing was learned from all 1,460 labelled training rows. Doing so would
leak validation information through imputation, category vocabularies and scaling
when applicable. The unscaled configuration is used here, with log_numeric=False.

- train_preprocessed.csv: Id first, named transformed predictors, SalePrice last.
- test_preprocessed.csv: Id first, exactly the same predictor columns/order; no target.

Only raw training predictors fitted the preprocessor. Test predictors were
transformed using that fitted state. IDs stay aligned and SalePrice stays in
original dollars. No regression estimator was trained and no model score computed.
The manifest and feature summary record the exact exported shapes and checks.

## Leakage-safe cross-validation

For leakage-safe CV, Member 4 should use build_preprocessor() inside the model
pipeline. Import a fresh UNFITTED object from the repository root:

```python
from src.member_03 import build_preprocessor
preprocessor = build_preprocessor(configuration='unscaled', log_numeric=False)
# Alternative: configuration='scaled' for the appropriate downstream estimator.
```

Fit that pipeline separately inside each training/CV fold; validation and Kaggle
test rows use transform only. Use raw predictors as the CV input, not these CSVs.
A full-training export does not replace this factory or establish a CV strategy.

## Reproducibility and metadata

Run notebooks/Member_03/Member3_Preprocessing_FeatureEngineering.ipynb end to end
to regenerate both CSVs and the following metadata:

- preprocessing_manifest.json: configuration, versions, raw fingerprints,
  full-export column names/shapes/checksums, and separate verification-fit provenance.
- feature_summary.csv: stage counts plus exported CSV names and shapes.
- README.md: this handover and CV policy.

The full-training export has its own fresh preprocessor and is separate from the
1,168-row temporary correctness fit. Encoded feature counts may differ across fits.
All files here remain gitignored by the existing data/processed rule. No fitted
object is exported. The notebook explains decisions; src/member_03/preprocessing.py
remains the unchanged reusable implementation.
