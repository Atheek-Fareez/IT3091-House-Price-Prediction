"""Leakage-safe Ames property preprocessing for Group 2026-AI-08K.

Author: Ahamed M.A.W (IT24103352), Member 3.
All learned state is fitted from the rows supplied to fit. No target is used.
See the Member 3 notebook and decision log for evidence and limitations.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

EXCLUDED = ["Id", "SalePrice", "SaleType", "SaleCondition", "YrSold", "MoSold", "MiscVal"]
MEASUREMENTS = "LotFrontage LotArea MasVnrArea BsmtFinSF1 BsmtFinSF2 BsmtUnfSF TotalBsmtSF 1stFlrSF 2ndFlrSF LowQualFinSF GrLivArea GarageArea WoodDeckSF OpenPorchSF EnclosedPorch 3SsnPorch ScreenPorch PoolArea".split()
COUNTS = "BsmtFullBath BsmtHalfBath FullBath HalfBath BedroomAbvGr KitchenAbvGr TotRmsAbvGrd Fireplaces GarageCars".split()
YEARS = ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]
NUMERIC = MEASUREMENTS + COUNTS + YEARS + ["OverallQual", "OverallCond"]
QUALITY = ["Po", "Fa", "TA", "Gd", "Ex"]
ORDINAL_ORDERS = {
    **{c: QUALITY for c in "ExterQual ExterCond BsmtQual BsmtCond HeatingQC KitchenQual FireplaceQu GarageQual GarageCond".split()},
    "PoolQC": ["Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Gtl", "Mod", "Sev"],
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    "PavedDrive": ["N", "P", "Y"],
}
NOMINAL = "MSSubClass MSZoning Street Alley LandContour LotConfig Neighborhood Condition1 Condition2 BldgType HouseStyle RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType Foundation BsmtFinType1 BsmtFinType2 Heating Electrical GarageType Fence MiscFeature".split()
PROPERTY_COLUMNS = NUMERIC + list(ORDINAL_ORDERS) + NOMINAL + ["CentralAir"]
GARAGE_CATS = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
BSMT_CATS = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
BSMT_PARTS = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"]
BSMT_NUMERIC = BSMT_PARTS + ["TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
PRESENCE = ["HasGarage", "HasBasement", "HasFireplace", "HasPool", "HasMasonryVeneer"]
ENGINEERED = ["TotalIndoorAreaSF", "FinishedBasementSF", "BathroomEquivalent", "TotalPorchSF"]
UNKNOWN_NUMERIC = ["LotFrontage", "MasVnrArea", "GarageYrBlt", "GarageCars", "GarageArea"] + BSMT_NUMERIC + YEARS[:2]
UNKNOWN_CATEGORICAL = ["MSZoning", "Electrical", "Exterior1st", "Exterior2nd", "MasVnrType", "GarageType", "MiscFeature", "BsmtFinType1", "BsmtFinType2"]
CONFLICTS = ["GarageConflict", "BasementConflict", "FireplaceConflict", "PoolConflict", "MasonryConflict", "InvalidYear", "RemodelBeforeBuilt"]
INDICATORS = (
    [c + "Unknown" for c in UNKNOWN_NUMERIC + list(ORDINAL_ORDERS) + UNKNOWN_CATEGORICAL]
    + [c + "Unknown" for c in PRESENCE] + CONFLICTS + ["CentralAirUnknown"]
)
DEFAULT_LOG_COLUMNS = ["LotArea", "LotFrontage", "GrLivArea", "MasVnrArea", "TotalIndoorAreaSF"]
DOMINANT_REVIEW = "Street Utilities Condition2 RoofMatl Heating LowQualFinSF KitchenAbvGr 3SsnPorch PoolArea PoolQC MiscFeature MiscVal".split()


def load_ames_data(repository_root):
    """Return untouched-token train/test frames; only empty cells become NaN.

    Literal 'NA' and 'None' survive for the semantic cleaner and raw audit.
    Inputs can be either the repository root or the raw dataset directory.
    """
    root = Path(repository_root)
    folder = root if (root / "train.csv").is_file() else root / "data/raw/house-prices-advanced-regression-techniques"
    frames = [pd.read_csv(folder / name, keep_default_na=False, na_values=[""])
              for name in ("train.csv", "test.csv")]
    train, test = frames
    if "SalePrice" not in train or "SalePrice" in test:
        raise ValueError("Expected target only in train.csv.")
    if list(train.drop(columns="SalePrice").columns) != list(test.columns):
        raise ValueError("Training and test predictor schemas differ.")
    for frame in frames:
        if frame.Id.isna().any() or not frame.Id.is_unique:
            raise ValueError("Ids must be nonmissing and unique within each file.")
    if set(train.Id) & set(test.Id):
        raise ValueError("Train/test Ids overlap.")
    target = pd.to_numeric(train.SalePrice, errors="raise")
    if not np.isfinite(target).all():
        raise ValueError("Training target must be finite and observed.")
    return train, test


def _unknown(series):
    return series.isna() | series.isin(["NA", "", "Unknown"])


class AmesSemanticCleaner(TransformerMixin, BaseEstimator):
    """Deterministic property-only cleaning; excluded inputs cannot affect output.

    Bounds are explicit plausibility assumptions, not estimated test statistics.
    2207 is made unknown, never corrected to a guessed year. Sale chronology is
    audited separately because actual sale dates are excluded prediction inputs.
    """

    def __init__(self, min_year=1800, max_year=2100):
        self.min_year = min_year
        self.max_year = max_year

    def fit(self, X, y=None):
        self.transform(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Ames preprocessing requires a named pandas DataFrame.")
        missing = sorted(set(PROPERTY_COLUMNS) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required property columns: {missing}")
        d = X.loc[:, PROPERTY_COLUMNS].copy(deep=True)
        for c in NUMERIC:
            d[c] = pd.to_numeric(d[c].mask(_unknown(d[c])), errors="raise").astype(float)
            if np.isinf(d[c]).any():
                raise ValueError(f"Infinite measurement in {c}.")
            if (d[c].dropna() < 0).any():
                raise ValueError(f"Negative quantity in {c}; audit before preprocessing.")
        for c in ["OverallQual", "OverallCond"]:
            d[c] = d[c].where(d[c].between(1, 10))
        for c in NOMINAL + list(ORDINAL_ORDERS) + ["CentralAir"]:
            d[c] = d[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
        # Canonical building-class strings work for integer and float input dtypes.
        d["MSSubClass"] = d.MSSubClass.map(
            lambda v: str(int(float(v))) if pd.notna(v) and v not in ["NA", "", "Unknown"] else "Unknown")
        invalid = pd.DataFrame({c: d[c].notna() & ~d[c].between(self.min_year, self.max_year) for c in YEARS})
        d["InvalidYear"] = invalid.any(axis=1).astype(int)
        for c in YEARS:
            d.loc[invalid[c], c] = np.nan
        d["RemodelBeforeBuilt"] = (d.YearRemodAdd < d.YearBuilt).astype(int)
        d.loc[d.RemodelBeforeBuilt.eq(1), "YearRemodAdd"] = np.nan

        garage_known = pd.concat([~_unknown(d[c]) for c in GARAGE_CATS], axis=1).any(axis=1)
        garage_positive = d[["GarageCars", "GarageArea"]].gt(0).any(axis=1)
        garage_zero = d.GarageCars.eq(0) & d.GarageArea.eq(0)
        garage_absent = garage_zero & ~garage_known
        d["GarageConflict"] = ((garage_zero & garage_known) | (garage_positive & _unknown(d.GarageType))).astype(int)
        d["HasGarage"] = np.select([garage_positive | garage_known, garage_absent], [1., 0.], default=np.nan)
        for c in GARAGE_CATS:
            d[c] = d[c].mask(_unknown(d[c]), "Unknown")
            d.loc[garage_absent, c] = "NoGarage"
        d.loc[garage_absent, ["GarageCars", "GarageArea"]] = 0.
        d.loc[garage_absent, "GarageYrBlt"] = np.nan

        # Recover area only when arithmetic identifies exactly one missing value.
        parts = d[BSMT_PARTS]
        recover_total = d.TotalBsmtSF.isna() & parts.notna().all(axis=1)
        d.loc[recover_total, "TotalBsmtSF"] = parts.sum(axis=1)
        for c in BSMT_PARTS:
            others = [v for v in BSMT_PARTS if v != c]
            remainder = d.TotalBsmtSF - d[others].sum(axis=1, min_count=2)
            recover = d[c].isna() & remainder.ge(0)
            d.loc[recover, c] = remainder[recover]
        basement_known = pd.concat([~_unknown(d[c]) for c in BSMT_CATS], axis=1).any(axis=1)
        basement_positive = d[BSMT_NUMERIC].gt(0).any(axis=1)
        basement_zero = d.TotalBsmtSF.eq(0) & ~basement_positive
        basement_absent = basement_zero & ~basement_known
        total_conflict = d[BSMT_PARTS + ["TotalBsmtSF"]].notna().all(axis=1) & ~np.isclose(d.TotalBsmtSF, d[BSMT_PARTS].sum(axis=1))
        d["BasementConflict"] = ((basement_zero & basement_known) | total_conflict).astype(int)
        d["HasBasement"] = np.select([basement_positive | basement_known, basement_absent], [1., 0.], default=np.nan)
        for c in BSMT_CATS:
            d[c] = d[c].mask(_unknown(d[c]), "Unknown")
            d.loc[basement_absent, c] = "NoBasement"
        d.loc[basement_absent, BSMT_NUMERIC] = 0.

        for quantity, quality, state, absence, flag in [
            ("Fireplaces", "FireplaceQu", "HasFireplace", "NoFireplace", "FireplaceConflict"),
            ("PoolArea", "PoolQC", "HasPool", "NoPool", "PoolConflict"),
        ]:
            known = ~_unknown(d[quality])
            positive = d[quantity].gt(0)
            absent = d[quantity].eq(0) & ~known
            d[flag] = ((d[quantity].eq(0) & known) | (positive & ~known)).astype(int)
            d[state] = np.select([positive | known, absent], [1., 0.], default=np.nan)
            d[quality] = d[quality].mask(_unknown(d[quality]), "Unknown")
            d.loc[absent, quality] = absence

        material_known = ~_unknown(d.MasVnrType) & d.MasVnrType.ne("None")
        positive = d.MasVnrArea.gt(0)
        none = d.MasVnrType.eq("None")
        d["MasonryConflict"] = ((none & positive) | (material_known & d.MasVnrArea.eq(0))).astype(int)
        d["HasMasonryVeneer"] = np.select([positive | material_known, none & ~positive], [1., 0.], default=np.nan)
        d.loc[none & d.MasVnrArea.isna(), "MasVnrArea"] = 0.
        d.loc[none & positive, "MasVnrType"] = "Unknown"
        for c, absence in [("Alley", "NoAlley"), ("Fence", "NoFence"), ("MiscFeature", "NoMiscFeature")]:
            # Only the explicit dictionary NA token asserts absence; blank is unknown.
            d[c] = d[c].replace("NA", absence)
        for c in NOMINAL:
            d[c] = d[c].mask(_unknown(d[c]), "Unknown")
        indicators = {c + "Unknown": d[c].isna().astype(int) for c in UNKNOWN_NUMERIC}
        indicators["GarageYrBltUnknown"].loc[garage_absent] = 0
        for c in UNKNOWN_CATEGORICAL:
            indicators[c + "Unknown"] = d[c].eq("Unknown").astype(int)
        for c in PRESENCE:
            indicators[c + "Unknown"] = d[c].isna().astype(int)
            d[c] = d[c].fillna(0.)  # Placeholder accompanied by its unknown flag.
        for c, order in ORDINAL_ORDERS.items():
            mapping = {value: float(i + 1) for i, value in enumerate(order)}
            mapping.update({v: 0. for v in ["NoGarage", "NoBasement", "NoFireplace", "NoPool"]})
            d[c] = d[c].map(mapping)
            indicators[c + "Unknown"] = d[c].isna().astype(int)
        d["CentralAir"] = d.CentralAir.map({"N": 0., "Y": 1.})
        indicators["CentralAirUnknown"] = d.CentralAir.isna().astype(int)
        d["CentralAir"] = d.CentralAir.fillna(0.)
        return pd.concat([d, pd.DataFrame(indicators, index=d.index)], axis=1)


class ApplicableMedianImputer(TransformerMixin, BaseEstimator):
    """Fold-local medians excluding structurally inapplicable observations.

    No-garage year receives a neutral median placeholder with HasGarage=0.
    If no applicable observations exist, use zero (1900 for a year) and warn;
    unknown indicators still distinguish that fallback from observed values.
    """

    def fit(self, X, y=None):
        self.statistics_ = {}
        self.fit_row_count_ = len(X)
        self.fallback_columns_ = []
        for c in NUMERIC + list(ORDINAL_ORDERS):
            observed = X[c].dropna()
            group = None
            if c.startswith("Garage"):
                group = "HasGarage"
            elif c.startswith("Bsmt") or c == "TotalBsmtSF":
                group = "HasBasement"
            elif c == "MasVnrArea":
                group = "HasMasonryVeneer"
            elif c == "FireplaceQu":
                group = "HasFireplace"
            elif c == "PoolQC":
                group = "HasPool"
            if group:
                observed = X.loc[X[group].eq(1) & X[group + "Unknown"].eq(0), c].dropna()
            if observed.empty:
                value = 1900. if c in YEARS else 0.
                self.fallback_columns_.append(c)
            else:
                value = float(observed.median())
            self.statistics_[c] = value
        if self.fallback_columns_:
            warnings.warn("No applicable observed training values; explicit fallback used for " + ", ".join(self.fallback_columns_), UserWarning)
        return self

    def transform(self, X):
        check_is_fitted(self, "statistics_")
        d = X.copy(deep=True)
        total_missing = d.TotalBsmtSF.isna()
        d = d.fillna(self.statistics_)
        # Preserve a missing total's relationship to the imputed components.
        d.loc[total_missing, "TotalBsmtSF"] = d.loc[total_missing, BSMT_PARTS].sum(axis=1)
        return d


class PropertyFeatureEngineer(TransformerMixin, BaseEstimator):
    """Small deterministic feature set; use after semantic cleaning/imputation."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy(deep=True)
        d["TotalIndoorAreaSF"] = d.GrLivArea + d.TotalBsmtSF
        d["FinishedBasementSF"] = d.BsmtFinSF1 + d.BsmtFinSF2
        d["BathroomEquivalent"] = d.FullBath + d.BsmtFullBath + .5 * (d.HalfBath + d.BsmtHalfBath)
        d["TotalPorchSF"] = d[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]].sum(axis=1, min_count=4)
        return d


def build_preprocessor(configuration="unscaled", *, log_numeric=False, sparse_output=True):
    """Return an UNFITTED preprocessor for attachment inside Member 4's CV.

    configuration: 'unscaled' or 'scaled'; only continuous/ordinal columns scale.
    log_numeric: explicitly opt into fixed log1p columns, never selected by y.
    Source and aggregate features are retained for transparent handover; their
    exact linear dependencies are documented, not selected using target scores.
    """
    if configuration not in {"unscaled", "scaled"}:
        raise ValueError("configuration must be 'unscaled' or 'scaled'")
    numeric = NUMERIC + list(ORDINAL_ORDERS) + ENGINEERED
    log_columns = DEFAULT_LOG_COLUMNS if log_numeric else []
    ordinary = [c for c in numeric if c not in log_columns]
    scale = StandardScaler() if configuration == "scaled" else "passthrough"
    transformers = [("numeric", scale, ordinary)]
    if log_columns:
        transformers.append(("log_numeric", Pipeline([
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler() if configuration == "scaled" else "passthrough"),
        ]), log_columns))
    transformers.extend([
        ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output, dtype=np.float64), NOMINAL),
        ("indicators", "passthrough", PRESENCE + INDICATORS + ["CentralAir"]),
    ])
    return Pipeline([
        ("semantic", AmesSemanticCleaner()),
        ("impute", ApplicableMedianImputer()),
        ("engineer", PropertyFeatureEngineer()),
        ("encode", ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0 if sparse_output else 0.0)),
    ])


def get_feature_names(fitted_preprocessor):
    """Names in exactly the fitted matrix's column order."""
    return fitted_preprocessor.named_steps["encode"].get_feature_names_out()
