"""Feature engineering and preprocessing pipelines.

Everything here is an sklearn transformer so that it is fit *inside* each CV fold (no
leakage) and is serialised together with the model.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import ADDON_SERVICE_COLS, CATEGORY_LEVELS, DURATION_COL, ID_COL, TARGET_COL

ModelKind = Literal["linear", "tree"]

# "No internet service" / "No phone service" duplicate the information already in
# InternetService / PhoneService. Collapsing them to "No" removes perfectly collinear
# dummy columns (which matter for Cox regression and for interpreting coefficients).
_REDUNDANT_LEVELS = {"No internet service": "No", "No phone service": "No"}

# Features derived from tenure. They are excluded from survival models, where tenure is
# the outcome: TotalCharges is roughly tenure * MonthlyCharges and would leak duration.
TENURE_DERIVED_COLS = [DURATION_COL, "TotalCharges", "avg_monthly_charge", "charge_delta"]

BASE_NUMERIC_COLS = ["MonthlyCharges", "n_addon_services"]


def model_category_levels() -> dict[str, list[str]]:
    """Category levels after collapsing redundant levels (reference level first)."""
    return {
        col: [lvl for lvl in levels if lvl not in _REDUNDANT_LEVELS]
        for col, levels in CATEGORY_LEVELS.items()
    }


class TelcoFeatureEngineer(TransformerMixin, BaseEstimator):
    """Stateless feature engineering (row-wise only, so it cannot leak across folds).

    Adds:
      * ``n_addon_services`` - number of add-on services subscribed (0-6).
      * ``avg_monthly_charge`` - TotalCharges / tenure (MonthlyCharges when tenure is 0).
      * ``charge_delta`` - MonthlyCharges - avg_monthly_charge: a recent price change.

    With ``include_tenure=False`` every tenure-derived column is dropped (survival use).
    """

    def __init__(self, include_tenure: bool = True):
        self.include_tenure = include_tenure

    def fit(self, X: pd.DataFrame, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(columns=[ID_COL, TARGET_COL], errors="ignore").copy()

        for col in CATEGORY_LEVELS:
            X[col] = X[col].replace(_REDUNDANT_LEVELS)

        X["n_addon_services"] = X[ADDON_SERVICE_COLS].eq("Yes").sum(axis=1).astype("float64")

        if self.include_tenure:
            tenure = X[DURATION_COL].astype("float64")
            X["avg_monthly_charge"] = np.where(
                tenure > 0, X["TotalCharges"] / tenure.where(tenure > 0, 1.0), X["MonthlyCharges"]
            )
            X["charge_delta"] = X["MonthlyCharges"] - X["avg_monthly_charge"]
        else:
            X = X.drop(columns=[c for c in TENURE_DERIVED_COLS if c in X.columns])

        numeric = numeric_feature_names(self.include_tenure)
        X[numeric] = X[numeric].astype("float64")
        return X[self.get_feature_names_out()]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        numeric = numeric_feature_names(self.include_tenure)
        return np.asarray([*numeric, *CATEGORY_LEVELS], dtype=object)


def numeric_feature_names(include_tenure: bool = True) -> list[str]:
    if include_tenure:
        return [
            DURATION_COL,
            "TotalCharges",
            *BASE_NUMERIC_COLS,
            "avg_monthly_charge",
            "charge_delta",
        ]
    return list(BASE_NUMERIC_COLS)


def source_feature(column: str) -> str:
    """Raw attribute a model column comes from: ``Contract_Two_year`` -> ``Contract``.

    SHAP values are additive, so summing them over the dummies of one attribute gives
    that attribute's total contribution (one bar per attribute instead of per level).
    """
    for col in CATEGORY_LEVELS:
        if column.startswith(f"{col}_"):
            return col
    return column


def _clean_feature_name(feature: str, category: object) -> str:
    """``PaymentMethod`` + ``Bank transfer (automatic)`` -> ``PaymentMethod_Bank_transfer_automatic``.

    LightGBM rejects some special characters in feature names, and clean names make SHAP
    plots and coefficient tables easier to read.
    """
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(category)).strip("_")
    return f"{feature}_{slug}"


def build_preprocessor(
    kind: ModelKind = "linear",
    include_tenure: bool = True,
    drop: Literal["first", "if_binary"] | None = "if_binary",
    exclude: tuple[str, ...] = (),
) -> Pipeline:
    """Feature engineering + encoding, returning a pandas DataFrame of floats.

    Args:
        kind: ``"linear"`` standardises numeric features; ``"tree"`` passes them through.
        include_tenure: keep tenure-derived features (False for survival models).
        drop: one-hot drop strategy. ``"first"`` drops each column's reference level,
            giving identifiable coefficients (needed for Cox hazard ratios).
        exclude: raw columns to leave out of the design matrix (e.g. Cox strata).
    """
    levels = {c: v for c, v in model_category_levels().items() if c not in exclude}
    numeric = [c for c in numeric_feature_names(include_tenure) if c not in exclude]

    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler() if kind == "linear" else "passthrough", numeric),
            (
                "cat",
                OneHotEncoder(
                    categories=list(levels.values()),
                    drop=drop,
                    handle_unknown="error",  # unseen levels are rejected upstream by the schema
                    sparse_output=False,
                    dtype=np.float64,
                    feature_name_combiner=_clean_feature_name,
                ),
                list(levels),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipe = Pipeline(
        [("engineer", TelcoFeatureEngineer(include_tenure=include_tenure)), ("encode", encoder)]
    )
    return pipe.set_output(transform="pandas")
