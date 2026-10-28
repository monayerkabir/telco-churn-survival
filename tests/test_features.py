import re

import numpy as np
import pytest

from churn.config import CATEGORY_LEVELS
from churn.features import (
    TENURE_DERIVED_COLS,
    TelcoFeatureEngineer,
    build_preprocessor,
    model_category_levels,
    numeric_feature_names,
)


def expected_n_columns(include_tenure: bool, drop, exclude=()) -> int:
    n = len([c for c in numeric_feature_names(include_tenure) if c not in exclude])
    for col, levels in model_category_levels().items():
        if col in exclude:
            continue
        if drop == "first" or (drop == "if_binary" and len(levels) == 2):
            n += len(levels) - 1
        else:
            n += len(levels)
    return n


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_preprocessor_shape_and_dtypes(clean_df, kind):
    Z = build_preprocessor(kind).fit_transform(clean_df)

    assert Z.shape == (len(clean_df), expected_n_columns(True, "if_binary"))
    assert (Z.dtypes == "float64").all()
    assert not Z.isna().any().any()
    # Clean names: safe for LightGBM and readable in SHAP plots.
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", c) for c in Z.columns)


def test_linear_preprocessor_standardises_numeric(clean_df):
    Z = build_preprocessor("linear").fit_transform(clean_df)
    assert Z["MonthlyCharges"].mean() == pytest.approx(0, abs=1e-9)
    assert Z["MonthlyCharges"].std(ddof=0) == pytest.approx(1)


def test_tree_preprocessor_passes_numeric_through(clean_df):
    Z = build_preprocessor("tree").fit_transform(clean_df)
    np.testing.assert_allclose(Z["MonthlyCharges"], clean_df["MonthlyCharges"])


def test_survival_preprocessor_excludes_tenure_and_strata(clean_df):
    Z = build_preprocessor("linear", include_tenure=False, drop="first", exclude=("Contract",))

    out = Z.fit_transform(clean_df)

    assert out.shape[1] == expected_n_columns(False, "first", exclude=("Contract",))
    assert not set(TENURE_DERIVED_COLS) & set(out.columns)
    assert not any(c.startswith("Contract") for c in out.columns)
    # drop="first" removes each reference level -> identifiable Cox coefficients
    assert "InternetService_DSL" not in out.columns
    assert "InternetService_Fiber_optic" in out.columns


def test_redundant_service_levels_are_collapsed(clean_df):
    Z = build_preprocessor("tree").fit_transform(clean_df)
    assert not any("No_internet_service" in c or "No_phone_service" in c for c in Z.columns)
    assert "No internet service" in CATEGORY_LEVELS["OnlineSecurity"]  # raw schema unchanged


def test_feature_engineer_values(clean_df):
    X = TelcoFeatureEngineer().fit_transform(clean_df)

    row = clean_df.iloc[0]
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    addons += ["StreamingTV", "StreamingMovies"]
    assert X.loc[0, "n_addon_services"] == sum(row[c] == "Yes" for c in addons)

    new = clean_df["tenure"] == 0
    np.testing.assert_allclose(
        X.loc[new, "avg_monthly_charge"], clean_df.loc[new, "MonthlyCharges"]
    )
    assert "customerID" not in X.columns and "Churn" not in X.columns
