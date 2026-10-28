"""Smoke tests: every model fits and predicts sensibly on synthetic data, and the full
training pipeline runs end to end and is reproducible."""

import dataclasses
import json

import numpy as np
import pytest

from churn import survival
from churn.config import CONFIG, LATEST_POINTER
from churn.train import make_calibrated, make_lgbm_pipeline, make_logistic_pipeline, run

TINY = dataclasses.replace(
    CONFIG,
    split=dataclasses.replace(CONFIG.split, n_splits=3),
    tuning=dataclasses.replace(CONFIG.tuning, n_trials=2),
    classification=dataclasses.replace(CONFIG.classification, n_bootstrap=20),
    survival=dataclasses.replace(CONFIG.survival, rsf_n_estimators=10),
)


@pytest.mark.parametrize(
    "make_model",
    [
        make_logistic_pipeline,
        lambda: make_lgbm_pipeline(n_estimators=20),
        lambda: make_calibrated(make_lgbm_pipeline(n_estimators=20), "isotonic", TINY),
        lambda: make_calibrated(make_logistic_pipeline(), "sigmoid", TINY),
    ],
    ids=["logreg", "lgbm", "lgbm_isotonic", "logreg_sigmoid"],
)
def test_classifier_smoke(clean_df, make_model):
    X, y = clean_df, clean_df["Churn"]
    proba = make_model().fit(X, y).predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", list(survival.build_survival_models()))
def test_survival_model_smoke(clean_df, name):
    model = survival.build_survival_models(TINY.survival)[name]
    df = survival.survival_frame(clean_df)
    duration, event = survival.survival_arrays(df)

    surv = model.fit(df, duration, event).predict_survival(df)

    assert surv.shape == (len(df), len(survival.TIME_GRID))
    assert np.all((surv >= 0) & (surv <= 1))
    assert np.all(np.diff(surv, axis=1) <= 1e-9), "survival curves must be non-increasing"


def test_stratified_cox_preserves_row_order(clean_df):
    df = survival.survival_frame(clean_df)
    model = survival.CoxPHModel(strata=("Contract",)).fit(df, *survival.survival_arrays(df))
    batch = model.predict_survival(df.head(15))
    one_by_one = np.vstack(
        [model.predict_survival(df.iloc[[i]].reset_index(drop=True)) for i in range(15)]
    )
    np.testing.assert_allclose(batch, one_by_one)


def test_survival_summaries_on_known_curve():
    # Constant monthly survival of 0.9 -> S(t) = 0.9 ** t on the 0..72 grid.
    t0 = np.array([0, 10, 70])
    surv = np.tile(0.9 ** survival.TIME_GRID.astype(float), (len(t0), 1))

    p12 = survival.churn_probability_within(surv, t0, horizon=12)
    np.testing.assert_allclose(p12[:2], 1 - 0.9**12)
    assert np.isnan(p12[2])  # 70 + 12 is beyond the 72-month observation window

    remaining = survival.expected_remaining_tenure(surv, t0)
    # memoryless: residual life from t0 is a geometric sum over the remaining window
    np.testing.assert_allclose(remaining, [(1 - 0.9 ** (72 - t)) / 0.1 for t in t0])


def test_training_pipeline_end_to_end_is_reproducible(clean_df, tmp_path):
    out1 = run(clean_df, TINY, models_dir=tmp_path / "models", reports_dir=None, data_path=None)
    out2 = run(clean_df, TINY, models_dir=tmp_path / "models2", reports_dir=None, data_path=None)

    for name in ["classifier.joblib", "survival_models.joblib", "metrics.json"]:
        assert (out1 / name).exists()
    assert (tmp_path / "models" / LATEST_POINTER).read_text().strip() == out1.name

    m1, m2 = (json.loads((o / "metrics.json").read_text()) for o in (out1, out2))
    assert m1["classification"]["test"]["final"]["roc_auc"]["value"] > 0.6
    # Synthetic churn is not generated as a survival process; just check it beats chance.
    assert m1["survival"]["models"]["Cox PH"]["c_index"] > 0.5
    for section in ["classification", "business", "survival"]:
        assert m1[section] == m2[section], f"{section} metrics are not reproducible"
