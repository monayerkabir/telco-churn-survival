"""The shared inference layer and the deployment bundle."""

import json

import numpy as np
import pandera.errors
import pytest
from sklearn.linear_model import LogisticRegression

from churn.bundle import BUNDLE_DIR, MANIFEST, verify_bundle
from churn.inference import FEATURE_COLUMNS, ChurnPredictor
from churn.train import make_logistic_pipeline


@pytest.fixture(scope="module")
def predictor(tiny_bundle):
    return ChurnPredictor.load(tiny_bundle)


def test_bundle_is_slim_and_verified(trained_run, tiny_bundle):
    manifest = json.loads((tiny_bundle / MANIFEST).read_text())
    assert set(manifest["sha256"]) == {
        "base_classifiers.joblib",
        "classifier.joblib",
        "feature_means.json",
        "metrics.json",
        "survival_models.joblib",
    }
    assert verify_bundle(tiny_bundle) == []
    bundle_size = sum(p.stat().st_size for p in tiny_bundle.iterdir())
    run_size = sum(p.stat().st_size for p in trained_run.glob("*.joblib"))
    assert bundle_size < run_size


def test_bundle_predicts_exactly_like_the_training_run(trained_run, tiny_bundle, clean_df):
    X = clean_df.drop(columns=["customerID", "Churn"])
    from_run = ChurnPredictor.load(trained_run).score(X)
    from_bundle = ChurnPredictor.load(tiny_bundle).score(X)
    np.testing.assert_allclose(from_run.to_numpy(float), from_bundle.to_numpy(float))


def test_tampered_bundle_is_rejected(tiny_bundle, tmp_path):
    import shutil

    copy = tmp_path / "bundle"
    shutil.copytree(tiny_bundle, copy)
    (copy / "metrics.json").write_text((copy / "metrics.json").read_text() + " ")
    with pytest.raises(ValueError, match="failed verification"):
        ChurnPredictor.load(copy)


def test_prepare_fills_total_charges_and_coerces_types(predictor, customer):
    customer.pop("TotalCharges")
    customer["SeniorCitizen"] = 1
    X = predictor.prepare([customer])
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.loc[0, "TotalCharges"] == pytest.approx(5 * 89.5)
    assert X.loc[0, "SeniorCitizen"] == "Yes"


def test_prepare_rejects_missing_and_inconsistent_fields(predictor, customer):
    with pytest.raises(ValueError, match="Missing customer fields"):
        predictor.prepare([{k: v for k, v in customer.items() if k != "Contract"}])
    with pytest.raises(pandera.errors.SchemaErrors):
        predictor.prepare([{**customer, "InternetService": "No"}])  # add-ons still "Yes"/"No"


def test_prediction_payload(predictor, customer):
    (pred,) = predictor.predict([customer], top_k=3, include_curve=True)

    assert 0 <= pred["churn_probability"] <= 1
    assert pred["revenue_at_risk"]["amount"] == pytest.approx(
        pred["churn_probability"] * customer["MonthlyCharges"] * 12
    )
    assert pred["decision"]["target"] == (pred["churn_probability"] >= predictor.threshold)

    surv = pred["survival"]
    assert surv["current_tenure_months"] == 5
    assert 0 <= surv["churn_probability_12m"] <= 1
    assert 0 <= surv["expected_active_months_next_12m"] <= 12
    curve = [point["survival"] for point in surv["curve"]]
    assert surv["curve"][0]["month"] == 5 and curve[0] == pytest.approx(1.0)
    assert np.all(np.diff(curve) <= 1e-12)

    drivers = pred["explanation"]["top_drivers"]
    assert len(drivers) == 3
    magnitudes = [abs(d["contribution"]) for d in drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explanation_is_additive(predictor, clean_df):
    """base value + all attribute contributions = the champion's log-odds."""
    X = predictor.prepare(clean_df.drop(columns=["customerID", "Churn"]))
    base, contrib = predictor.explain(X)
    pipe = predictor.explained_model
    raw = pipe["model"].predict(pipe["preprocess"].transform(X), raw_score=True)
    np.testing.assert_allclose(base + contrib.sum(axis=1), raw, atol=1e-6)


def test_linear_champion_is_explained_with_feature_means(predictor, clean_df):
    X, y = clean_df.drop(columns=["customerID", "Churn"]), clean_df["Churn"]
    pipe = make_logistic_pipeline().fit(X, y)
    means = pipe["preprocess"].transform(X).mean().to_dict()
    linear = ChurnPredictor(pipe, pipe, predictor.survival_model, predictor.metrics, means)
    assert isinstance(linear.explained_model["model"], LogisticRegression)

    base, contrib = linear.explain(linear.prepare(X))
    np.testing.assert_allclose(base + contrib.sum(axis=1), pipe.decision_function(X))


def test_horizon_summaries_are_null_beyond_the_observation_window(predictor, customer):
    (pred,) = predictor.predict([{**customer, "tenure": 66}])
    surv = pred["survival"]
    assert surv["churn_probability_12m"] is None  # 66 + 12 > 72 months observed
    by_horizon = {p["months_ahead"]: p["probability"] for p in surv["survival_probability"]}
    assert by_horizon[3] is not None and by_horizon[12] is None
    assert surv["curve"] is None  # only returned on request


@pytest.mark.skipif(not BUNDLE_DIR.exists(), reason="no committed model bundle")
def test_committed_bundle_loads_with_pinned_libraries(customer):
    """The repository's model_bundle/ must load (and verify) with the pinned versions."""
    assert verify_bundle(BUNDLE_DIR) == []
    (pred,) = ChurnPredictor.load(BUNDLE_DIR).predict([customer])
    assert 0 <= pred["churn_probability"] <= 1
