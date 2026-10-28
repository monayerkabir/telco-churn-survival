"""HTTP contract of the FastAPI service, exercised with TestClient against a model bundle
trained on synthetic data (no network, no committed artifacts needed)."""

import typing

import pytest
from fastapi.testclient import TestClient

from api.main import MODEL_DIR_ENV, N_ATTRIBUTES, create_app
from api.schemas import MAX_BATCH_SIZE, CustomerFeatures
from churn.config import CATEGORY_LEVELS
from churn.inference import ChurnPredictor


def assert_close(a, b):
    """Recursive equality with float tolerance for nested JSON payloads."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_close(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_close(x, y)
    elif isinstance(a, float):
        assert a == pytest.approx(b)
    else:
        assert a == b


@pytest.fixture(scope="module")
def client(tiny_bundle):
    with TestClient(create_app(tiny_bundle)) as c:  # context manager runs the lifespan
        yield c


def test_health_reports_the_loaded_model(client, tiny_bundle):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    predictor = ChurnPredictor.load(tiny_bundle)
    assert body == {
        "status": "ok",
        "model_version": predictor.version,
        "champion": predictor.champion,
        "survival_model": predictor.survival_model_name,
        "threshold": predictor.threshold,
    }


def test_predict_returns_every_section(client, customer):
    response = client.post("/predict", json={**customer, "customerID": "abc-1"})
    assert response.status_code == 200
    body = response.json()

    assert body["customerID"] == "abc-1"
    assert 0 <= body["churn_probability"] <= 1
    assert body["revenue_at_risk"] == {
        "horizon_months": 12,
        "amount": pytest.approx(body["churn_probability"] * customer["MonthlyCharges"] * 12),
    }
    assert body["decision"]["target"] == (
        body["churn_probability"] >= body["decision"]["threshold"]
    )
    survival = body["survival"]
    assert survival["current_tenure_months"] == customer["tenure"]
    assert [h["months_ahead"] for h in survival["survival_probability"]] == [3, 6, 12, 24]
    assert 0 <= survival["expected_remaining_tenure_months"] <= 72 - customer["tenure"]
    assert survival["curve"] is None
    assert len(body["explanation"]["top_drivers"]) == 5


def test_predict_matches_the_inference_layer(client, tiny_bundle, customer):
    body = client.post("/predict", json=customer).json()
    (direct,) = ChurnPredictor.load(tiny_bundle).predict([customer])
    assert body["churn_probability"] == pytest.approx(direct["churn_probability"])
    assert_close(body["explanation"], direct["explanation"])


def test_query_parameters(client, customer):
    body = client.post("/predict?top_k=2&include_curve=true", json=customer).json()
    assert len(body["explanation"]["top_drivers"]) == 2
    curve = body["survival"]["curve"]
    assert curve[0] == {"month": customer["tenure"], "survival": pytest.approx(1.0)}
    assert curve[-1]["month"] == 72

    assert client.post(f"/predict?top_k={N_ATTRIBUTES + 1}", json=customer).status_code == 422


def test_convenient_inputs_are_accepted(client, customer):
    """0/1 flags, omitted TotalCharges, and plain "No" for services the customer can't have."""
    no_internet = {
        **customer,
        "SeniorCitizen": 1,
        "InternetService": "No",
        **{col: "No" for col in CATEGORY_LEVELS if "No internet service" in CATEGORY_LEVELS[col]},
    }
    del no_internet["TotalCharges"]
    assert client.post("/predict", json=no_internet).status_code == 200

    explicit = {**no_internet, "SeniorCitizen": "Yes", "TotalCharges": 5 * 89.5}
    for col in CATEGORY_LEVELS:
        if "No internet service" in CATEGORY_LEVELS[col]:
            explicit[col] = "No internet service"
    a = client.post("/predict", json=no_internet).json()["churn_probability"]
    b = client.post("/predict", json=explicit).json()["churn_probability"]
    assert a == pytest.approx(b)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"Contract": "Three year"}, "Contract"),
        ({"tenure": 73}, "less than or equal to 72"),
        ({"tenure": -1}, "greater than or equal to 0"),
        ({"MonthlyCharges": 0}, "greater than 0"),
        ({"TotalCharges": -10}, "greater than or equal to 0"),
        ({"InternetService": "No", "OnlineSecurity": "Yes"}, "cannot be 'Yes'"),
        ({"OnlineSecurity": "No internet service"}, "InternetService is 'Fiber optic'"),
        ({"PhoneService": "No", "MultipleLines": "Yes"}, "PhoneService is 'No'"),
        ({"favourite_colour": "blue"}, "Extra inputs are not permitted"),
    ],
)
def test_invalid_customers_are_rejected(client, customer, change, message):
    response = client.post("/predict", json={**customer, **change})
    assert response.status_code == 422
    assert message in response.text


def test_missing_field_is_rejected(client, customer):
    del customer["Contract"]
    response = client.post("/predict", json=customer)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "Contract"]


def test_batch_matches_single_predictions_in_order(client, customer):
    customers = [
        {**customer, "customerID": "a", "tenure": 1},
        {**customer, "customerID": "b", "tenure": 40, "Contract": "Two year"},
        {**customer, "customerID": "c", "tenure": 70},
    ]
    response = client.post("/predict/batch", json={"customers": customers})
    assert response.status_code == 200
    batch = response.json()["predictions"]
    assert [p["customerID"] for p in batch] == ["a", "b", "c"]
    for c, p in zip(customers, batch, strict=True):
        single = client.post("/predict", json=c).json()
        assert p["churn_probability"] == pytest.approx(single["churn_probability"])
        assert_close(p["survival"], single["survival"])


@pytest.mark.parametrize("size", [0, MAX_BATCH_SIZE + 1])
def test_batch_size_limits(client, customer, size):
    response = client.post("/predict/batch", json={"customers": [customer] * size})
    assert response.status_code == 422


def test_openapi_documents_the_example(client):
    schema = client.get("/openapi.json").json()
    assert {"/health", "/predict", "/predict/batch"} <= set(schema["paths"])
    assert schema["components"]["schemas"]["CustomerFeatures"]["examples"]


def test_model_dir_comes_from_the_environment(monkeypatch, tiny_bundle):
    monkeypatch.setenv(MODEL_DIR_ENV, str(tiny_bundle))
    with TestClient(create_app()) as c:
        assert c.get("/health").status_code == 200


def test_schema_levels_match_the_training_schema():
    """The API's allowed values must stay in sync with churn.config.CATEGORY_LEVELS."""
    for col, levels in CATEGORY_LEVELS.items():
        annotation = CustomerFeatures.model_fields[col].annotation
        assert set(typing.get_args(annotation)) == set(levels), col
