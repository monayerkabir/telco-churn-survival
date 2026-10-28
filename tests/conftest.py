"""Shared fixtures. Tests never touch the network: they use a synthetic dataset that
mimics the raw CSV format, including its quirks (text TotalCharges with blanks)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.bundle import export_bundle
from churn.config import ADDON_SERVICE_COLS, CONFIG
from churn.data import clean_data

# Small budgets so a full training run on synthetic data takes a few seconds.
TINY_CONFIG = dataclasses.replace(
    CONFIG,
    split=dataclasses.replace(CONFIG.split, n_splits=3),
    tuning=dataclasses.replace(CONFIG.tuning, n_trials=2),
    classification=dataclasses.replace(CONFIG.classification, n_bootstrap=20),
    survival=dataclasses.replace(CONFIG.survival, rsf_n_estimators=10),
)


def make_raw_telco(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic data in the raw IBM Telco format with a learnable churn signal."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    tenure = rng.integers(1, 73, n)
    tenure[:5] = 0  # brand-new customers -> blank TotalCharges, like the real data
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    monthly = rng.uniform(18, 120, n).round(2)

    logit = (
        -0.5
        + 1.5 * (contract == "Month-to-month")
        - 0.04 * tenure
        + 0.6 * (internet == "Fiber optic")
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    churn[tenure == 0] = False

    total = (tenure * monthly * rng.uniform(0.95, 1.05, n)).round(2).astype(str)
    total = np.where(tenure == 0, " ", total)

    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(
                phone == "No", "No phone service", rng.choice(["Yes", "No"], n)
            ),
            "InternetService": internet,
        }
    )
    for col in ADDON_SERVICE_COLS:
        df[col] = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))
    df["Contract"] = contract
    df["PaperlessBilling"] = rng.choice(["Yes", "No"], n)
    df["PaymentMethod"] = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
    )
    df["MonthlyCharges"] = monthly
    df["TotalCharges"] = total
    df["Churn"] = np.where(churn, "Yes", "No")
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_telco()


@pytest.fixture
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(raw_df)


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory) -> Path:
    """One full training run on synthetic data, shared by the inference/API/app tests."""
    from churn.train import run  # imports optuna; keep it out of collection for other tests

    df = clean_data(make_raw_telco())
    models_dir = tmp_path_factory.mktemp("models")
    return run(df, TINY_CONFIG, models_dir=models_dir, reports_dir=None, data_path=None)


@pytest.fixture(scope="session")
def tiny_bundle(trained_run: Path, tmp_path_factory) -> Path:
    """Deployment bundle exported from :func:`trained_run`."""
    return export_bundle(trained_run, tmp_path_factory.mktemp("bundle") / "model_bundle")


@pytest.fixture
def customer() -> dict:
    """A valid customer record in the API/inference input format."""
    return {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 447.5,
    }
