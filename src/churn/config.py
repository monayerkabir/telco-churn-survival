"""Single source of truth for paths, seeds, schema, and modeling/business assumptions.

Everything a reviewer might want to question (offer cost, CV folds, tuning budget, ...)
lives here so that it is visible in one place and snapshotted into ``metrics.json``.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("CHURN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
# Checksum of the file the reported results were produced with; guards against silent
# upstream changes.
RAW_DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy global RNGs. Estimators also get explicit ``random_state``s."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global RNG used by some libraries


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------
ID_COL = "customerID"
TARGET_COL = "Churn"  # 1 = churned (event), 0 = still active (censored)
DURATION_COL = "tenure"  # months with the company
MAX_TENURE = 72  # dataset follow-up is capped at 72 months

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Allowed levels per categorical column, after cleaning. The first level of each list is
# the reference category when a model needs one dropped (e.g. Cox hazard ratios).
CATEGORY_LEVELS: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
CATEGORICAL_COLS = list(CATEGORY_LEVELS)
ADDON_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


# --------------------------------------------------------------------------------------
# Modeling configuration
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.20  # stratified hold-out, touched only for final evaluation
    n_splits: int = 5  # stratified K-fold on the training set (CV, tuning, calibration)


@dataclass(frozen=True)
class TuningConfig:
    n_trials: int = 40  # small Optuna budget; TPE converges quickly on ~10 params
    metric: str = "roc_auc"  # tune ranking quality; calibration fixes the probability scale


@dataclass(frozen=True)
class ClassificationConfig:
    selection_metric: str = "roc_auc"  # champion = best mean CV score on this metric
    calibration_methods: tuple[str, ...] = ("none", "sigmoid", "isotonic")
    n_bootstrap: int = 1000  # resamples for test-set confidence intervals


@dataclass(frozen=True)
class SurvivalConfig:
    cox_penalizer: float = 0.01  # small L2 penalty: stabilises correlated service dummies
    cox_strata: tuple[str, ...] = ("Contract",)  # strongest proportional-hazards violator
    rsf_n_estimators: int = 300
    rsf_min_samples_leaf: int = 15
    rsf_max_features: str = "sqrt"
    churn_horizon_months: int = 12  # horizon for "P(churn in next N months)"


@dataclass(frozen=True)
class BusinessConfig:
    """Assumptions for the retention campaign. These are *assumptions*, not data.

    A targeted customer receives an offer costing ``offer_cost``. If they would otherwise
    have churned, the offer retains them with probability ``offer_success_rate``, saving
    ``MonthlyCharges * horizon_months`` of revenue. Customers who would have stayed
    anyway still cost the offer. Profit is retained revenue minus offer cost (i.e. it
    ignores the marginal cost of serving a retained customer).
    """

    horizon_months: int = 12
    offer_cost: float = 50.0  # incentive + contact cost per targeted customer (USD)
    offer_success_rate: float = 0.30  # P(retained | targeted, would have churned)


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    split: SplitConfig = field(default_factory=SplitConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    survival: SurvivalConfig = field(default_factory=SurvivalConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG = Config()


# --------------------------------------------------------------------------------------
# Artifact locations
# --------------------------------------------------------------------------------------
LATEST_POINTER = "LATEST"  # text file in MODELS_DIR naming the most recent run


def latest_model_dir(models_dir: Path = MODELS_DIR) -> Path:
    """Directory of the most recent training run (as recorded by ``churn.train``)."""
    pointer = models_dir / LATEST_POINTER
    if not pointer.exists():
        raise FileNotFoundError(f"No trained models found in {models_dir}. Run `make train`.")
    return models_dir / pointer.read_text().strip()
