"""Scoring customers with trained artifacts: the single code path behind the API and the app.

For each customer it returns the churn probability, the retention decision, revenue at
risk, a survival summary (conditional on the customer's current tenure) and the SHAP
drivers behind the prediction.

Artifacts are read from a directory laid out like a training run (``models/<version>/``)
or like the slim deployment bundle produced by :mod:`churn.bundle`; both share these files:

    classifier.joblib        deployed classifier (champion + selected calibration)
    base_classifiers.joblib  {name: uncalibrated pipeline}; the champion is explained
    survival_models.joblib   {name: survival model}; the primary model is used
    metrics.json             names of champion / primary model, threshold, assumptions
    feature_means.json       SHAP baseline for a linear champion (optional for trees)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn import business, survival
from churn.bundle import verify_bundle
from churn.config import (
    CATEGORY_LEVELS,
    DURATION_COL,
    ID_COL,
    MAX_TENURE,
    TARGET_COL,
    BusinessConfig,
)
from churn.data import SCHEMA
from churn.features import TelcoFeatureEngineer, source_feature

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [*CATEGORY_LEVELS, DURATION_COL, "MonthlyCharges", "TotalCharges"]
SURVIVAL_HORIZONS = (3, 6, 12, 24)  # months ahead reported in the survival summary
CHURN_HORIZON = 12  # months, for P(churn within) and expected active months

# Same column and cross-column checks as training data, minus the label and the ID.
FEATURE_SCHEMA = SCHEMA.remove_columns([TARGET_COL, ID_COL])


class ChurnPredictor:
    """Loads the deployed models once and scores customers in batches."""

    def __init__(
        self,
        classifier,
        explained_model,
        survival_model,
        metrics: dict,
        feature_means: Mapping[str, float] | None = None,
        source: Path | None = None,
    ):
        self.classifier = classifier
        self.explained_model = explained_model
        self.survival_model = survival_model
        self.metrics = metrics
        self.source = source
        self._init_explainer(feature_means)

    @classmethod
    def load(cls, model_dir: str | Path) -> ChurnPredictor:
        """Load from a run directory or bundle; refuses files that fail their checksum."""
        model_dir = Path(model_dir)
        problems = verify_bundle(model_dir)
        corrupt = [p for p in problems if "checksum" in p]
        if corrupt:
            raise ValueError(f"Model bundle {model_dir} failed verification: {corrupt}")
        for problem in problems:
            logger.warning("Library version differs from the model bundle: %s", problem)

        metrics = json.loads((model_dir / "metrics.json").read_text())
        champion = metrics["classification"]["champion"]
        primary = metrics["survival"]["primary_model"]
        means_path = model_dir / "feature_means.json"
        return cls(
            classifier=joblib.load(model_dir / "classifier.joblib"),
            explained_model=joblib.load(model_dir / "base_classifiers.joblib")[champion],
            survival_model=joblib.load(model_dir / "survival_models.joblib")[primary],
            metrics=metrics,
            feature_means=json.loads(means_path.read_text()) if means_path.exists() else None,
            source=model_dir,
        )

    # ----------------------------------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------------------------------
    @property
    def version(self) -> str:
        return self.metrics["run"]["version"]

    @property
    def champion(self) -> str:
        return self.metrics["classification"]["champion"]

    @property
    def survival_model_name(self) -> str:
        return self.metrics["survival"]["primary_model"]

    @property
    def threshold(self) -> float:
        """Retention threshold chosen on out-of-fold training predictions."""
        return float(self.metrics["business"]["threshold"])

    @property
    def business_config(self) -> BusinessConfig:
        """The assumptions the threshold was optimised for."""
        return BusinessConfig(**self.metrics["business"]["assumptions"])

    # ----------------------------------------------------------------------------------
    # Input handling
    # ----------------------------------------------------------------------------------
    def prepare(self, customers: pd.DataFrame | Sequence[Mapping]) -> pd.DataFrame:
        """Coerce raw customer records to the training schema and validate them.

        ``TotalCharges`` may be omitted (or null), in which case it is filled with
        ``tenure * MonthlyCharges``, the same rule used when cleaning the training data.
        ``SeniorCitizen`` accepts 0/1 as well as "No"/"Yes".
        """
        df = pd.DataFrame(customers).reset_index(drop=True)
        missing = set(FEATURE_COLUMNS) - {"TotalCharges"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing customer fields: {sorted(missing)}")
        if "TotalCharges" not in df.columns:
            df["TotalCharges"] = np.nan

        df["SeniorCitizen"] = df["SeniorCitizen"].map(_yes_no)
        for col in CATEGORY_LEVELS:
            df[col] = df[col].astype(str).str.strip()
        df[DURATION_COL] = pd.to_numeric(df[DURATION_COL]).astype("int64")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"]).astype("float64")
        total = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = total.fillna(df[DURATION_COL] * df["MonthlyCharges"]).astype("float64")

        return FEATURE_SCHEMA.validate(df[FEATURE_COLUMNS], lazy=True)

    # ----------------------------------------------------------------------------------
    # Model outputs
    # ----------------------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(X)[:, 1]

    def survival_curves(self, X: pd.DataFrame) -> np.ndarray:
        """Unconditional S(t) on months 0..72, shape (n_customers, 73)."""
        return self.survival_model.predict_survival(X, survival.TIME_GRID)

    def _init_explainer(self, feature_means: Mapping[str, float] | None) -> None:
        model = self.explained_model["model"]
        if hasattr(model, "coef_"):  # linear champion: exact SHAP under feature independence
            if feature_means is None:
                raise ValueError("A linear champion needs feature_means.json to be explained")
            self._means = pd.Series(feature_means, dtype=float)
            self._tree_explainer = None
        else:
            import shap  # heavy import, only needed for tree models

            self._tree_explainer = shap.TreeExplainer(model)

    def explain(self, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """SHAP values of the uncalibrated champion, in log-odds, summed per attribute.

        Returns the base values (one per customer) and a frame of contributions with one
        column per business attribute (one-hot dummies are summed back to their source
        column, which is valid because SHAP values are additive).
        """
        Z = self.explained_model["preprocess"].transform(X)
        model = self.explained_model["model"]
        if self._tree_explainer is not None:
            sv = self._tree_explainer(Z)
            values, base = sv.values, np.asarray(sv.base_values, dtype=float).reshape(-1)
        else:
            means = self._means.reindex(Z.columns).to_numpy()
            coef = model.coef_.reshape(-1)
            values = (Z.to_numpy() - means) * coef
            base = np.full(len(Z), float(model.intercept_[0] + coef @ means))
        grouped = (
            pd.DataFrame(values, columns=Z.columns).T.groupby(source_feature, sort=False).sum().T
        )
        return base, grouped

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Numeric scores for many customers at once (one row per customer)."""
        cfg = self.business_config
        p = self.predict_proba(X)
        charges = X["MonthlyCharges"].to_numpy()
        tenure = X[DURATION_COL].to_numpy()
        value = business.customer_value(charges, cfg.horizon_months)
        surv = self.survival_curves(X)
        months_12 = survival.expected_months_within(surv, tenure, CHURN_HORIZON)

        out = pd.DataFrame(
            {
                "p_churn": p,
                "revenue_at_risk": business.revenue_at_risk(p, charges, cfg.horizon_months),
                "target": p >= self.threshold,
                "value_aware_target": business.value_aware_targets(p, charges, cfg),
                "breakeven_probability": business.breakeven_probability(charges, cfg),
                "expected_profit_if_targeted": p * cfg.offer_success_rate * value - cfg.offer_cost,
                "p_churn_12m_survival": survival.churn_probability_within(
                    surv, tenure, CHURN_HORIZON
                ),
                "expected_active_months_12m": months_12,
                "expected_revenue_lost_12m": charges * (CHURN_HORIZON - months_12),
                "expected_remaining_tenure": survival.expected_remaining_tenure(surv, tenure),
            },
            index=X.index,
        )
        cond = survival.conditional_survival(surv, tenure)
        for h in SURVIVAL_HORIZONS:
            end = tenure + h
            ok = end <= MAX_TENURE
            col = np.full(len(X), np.nan)
            col[ok] = cond[np.flatnonzero(ok), end[ok]]
            out[f"survival_{h}m"] = col
        return out

    # ----------------------------------------------------------------------------------
    # JSON-ready predictions (API)
    # ----------------------------------------------------------------------------------
    def predict(
        self,
        customers: pd.DataFrame | Sequence[Mapping],
        top_k: int = 5,
        include_curve: bool = False,
    ) -> list[dict]:
        """Full prediction payload per customer, in input order."""
        X = self.prepare(customers)
        scores = self.score(X)
        base, contrib = self.explain(X)
        display = display_values(X)
        curves = (
            survival.conditional_survival(self.survival_curves(X), X[DURATION_COL])
            if include_curve
            else None
        )
        cfg = self.business_config

        results = []
        for i in range(len(X)):
            s = scores.iloc[i]
            tenure = int(X[DURATION_COL].iloc[i])
            row = contrib.iloc[i]
            order = row.abs().sort_values(ascending=False).index
            top, rest = order[:top_k], order[top_k:]
            results.append(
                {
                    "churn_probability": float(s["p_churn"]),
                    "revenue_at_risk": {
                        "horizon_months": cfg.horizon_months,
                        "amount": float(s["revenue_at_risk"]),
                    },
                    "decision": {
                        "target": bool(s["target"]),
                        "threshold": self.threshold,
                        "value_aware_target": bool(s["value_aware_target"]),
                        "breakeven_probability": float(s["breakeven_probability"]),
                        "expected_profit_if_targeted": float(s["expected_profit_if_targeted"]),
                    },
                    "survival": {
                        "model": self.survival_model_name,
                        "current_tenure_months": tenure,
                        "survival_probability": [
                            {"months_ahead": h, "probability": _none_if_nan(s[f"survival_{h}m"])}
                            for h in SURVIVAL_HORIZONS
                        ],
                        "churn_probability_12m": _none_if_nan(s["p_churn_12m_survival"]),
                        "expected_active_months_next_12m": _none_if_nan(
                            s["expected_active_months_12m"]
                        ),
                        "expected_revenue_lost_next_12m": _none_if_nan(
                            s["expected_revenue_lost_12m"]
                        ),
                        "expected_remaining_tenure_months": float(s["expected_remaining_tenure"]),
                        "observation_window_months": MAX_TENURE,
                        "curve": (
                            [
                                {"month": int(t), "survival": float(curves[i, t])}
                                for t in survival.TIME_GRID
                                if t >= tenure
                            ]
                            if curves is not None
                            else None
                        ),
                    },
                    "explanation": {
                        "base_log_odds": float(base[i]),
                        "top_drivers": [
                            {
                                "feature": f,
                                "value": display.iloc[i][f],
                                "contribution": float(row[f]),
                                "direction": "increases_risk" if row[f] > 0 else "decreases_risk",
                            }
                            for f in top
                        ],
                        "other_features_contribution": float(row[rest].sum()),
                    },
                }
            )
        return results


def display_values(X: pd.DataFrame) -> pd.DataFrame:
    """Human-readable value of every explained attribute: raw categories, numeric inputs
    and engineered features (e.g. ``avg_monthly_charge``)."""
    engineered = TelcoFeatureEngineer(include_tenure=True).fit_transform(X)
    out = engineered.astype(object)
    for col in CATEGORY_LEVELS:
        out[col] = X[col].to_numpy()
    for col in out.columns.difference(list(CATEGORY_LEVELS)):
        out[col] = [round(float(v), 2) for v in engineered[col]]
    return out


def _yes_no(value):
    """0/1 and booleans (the raw data's encoding) to "No"/"Yes"; anything else unchanged."""
    if isinstance(value, bool | int | np.integer | np.bool_):
        return "Yes" if value else "No"
    return value


def _none_if_nan(value) -> float | None:
    value = float(value)
    return None if np.isnan(value) else value
