"""End-to-end training pipeline.

1. Stratified train/test split (the test set is only used for the final evaluation).
2. Classification on the training set with stratified K-fold CV:
   logistic-regression baseline vs LightGBM (default and Optuna-tuned).
3. Probability calibration (none / Platt / isotonic) chosen by out-of-fold Brier score.
4. Business layer: retention threshold chosen on out-of-fold predictions, then
   evaluated on the test set.
5. Survival models (Kaplan-Meier, Cox PH, stratified Cox PH, random survival forest).
6. Versioned artifacts and ``metrics.json`` written to ``models/<version>/``.

Usage:
    python -m churn.train [--n-trials N] [--models-dir DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from churn import business, survival
from churn.config import (
    CONFIG,
    DURATION_COL,
    ID_COL,
    LATEST_POINTER,
    MODELS_DIR,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COL,
    Config,
    set_global_seed,
)
from churn.data import load_clean_data, sha256_of, split_train_test
from churn.evaluate import (
    LOWER_IS_BETTER,
    SKLEARN_SCORERS,
    classification_metrics,
    classification_metrics_with_ci,
)
from churn.features import build_preprocessor

logger = logging.getLogger(__name__)

N_JOBS = -1  # parallelism across CV folds; each LightGBM fit is single-threaded

LGBM_FIXED_PARAMS = {
    "objective": "binary",
    "n_jobs": 1,
    "verbose": -1,
    # Deterministic, thread-count independent training -> reproducible results.
    "deterministic": True,
    "force_row_wise": True,
}

MODEL_NAMES = {
    "logreg": "Logistic regression",
    "lgbm_default": "LightGBM (default)",
    "lgbm_tuned": "LightGBM (tuned)",
}


# --------------------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------------------
def make_logistic_pipeline(seed: int = CONFIG.seed) -> Pipeline:
    """Baseline: one-hot + standardised features into an L2 logistic regression."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor("linear")),
            ("model", LogisticRegression(C=1.0, max_iter=5000, random_state=seed)),
        ]
    )


def make_lgbm_pipeline(seed: int = CONFIG.seed, **params) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor("tree")),
            ("model", LGBMClassifier(random_state=seed, **LGBM_FIXED_PARAMS, **params)),
        ]
    )


def lgbm_search_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def make_calibrated(estimator: Pipeline, method: str, cfg: Config = CONFIG):
    """Wrap ``estimator`` in cross-validated calibration (``method="none"`` = unchanged).

    ``CalibratedClassifierCV`` fits the whole pipeline (preprocessing included) on each
    inner training fold and the calibrator on the matching held-out fold, so the
    calibration map is never learnt on data the model was trained on.
    """
    if method == "none":
        return clone(estimator)
    inner_cv = StratifiedKFold(cfg.split.n_splits, shuffle=True, random_state=cfg.seed)
    return CalibratedClassifierCV(clone(estimator), method=method, cv=inner_cv, ensemble=True)


# --------------------------------------------------------------------------------------
# Classification steps
# --------------------------------------------------------------------------------------
def cv_splitter(cfg: Config = CONFIG) -> StratifiedKFold:
    return StratifiedKFold(cfg.split.n_splits, shuffle=True, random_state=cfg.seed)


def cross_validate_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv) -> pd.DataFrame:
    """Per-fold ROC-AUC, PR-AUC, Brier and log loss (losses reported as positive)."""
    res = cross_validate(pipe, X, y, cv=cv, scoring=SKLEARN_SCORERS, n_jobs=N_JOBS)
    folds = pd.DataFrame({m: res[f"test_{m}"] for m in SKLEARN_SCORERS})
    for m in LOWER_IS_BETTER:
        folds[m] = -folds[m]
    folds.insert(0, "fold", np.arange(len(folds)))
    return folds


def tune_lightgbm(X: pd.DataFrame, y: pd.Series, cfg: Config = CONFIG) -> optuna.Study:
    """Optuna TPE search maximising mean CV ROC-AUC (preprocessing refit in every fold)."""
    cv = cv_splitter(cfg)
    scorer = SKLEARN_SCORERS[cfg.tuning.metric]

    def objective(trial: optuna.Trial) -> float:
        pipe = make_lgbm_pipeline(cfg.seed, **lgbm_search_space(trial))
        res = cross_validate(pipe, X, y, cv=cv, scoring=scorer, n_jobs=N_JOBS)
        return float(np.mean(res["test_score"]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        study_name="lightgbm_tuning",
    )
    study.optimize(objective, n_trials=cfg.tuning.n_trials)
    logger.info("Best CV %s %.4f with %s", cfg.tuning.metric, study.best_value, study.best_params)
    return study


def tuned_params(study: optuna.Study) -> dict:
    return {**study.best_params, "subsample_freq": 1}


def oof_probabilities(estimator, X: pd.DataFrame, y: pd.Series, cv) -> np.ndarray:
    return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba", n_jobs=N_JOBS)[:, 1]


def train_classification(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Config = CONFIG
) -> tuple[dict, dict, dict]:
    """Returns (metrics, fitted models, tables to persist)."""
    X_train, y_train = train_df, train_df[TARGET_COL]
    X_test, y_test = test_df, test_df[TARGET_COL]
    cv = cv_splitter(cfg)

    logger.info("Tuning LightGBM with Optuna (%d trials)", cfg.tuning.n_trials)
    study = tune_lightgbm(X_train, y_train, cfg)

    candidates = {
        "logreg": make_logistic_pipeline(cfg.seed),
        "lgbm_default": make_lgbm_pipeline(cfg.seed),
        "lgbm_tuned": make_lgbm_pipeline(cfg.seed, **tuned_params(study)),
    }

    logger.info("Cross-validating candidates")
    cv_tables = []
    for key, pipe in candidates.items():
        folds = cross_validate_model(pipe, X_train, y_train, cv)
        folds.insert(0, "model", key)
        cv_tables.append(folds)
    cv_results = pd.concat(cv_tables, ignore_index=True)
    cv_summary = {
        key: {
            m: {"mean": float(g[m].mean()), "std": float(g[m].std(ddof=1))} for m in SKLEARN_SCORERS
        }
        for key, g in cv_results.groupby("model", sort=False)
    }
    metric = cfg.classification.selection_metric
    champion_key = max(cv_summary, key=lambda k: cv_summary[k][metric]["mean"])
    logger.info("Champion by CV %s: %s", metric, champion_key)
    champion = candidates[champion_key]

    logger.info("Selecting calibration method by out-of-fold Brier score")
    oof = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET_COL: y_train})
    for method in cfg.classification.calibration_methods:
        oof[f"p_{method}"] = oof_probabilities(
            make_calibrated(champion, method, cfg), X_train, y_train, cv
        )
    oof_brier = {
        m: classification_metrics(y_train, oof[f"p_{m}"])["brier"]
        for m in cfg.classification.calibration_methods
    }
    calibration_method = min(oof_brier, key=oof_brier.get)
    logger.info("Calibration: %s (OOF Brier %s)", calibration_method, oof_brier)

    # Final models are refit on the full training set.
    base_models = {key: clone(pipe).fit(X_train, y_train) for key, pipe in candidates.items()}
    classifier = make_calibrated(champion, calibration_method, cfg).fit(X_train, y_train)

    p_test_uncal = base_models[champion_key].predict_proba(X_test)[:, 1]
    p_test = classifier.predict_proba(X_test)[:, 1]
    test_metrics = {
        # "final" is the deployed model (champion + selected calibration, possibly "none").
        "final": classification_metrics_with_ci(
            y_test, p_test, n_boot=cfg.classification.n_bootstrap, seed=cfg.seed
        ),
        "uncalibrated": classification_metrics(y_test, p_test_uncal),
    }

    metrics = {
        "cv": cv_summary,
        "champion": champion_key,
        "tuning": {
            "n_trials": cfg.tuning.n_trials,
            "metric": cfg.tuning.metric,
            "best_cv_score": float(study.best_value),
            "best_params": tuned_params(study),
        },
        "calibration": {"oof_brier": oof_brier, "selected": calibration_method},
        "test": test_metrics,
    }
    models = {"classifier": classifier, "base_models": base_models}
    tables = {
        "cv_results": cv_results,
        "optuna_trials": study.trials_dataframe(attrs=("number", "value", "params", "state")),
        "oof_predictions": oof.assign(p_churn=oof[f"p_{calibration_method}"]),
        "test_predictions": pd.DataFrame(
            {
                ID_COL: test_df[ID_COL],
                TARGET_COL: y_test,
                DURATION_COL: test_df[DURATION_COL],
                "MonthlyCharges": test_df["MonthlyCharges"],
                "Contract": test_df["Contract"],
                "p_uncalibrated": p_test_uncal,
                "p_churn": p_test,
            }
        ),
    }
    return metrics, models, tables


# --------------------------------------------------------------------------------------
# Business step
# --------------------------------------------------------------------------------------
def evaluate_business(
    oof: pd.DataFrame, train_df: pd.DataFrame, test_pred: pd.DataFrame, cfg: Config = CONFIG
) -> tuple[dict, dict]:
    """Pick the threshold on out-of-fold predictions; report its profit on the test set."""
    bcfg = cfg.business
    curve_oof = business.profit_curve(
        oof[TARGET_COL], oof["p_churn"], train_df["MonthlyCharges"], cfg=bcfg
    )
    best = business.optimal_threshold(curve_oof)
    threshold = float(best["threshold"])

    y, p, charges = test_pred[TARGET_COL], test_pred["p_churn"], test_pred["MonthlyCharges"]
    curve_test = business.profit_curve(y, p, charges, cfg=bcfg)
    at = curve_test.set_index("threshold").loc[threshold]
    hindsight = business.optimal_threshold(curve_test)
    rar = business.revenue_at_risk(p, charges, bcfg.horizon_months)
    targeted = p.to_numpy() >= threshold
    value_aware = business.value_aware_targets(p, charges, bcfg)  # per-customer breakeven

    metrics = {
        "assumptions": dataclasses.asdict(bcfg),
        "threshold": threshold,
        "train_oof": {
            "profit": float(best["profit"]),
            "share_targeted": float(best["share_targeted"]),
        },
        "test": {
            "n_customers": len(test_pred),
            "n_targeted": int(at["n_targeted"]),
            "share_targeted": float(at["share_targeted"]),
            "churners_reached": int(at["churners_reached"]),
            "churners_total": int(y.sum()),
            "profit": float(at["profit"]),
            "profit_target_everyone": float(curve_test.iloc[0]["profit"]),
            "hindsight_best_threshold": float(hindsight["threshold"]),
            "hindsight_best_profit": float(hindsight["profit"]),
            "revenue_at_risk_total": float(rar.sum()),
            "revenue_at_risk_targeted": float(rar[targeted].sum()),
            "forecast_profit": business.expected_profit(p, p, charges, threshold, bcfg),
            "value_aware_n_targeted": int(value_aware.sum()),
            "value_aware_churners_reached": int(y.to_numpy()[value_aware].sum()),
            "value_aware_profit": business.policy_profit(y, value_aware, charges, bcfg),
        },
    }
    tables = {"profit_curve_oof": curve_oof, "profit_curve_test": curve_test}
    return metrics, {"revenue_at_risk": rar, "targeted": targeted, "tables": tables}


# --------------------------------------------------------------------------------------
# Survival step
# --------------------------------------------------------------------------------------
def train_survival(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Config = CONFIG
) -> tuple[dict, dict, dict]:
    s_train, s_test = survival.survival_frame(train_df), survival.survival_frame(test_df)
    duration, event = survival.survival_arrays(s_train)

    models, results = {}, {}
    for name, model in survival.build_survival_models(cfg.survival, cfg.seed).items():
        logger.info("Fitting survival model: %s", name)
        models[name] = model.fit(s_train, duration, event)
        results[name] = survival.evaluate_survival_model(models[name], s_train, s_test)

    # Primary model = best integrated Brier score (rewards calibration + discrimination).
    primary = min(results, key=lambda k: results[k]["integrated_brier_score"])
    surv = models[primary].predict_survival(s_test)
    horizon = cfg.survival.churn_horizon_months
    per_customer = pd.DataFrame(
        {
            ID_COL: s_test[ID_COL],
            "expected_remaining_tenure": survival.expected_remaining_tenure(
                surv, s_test[DURATION_COL]
            ),
            f"expected_months_next_{horizon}m": survival.expected_months_within(
                surv, s_test[DURATION_COL], horizon
            ),
            f"p_churn_{horizon}m_survival": survival.churn_probability_within(
                surv, s_test[DURATION_COL], horizon
            ),
        }
    )

    ph_tables = {
        "ph_test_cox": models["Cox PH"].proportional_hazards_test(),
        "ph_test_cox_stratified": models[
            "Cox PH (stratified by contract)"
        ].proportional_hazards_test(),
    }
    alpha = 0.05
    metrics = {
        "n_train": len(s_train),
        "n_test": len(s_test),
        "dropped_zero_tenure": int(len(train_df) - len(s_train) + len(test_df) - len(s_test)),
        "models": results,
        "primary_model": primary,
        "ph_test": {
            name: {
                "n_covariates": len(t),
                "n_violations_p<0.05": int((t["p"] < alpha).sum()),
                "violations": t.index[t["p"] < alpha].tolist(),
            }
            for name, t in ph_tables.items()
        },
    }
    tables = {
        **{k: v.reset_index(names="covariate") for k, v in ph_tables.items()},
        "cox_summary": models["Cox PH"].summary.reset_index(),
    }
    return metrics, {"models": models, "per_customer": per_customer}, tables


# --------------------------------------------------------------------------------------
# Orchestration + artifacts
# --------------------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(version_name: str, data_path: Path | None) -> dict:
    packages = {}
    for pkg in [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "optuna",
        "lifelines",
        "scikit-survival",
    ]:
        try:
            packages[pkg] = version(pkg)
        except PackageNotFoundError:  # pragma: no cover
            packages[pkg] = None
    return {
        "version": version_name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "data_sha256": sha256_of(data_path) if data_path and Path(data_path).exists() else None,
        "python": platform.python_version(),
        "packages": packages,
    }


def new_version_name() -> str:
    sha = _git("rev-parse", "--short", "HEAD") or "nogit"
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{sha}"


def run(
    df: pd.DataFrame,
    cfg: Config = CONFIG,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path | None = REPORTS_DIR,
    data_path: Path | None = RAW_DATA_PATH,
) -> Path:
    """Train everything on the cleaned dataframe ``df``; return the artifact directory."""
    set_global_seed(cfg.seed)
    version_name = new_version_name()
    out = Path(models_dir) / version_name
    out.mkdir(parents=True, exist_ok=False)
    logger.info("Writing artifacts to %s", out)

    train_df, test_df = split_train_test(df, cfg.split.test_size, cfg.seed)

    clf_metrics, clf_models, clf_tables = train_classification(train_df, test_df, cfg)
    biz_metrics, biz = evaluate_business(
        clf_tables["oof_predictions"], train_df, clf_tables["test_predictions"], cfg
    )
    surv_metrics, surv, surv_tables = train_survival(train_df, test_df, cfg)

    test_predictions = clf_tables["test_predictions"].assign(
        revenue_at_risk=biz["revenue_at_risk"], targeted=biz["targeted"]
    )
    test_predictions = test_predictions.merge(surv["per_customer"], on=ID_COL, how="left")

    metrics = {
        "run": run_metadata(version_name, data_path),
        "config": cfg.to_dict(),
        "data": {
            "n_rows": len(df),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "churn_rate_train": float(train_df[TARGET_COL].mean()),
            "churn_rate_test": float(test_df[TARGET_COL].mean()),
        },
        "classification": clf_metrics,
        "business": biz_metrics,
        "survival": surv_metrics,
    }

    joblib.dump(clf_models["classifier"], out / "classifier.joblib")
    joblib.dump(clf_models["base_models"], out / "base_classifiers.joblib")
    joblib.dump(surv["models"], out / "survival_models.joblib", compress=3)
    tables = {
        **{k: v for k, v in clf_tables.items() if k != "test_predictions"},
        **biz["tables"],
        **surv_tables,
        "test_predictions": test_predictions,
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    (Path(models_dir) / LATEST_POINTER).write_text(version_name + "\n")
    if reports_dir is not None:
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "metrics.json", Path(reports_dir) / "metrics.json")

    logger.info("Done. Test ROC-AUC %.3f", clf_metrics["test"]["final"]["roc_auc"]["value"])
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train churn classification + survival models.")
    parser.add_argument("--n-trials", type=int, default=None, help="override the Optuna budget")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = CONFIG
    if args.n_trials is not None:
        cfg = dataclasses.replace(
            cfg, tuning=dataclasses.replace(cfg.tuning, n_trials=args.n_trials)
        )
    run(load_clean_data(), cfg, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
