"""Export a slim deployment bundle from a training run.

``models/`` is gitignored, but the API image and the Streamlit app (which deploys straight
from the repository to Streamlit Community Cloud) need models. The bundle keeps only what
inference uses: the deployed classifier, the champion pipeline (for SHAP explanations),
the primary survival model and the run's metrics. It mirrors the run-directory layout, so
:class:`churn.inference.ChurnPredictor` loads either. ``manifest.json`` records the source
run, library versions and a SHA-256 per file, which the loader verifies.

Usage:
    python -m churn.bundle [--source models/<version>] [--dest model_bundle]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib

from churn.config import PROJECT_ROOT, latest_model_dir
from churn.data import sha256_of

logger = logging.getLogger(__name__)

BUNDLE_DIR = PROJECT_ROOT / "model_bundle"
MANIFEST = "manifest.json"
# Libraries whose versions must match when unpickling the models.
PINNED_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "lifelines",
    "scikit-survival",
    "shap",
    "joblib",
]


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for pkg in PINNED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:  # pragma: no cover
            versions[pkg] = None
    return versions


def export_bundle(source: Path, dest: Path = BUNDLE_DIR) -> Path:
    """Write the deployment bundle for training run ``source`` into ``dest``."""
    source, dest = Path(source), Path(dest)
    metrics = json.loads((source / "metrics.json").read_text())
    champion = metrics["classification"]["champion"]
    primary = metrics["survival"]["primary_model"]

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    shutil.copyfile(source / "classifier.joblib", dest / "classifier.joblib")
    base = joblib.load(source / "base_classifiers.joblib")
    joblib.dump({champion: base[champion]}, dest / "base_classifiers.joblib", compress=3)

    surv = joblib.load(source / "survival_models.joblib")[primary]
    if hasattr(surv, "training_design_"):
        surv.training_design_ = None  # only needed for the PH test, not for prediction
    joblib.dump({primary: surv}, dest / "survival_models.joblib", compress=3)

    shutil.copyfile(source / "metrics.json", dest / "metrics.json")
    if (source / "feature_means.json").exists():
        shutil.copyfile(source / "feature_means.json", dest / "feature_means.json")

    files = sorted(p.name for p in dest.iterdir())
    manifest = {
        "source_run": metrics["run"]["version"],
        "git_sha": metrics["run"]["git_sha"],
        "champion": champion,
        "survival_model": primary,
        "packages": _package_versions(),
        "sha256": {name: sha256_of(dest / name) for name in files},
    }
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Bundle for run %s written to %s", manifest["source_run"], dest)
    return dest


def verify_bundle(path: Path) -> list[str]:
    """Problems with a bundle: files whose checksum differs from the manifest, and
    installed library versions that differ from the ones the bundle was built with.

    A run directory without a manifest has nothing to verify and returns no problems.
    """
    path = Path(path)
    manifest_path = path / MANIFEST
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    problems = [
        f"{name}: checksum mismatch"
        for name, expected in manifest["sha256"].items()
        if not (path / name).exists() or sha256_of(path / name) != expected
    ]
    installed = _package_versions()
    problems += [
        f"{pkg}: bundle built with {built}, installed {installed.get(pkg)}"
        for pkg, built in manifest["packages"].items()
        if installed.get(pkg) != built
    ]
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the deployment model bundle.")
    parser.add_argument("--source", type=Path, default=None, help="training run directory")
    parser.add_argument("--dest", type=Path, default=BUNDLE_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export_bundle(args.source or latest_model_dir(), args.dest)


if __name__ == "__main__":
    main()
