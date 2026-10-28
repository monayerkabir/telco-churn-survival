"""Loading, cleaning, validating and splitting the IBM Telco Customer Churn data."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from sklearn.model_selection import train_test_split

from churn.config import (
    CATEGORICAL_COLS,
    CATEGORY_LEVELS,
    CONFIG,
    DATA_URL,
    DURATION_COL,
    ID_COL,
    MAX_TENURE,
    RAW_DATA_PATH,
    RAW_DATA_SHA256,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

TARGET_MAP = {"No": 0, "Yes": 1}


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw_data(
    url: str = DATA_URL,
    dest: Path = RAW_DATA_PATH,
    expected_sha256: str | None = RAW_DATA_SHA256,
    force: bool = False,
) -> Path:
    """Download the raw CSV to ``dest`` and verify its checksum.

    The file is written to a temporary path first so an interrupted download never
    leaves a truncated CSV behind.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        logger.info("Raw data already present at %s", dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        logger.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as fh:
            fh.write(response.read())
        tmp.replace(dest)

    if expected_sha256 is not None:
        actual = sha256_of(dest)
        if actual != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {dest}: expected {expected_sha256}, got {actual}. "
                "The upstream file may have changed; results may not be reproducible."
            )
    return dest


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, keeping every non-numeric column as a string."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    return pd.read_csv(path, dtype={ID_COL: str, "TotalCharges": str})


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a typed, validated copy of the raw data.

    * ``TotalCharges`` is stored as text with blank strings for 11 brand-new customers
      (``tenure == 0``) who have not been billed yet, so the blanks become 0 rather than an
      imputed median. Any other missing value is filled with ``tenure * MonthlyCharges``.
    * ``SeniorCitizen`` (0/1) is recoded to "No"/"Yes" like every other binary flag.
    * ``Churn`` is encoded as an integer target: 1 = churned, 0 = active.
    """
    df = raw.copy()

    for col in [ID_COL, *CATEGORICAL_COLS, TARGET_COL]:
        if col == "SeniorCitizen":
            continue
        df[col] = df[col].astype(str).str.strip()

    df["SeniorCitizen"] = (
        pd.to_numeric(df["SeniorCitizen"], errors="raise").map({0: "No", 1: "Yes"}).astype(str)
    )

    df[DURATION_COL] = pd.to_numeric(df[DURATION_COL], errors="raise").astype("int64")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="raise").astype("float64")

    total = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    n_missing = int(total.isna().sum())
    if n_missing:
        logger.info("Filling %d missing TotalCharges with tenure * MonthlyCharges", n_missing)
    df["TotalCharges"] = total.fillna(df[DURATION_COL] * df["MonthlyCharges"]).astype("float64")

    unknown = set(df[TARGET_COL].unique()) - set(TARGET_MAP)
    if unknown:
        raise ValueError(f"Unexpected {TARGET_COL} labels: {sorted(unknown)}")
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAP).astype("int64")

    return validate(df)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def _services_consistent(df: pd.DataFrame) -> pd.Series:
    """Add-on services say "No internet service" exactly when InternetService is "No"."""
    no_internet = df["InternetService"].eq("No")
    ok = pd.Series(True, index=df.index)
    for col in CATEGORY_LEVELS:
        if "No internet service" in CATEGORY_LEVELS[col]:
            ok &= df[col].eq("No internet service") == no_internet
    ok &= df["MultipleLines"].eq("No phone service") == df["PhoneService"].eq("No")
    return ok


SCHEMA = pa.DataFrameSchema(
    columns={
        ID_COL: pa.Column(str, unique=True, nullable=False),
        **{
            col: pa.Column(str, pa.Check.isin(levels), nullable=False)
            for col, levels in CATEGORY_LEVELS.items()
        },
        DURATION_COL: pa.Column("int64", pa.Check.in_range(0, MAX_TENURE), nullable=False),
        "MonthlyCharges": pa.Column("float64", pa.Check.in_range(0, 500), nullable=False),
        "TotalCharges": pa.Column("float64", pa.Check.ge(0), nullable=False),
        TARGET_COL: pa.Column("int64", pa.Check.isin([0, 1]), nullable=False),
    },
    checks=[
        pa.Check(
            lambda df: (df[DURATION_COL] > 0) | df["TotalCharges"].eq(0),
            error="customers with tenure 0 cannot have been billed",
        ),
        pa.Check(_services_consistent, error="service columns are mutually inconsistent"),
    ],
    strict=True,
    ordered=False,
    coerce=False,
    name="telco_clean",
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate against :data:`SCHEMA`, reporting every failure at once."""
    return SCHEMA.validate(df, lazy=True)


def load_clean_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return clean_data(load_raw_data(path))


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------
def split_train_test(
    df: pd.DataFrame,
    test_size: float = CONFIG.split.test_size,
    seed: int = CONFIG.seed,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified hold-out split shared by the classification and survival models."""
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
