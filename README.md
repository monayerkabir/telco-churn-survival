# telco-churn-survival

> **Work in progress.** A full write-up (results, figures, design decisions) lands at
> the end of the project. This placeholder only covers how to run things.

Predicting **whether** and **when** telecom customers churn (classification + survival
analysis on the IBM Telco Customer Churn dataset), and turning those predictions into a
retention-targeting decision.

## Quickstart

```bash
make install    # create .venv (Python 3.11) and install pinned dependencies
make data       # download the raw CSV into data/raw/
make train      # train all models, write versioned artifacts + metrics.json to models/
make test       # run the test suite
make lint       # ruff + black checks
make notebooks  # re-execute the analysis notebooks in notebooks/
```

## Layout

```
src/churn/     package: config, data, features, train, survival, evaluate, business, plots
notebooks/     01_eda, 02_modeling, 03_survival (executed, with outputs)
scripts/       download_data.py
tests/         pytest suite (runs offline on synthetic data)
reports/       figures/ and the metrics.json of the latest training run
models/        versioned training artifacts (gitignored)
```

Licensed under MIT.
