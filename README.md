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
make bundle     # export the latest run to model_bundle/ (what the API and app serve)
make test       # run the test suite
make lint       # ruff + black checks
make notebooks  # re-execute the analysis notebooks in notebooks/
```

## Serving

```bash
make api        # FastAPI on http://localhost:8000 (interactive docs at /docs)
make app        # Streamlit app on http://localhost:8501
make up         # both, in Docker (docker compose up --build)
```

Example request:

```bash
curl -X POST "localhost:8000/predict?top_k=3" -H "content-type: application/json" \
  -d '{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
       "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
       "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
       "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
       "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5}'
```

Both services load the committed `model_bundle/` by default; set `CHURN_MODEL_DIR` to a
`models/<version>/` directory to serve a fresh training run instead.

**Streamlit Community Cloud:** deploy `app/streamlit_app.py` from this repository with
Python 3.11. Dependencies come from `app/requirements.txt`, the models from
`model_bundle/`, and the dataset is downloaded (checksum-verified) on first use.

## Layout

```
src/churn/       package: config, data, features, train, survival, evaluate, business,
                 plots, inference (shared scoring layer), bundle (deployment export)
api/             FastAPI service (schemas + app)
app/             Streamlit app (standalone; loads models directly)
model_bundle/    slim, checksummed model bundle served by the API and app
notebooks/       01_eda, 02_modeling, 03_survival (executed, with outputs)
scripts/         download_data.py
tests/           pytest suite (offline: synthetic data, TestClient, Streamlit AppTest)
reports/         figures/ and the metrics.json of the reference training run
models/          versioned training artifacts (gitignored)
requirements/    serve.txt (runtime pins) and dev.txt (training, notebooks, tests)
Dockerfile       api and app targets; docker-compose.yml runs both
```

Licensed under MIT.
