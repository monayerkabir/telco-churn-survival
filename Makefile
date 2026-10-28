.DEFAULT_GOAL := help
.PHONY: help install data train bundle api app docker up test lint format notebooks all clean
.NOTPARALLEL:  # the targets form one sequential pipeline

PYTHON_BIN ?= python3.11
VENV       ?= .venv
PYTHON     ?= $(VENV)/bin/python
INSTALLED  := $(VENV)/.installed
RAW_DATA   := data/raw/Telco-Customer-Churn.csv
NOTEBOOKS  := $(sort $(wildcard notebooks/*.ipynb))
LINT_PATHS := src api app tests scripts notebooks

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

$(INSTALLED): requirements.txt requirements/serve.txt requirements/dev.txt pyproject.toml
	$(PYTHON_BIN) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps
	# Register this interpreter as the notebooks' "python3" kernel.
	$(PYTHON) -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (ipykernel)"
	touch $@

install: $(INSTALLED)  ## Create .venv (Python 3.11) with the pinned dependencies (re-runs when they change)

$(RAW_DATA):
	$(PYTHON) scripts/download_data.py

data: $(RAW_DATA)  ## Download the raw dataset (verified by SHA-256) into data/raw/

train: $(RAW_DATA)  ## Train all models; write versioned artifacts + metrics.json to models/
	$(PYTHON) -m churn.train

bundle:  ## Export the latest training run to the committed deployment bundle (model_bundle/)
	$(PYTHON) -m churn.bundle

api:  ## Serve the API on http://localhost:8000 (docs at /docs) from model_bundle/
	$(PYTHON) -m uvicorn api.main:app --reload --port 8000

app:  ## Run the Streamlit app on http://localhost:8501 from model_bundle/
	$(PYTHON) -m streamlit run app/streamlit_app.py

docker:  ## Build the api and app images
	docker compose build

up:  ## Run the API (:8000) and app (:8501) in containers
	docker compose up --build

test:  ## Run the test suite (offline, synthetic data)
	$(PYTHON) -m pytest

lint:  ## Lint (ruff) and check formatting (black)
	$(PYTHON) -m ruff check $(LINT_PATHS)
	$(PYTHON) -m black --check $(LINT_PATHS)

format:  ## Auto-fix lint issues and format code
	$(PYTHON) -m ruff check --fix $(LINT_PATHS)
	$(PYTHON) -m black $(LINT_PATHS)

notebooks:  ## Re-execute all notebooks in place (requires `make train` first)
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS)

all: install data train notebooks test lint  ## From a fresh clone: install -> data -> train -> notebooks -> checks

clean:  ## Remove caches (keeps .venv, data/ and models/)
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} +
