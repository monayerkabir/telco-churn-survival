# syntax=docker/dockerfile:1
#
# Two images from one file, sharing every layer up to the model bundle:
#   docker build --target api -t telco-churn-api .    # FastAPI on :8000
#   docker build --target app -t telco-churn-app .    # Streamlit on :8501
# or simply: docker compose up --build
#
# PYTHON_IMAGE can point at a registry mirror if Docker Hub rate-limits your pulls, e.g.
#   --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.11-slim

ARG PYTHON_IMAGE=python:3.11-slim

FROM ${PYTHON_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Resolve data/ and model_bundle/ under /app (the package is installed in site-packages).
    CHURN_PROJECT_ROOT=/app \
    CHURN_MODEL_DIR=/app/model_bundle \
    # shap's numba kernels cache compiled code; site-packages is read-only for the app user.
    NUMBA_CACHE_DIR=/tmp/numba

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code changes don't invalidate this (slow) layer.
COPY requirements/serve.txt requirements/serve.txt
RUN pip install -r requirements/serve.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps . \
    && useradd --create-home --uid 1000 churn \
    && mkdir -p /app/data \
    && chown churn /app/data

COPY model_bundle ./model_bundle


FROM base AS api

COPY api ./api
USER churn
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS app

COPY app ./app
COPY .streamlit ./.streamlit
USER churn
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"]
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
