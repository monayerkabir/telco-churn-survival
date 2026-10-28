"""FastAPI service: churn probability, survival summary, revenue at risk and SHAP drivers.

Run locally with ``make api`` (or ``uvicorn api.main:app``). Models are loaded once at
startup from ``$CHURN_MODEL_DIR``, defaulting to the committed ``model_bundle/``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandera.errors
from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)
from churn import __version__
from churn.bundle import BUNDLE_DIR
from churn.inference import ChurnPredictor

logger = logging.getLogger("churn.api")

MODEL_DIR_ENV = "CHURN_MODEL_DIR"
N_ATTRIBUTES = 22  # business attributes the explanations are grouped into

TopK = Annotated[int, Query(ge=1, le=N_ATTRIBUTES, description="Number of SHAP drivers to return.")]
IncludeCurve = Annotated[bool, Query(description="Include the full conditional survival curve.")]


def get_predictor(request: Request) -> ChurnPredictor:
    return request.app.state.predictor


Predictor = Annotated[ChurnPredictor, Depends(get_predictor)]


def create_app(model_dir: str | Path | None = None) -> FastAPI:
    """Build the app. ``model_dir`` overrides ``$CHURN_MODEL_DIR`` (used by tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = Path(model_dir or os.environ.get(MODEL_DIR_ENV) or BUNDLE_DIR)
        app.state.predictor = ChurnPredictor.load(path)
        logger.info("Loaded model %s from %s", app.state.predictor.version, path)
        yield

    app = FastAPI(
        title="Telco churn API",
        version=__version__,
        summary="Who will churn, when, and is a retention offer worth it?",
        lifespan=lifespan,
    )

    @app.exception_handler(pandera.errors.SchemaErrors)
    async def schema_error(request: Request, exc: pandera.errors.SchemaErrors) -> JSONResponse:
        # Defence in depth: the Pydantic models should already reject these inputs.
        return JSONResponse(status_code=422, content={"detail": str(exc.failure_cases)})

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health(predictor: Predictor) -> dict:
        """Liveness plus the identity of the loaded models."""
        return {
            "status": "ok",
            "model_version": predictor.version,
            "champion": predictor.champion,
            "survival_model": predictor.survival_model_name,
            "threshold": predictor.threshold,
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
    def predict(
        customer: CustomerFeatures,
        predictor: Predictor,
        top_k: TopK = 5,
        include_curve: IncludeCurve = False,
    ) -> dict:
        """Score one customer."""
        (result,) = predictor.predict([customer.features()], top_k, include_curve)
        return {"customerID": customer.customerID, "model_version": predictor.version, **result}

    @app.post("/predict/batch", response_model=BatchResponse, tags=["predictions"])
    def predict_batch(
        request: BatchRequest,
        predictor: Predictor,
        top_k: TopK = 5,
        include_curve: IncludeCurve = False,
    ) -> dict:
        """Score up to 1,000 customers in one vectorised call (results keep input order)."""
        results = predictor.predict([c.features() for c in request.customers], top_k, include_curve)
        return {
            "model_version": predictor.version,
            "predictions": [
                {"customerID": c.customerID, **r}
                for c, r in zip(request.customers, results, strict=True)
            ],
        }

    return app


app = create_app()
