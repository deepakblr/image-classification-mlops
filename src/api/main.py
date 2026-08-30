"""FastAPI inference service with logging and Prometheus metrics."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.inference import get_model_path, load_model, predict_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("cats_dogs_api")

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions served",
    ["predicted_class"],
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        load_model()
        logger.info("Loaded model from %s", get_model_path())
    except FileNotFoundError as exc:
        logger.warning("Model not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title="Cats vs Dogs Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next: Callable):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
    logger.info(
        "request method=%s path=%s status=%s latency_ms=%.2f",
        request.method,
        endpoint,
        response.status_code,
        elapsed * 1000,
    )
    return response


@app.get("/health")
def health() -> dict:
    model_path = get_model_path()
    model_loaded = model_path.exists()
    status = "healthy" if model_loaded else "degraded"
    return {
        "status": status,
        "model_loaded": model_loaded,
        "model_path": str(model_path),
    }


@app.get("/metrics")
def metrics() -> Response:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image upload, got content_type={file.content_type}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")

    try:
        result = predict_bytes(data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    PREDICTIONS_TOTAL.labels(result["label"]).inc()
    logger.info(
        "prediction label=%s confidence=%.4f filename=%s",
        result["label"],
        result["confidence"],
        file.filename,
    )
    return result
