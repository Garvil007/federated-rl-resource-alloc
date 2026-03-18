from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import mlflow.pytorch
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
import time
import os

app = FastAPI(title="Fed-RL Resource Allocator API")

# ─── Prometheus Metrics ───
PREDICT_COUNT = Counter(
    "predictions_total",
    "Total predictions served",
    ["model_version"],
)
PREDICT_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
PREDICT_ERRORS = Counter(
    "prediction_errors_total",
    "Total prediction errors",
)
MODEL_VERSION = Gauge(
    "model_version_info",
    "Currently loaded model version",
)

# ─── Load Production Model ───
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "fed-rl-resource-alloc"
mlflow.set_tracking_uri(MLFLOW_URI)

try:
    MODEL = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/Production")
    CURRENT_VERSION = "production"
    MODEL_VERSION.set(1)
except Exception as e:
    print(f"Warning: Could not load production model: {e}")
    MODEL = None


# ─── Request/Response Schemas ───
class PredictRequest(BaseModel):
    observation: list[float]


class PredictResponse(BaseModel):
    action: list[int]
    confidence: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


# ─── Endpoints ───
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if MODEL else "degraded",
        model_loaded=MODEL is not None,
        model_name=MODEL_NAME,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    start = time.time()
    try:
        obs = torch.FloatTensor(req.observation).unsqueeze(0)
        with torch.no_grad():
            logits = MODEL(obs)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1)
            confidence = probs.max().item()

        latency = time.time() - start
        PREDICT_COUNT.labels(model_version=CURRENT_VERSION).inc()
        PREDICT_LATENCY.observe(latency)

        return PredictResponse(
            action=action.numpy().flatten().tolist(),
            confidence=confidence,
            latency_ms=latency * 1000,
        )
    except Exception as e:
        PREDICT_ERRORS.inc()
        raise HTTPException(500, str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/reload")
async def reload_model():
    """Hot-reload the production model without downtime."""
    global MODEL, CURRENT_VERSION
    try:
        MODEL = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/Production")
        CURRENT_VERSION = "production-reloaded"
        return {"status": "reloaded", "model": MODEL_NAME}
    except Exception as e:
        raise HTTPException(500, f"Reload failed: {e}")