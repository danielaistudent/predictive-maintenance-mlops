"""
FastAPI serving modul.

Felelősség:
- A tanított Isolation Forest modell betöltése
- HTTP endpointok biztosítása: health, info, predict, predict_batch
- Az anomaly score és a bináris predikció visszaadása

A modell **induláskor** töltődik be (nem minden kérésnél),
ez a teljesítmény szempontjából kritikus.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import get_feature_columns


# --- Konfiguráció ---

MODEL_PATH = Path("models/isolation_forest.pkl")
METRICS_PATH = Path("models/metrics.json")
N_FEATURES = 21


# --- Pydantic modellek (request/response séma) ---

class PredictRequest(BaseModel):
    """Egy megfigyelés predikciója."""
    features: list[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=f"A {N_FEATURES} szenzor értéke",
    )


class PredictBatchRequest(BaseModel):
    """Több megfigyelés egyszerre."""
    features: list[list[float]] = Field(
        ...,
        description=f"Lista, minden elem {N_FEATURES} szenzor értéke",
    )


class PredictResponse(BaseModel):
    """Egy predikció eredménye."""
    is_anomaly: bool
    anomaly_score: float = Field(
        ..., description="Minél magasabb, annál inkább anomaly"
    )
    raw_prediction: int = Field(
        ..., description="-1 = anomaly, 1 = normal (IsolationForest nyers kimenet)"
    )


class BatchPredictResponse(BaseModel):
    """Batch predikció eredménye."""
    predictions: list[PredictResponse]
    n_anomalies: int
    n_total: int
    anomaly_rate: float


class ModelInfo(BaseModel):
    """A modell metaadatai."""
    model_type: str
    n_features: int
    contamination: float
    n_estimators: int
    metrics: dict | None = None


# --- Globális állapot (a modell és metaadatok) ---

ml_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Az alkalmazás életciklusa.

    Induláskor betölti a modellt és a metrikákat.
    Leálláskor felszabadítja az erőforrásokat.
    """
    # Startup
    print("Modell betöltése...")
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"A modell nem található: {MODEL_PATH}\n"
            f"Futtasd először: python -m src.train"
        )

    model = joblib.load(MODEL_PATH)
    ml_models["model"] = model

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            ml_models["metrics"] = json.load(f)

    print(f"Modell betöltve: {MODEL_PATH}")
    print(f"  n_features:      {model.n_features_in_}")
    print(f"  contamination:   {model.contamination}")
    print(f"  n_estimators:    {model.n_estimators}")

    yield

    # Shutdown
    print("Leállás...")
    ml_models.clear()


# --- FastAPI alkalmazás ---

app = FastAPI(
    title="Predictive Maintenance API",
    description="Ipari szenzoradatok anomaly detection API (Isolation Forest)",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Endpointok ---

@app.get("/", tags=["health"])
def root() -> dict:
    """Health check — a szolgáltatás fut-e."""
    return {"status": "ok", "service": "predictive-maintenance-api"}


@app.get("/info", response_model=ModelInfo, tags=["info"])
def info() -> ModelInfo:
    """A betöltött modell metaadatai."""
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Modell még nem töltődött be")

    model = ml_models["model"]
    return ModelInfo(
        model_type=type(model).__name__,
        n_features=model.n_features_in_,
        contamination=float(model.contamination),
        n_estimators=model.n_estimators,
        metrics=ml_models.get("metrics"),
    )


def _predict_one(model, features: list[float]) -> PredictResponse:
    """Egy megfigyelés predikciója (belső segédfüggvény)."""
    X = np.array(features, dtype=float).reshape(1, -1)
    raw_pred = int(model.predict(X)[0])
    # Az anomaly score: minél negatívabb a decision_function, annál anomalybb
    # Ezért negáljuk, hogy pozitív érték jelentse az anomalyt
    anomaly_score = float(-model.decision_function(X)[0])

    return PredictResponse(
        is_anomaly=(raw_pred == -1),
        anomaly_score=anomaly_score,
        raw_prediction=raw_pred,
    )


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(request: PredictRequest) -> PredictResponse:
    """Egy megfigyelés predikciója."""
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Modell még nem töltődött be")

    return _predict_one(ml_models["model"], request.features)


@app.post("/predict_batch", response_model=BatchPredictResponse, tags=["predict"])
def predict_batch(request: PredictBatchRequest) -> BatchPredictResponse:
    """Több megfigyelés egyszerre."""
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Modell még nem töltődött be")

    if not request.features:
        raise HTTPException(status_code=400, detail="Üres features lista")

    # Ellenőrizzük, hogy minden sor N_FEATURES hosszú
    for i, row in enumerate(request.features):
        if len(row) != N_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"A {i}. sor hossza {len(row)}, várt: {N_FEATURES}",
            )

    model = ml_models["model"]
    predictions = [_predict_one(model, row) for row in request.features]

    n_anomalies = sum(p.is_anomaly for p in predictions)
    n_total = len(predictions)

    return BatchPredictResponse(
        predictions=predictions,
        n_anomalies=n_anomalies,
        n_total=n_total,
        anomaly_rate=n_anomalies / n_total if n_total > 0 else 0.0,
    )
