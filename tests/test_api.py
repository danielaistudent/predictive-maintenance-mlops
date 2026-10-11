"""
Az app/main.py FastAPI endpointok tesztjei.

Ezek a tesztek a TestClient-tel hívják az API-t,
nem indítanak valódi szervert.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


# --- Fixture-ök ---

@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI TestClient — a valódi szerver nélkül."""
    model_path = Path("models/isolation_forest.pkl")
    if not model_path.exists():
        pytest.skip(f"Modell nem található: {model_path}. Futtasd: python -m src.train")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_features() -> list[float]:
    """21 érvényes feature érték."""
    return [0.5] * 21


@pytest.fixture
def anomaly_features() -> list[float]:
    """21 magas érték (valószínűleg anomaly)."""
    return [0.9] * 21


# --- Tesztek ---

def test_root(client: TestClient) -> None:
    """A gyökér endpoint visszaad egy státuszt."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_info(client: TestClient) -> None:
    """Az /info endpoint a modell metaadatait adja."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["n_features"] == 21
    assert data["model_type"] == "IsolationForest"


def test_predict_valid(client: TestClient, valid_features: list[float]) -> None:
    """A /predict érvényes bemenetre válaszol."""
    response = client.post("/predict", json={"features": valid_features})
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert isinstance(data["is_anomaly"], bool)


def test_predict_invalid_length(client: TestClient) -> None:
    """A /predict hibát ad, ha nem 21 feature van."""
    response = client.post("/predict", json={"features": [0.5] * 10})
    assert response.status_code == 422  # Pydantic validation error


def test_predict_batch(client: TestClient, valid_features: list[float]) -> None:
    """A /predict_batch több megfigyelést fogad."""
    response = client.post(
        "/predict_batch",
        json={"features": [valid_features, valid_features, valid_features]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["n_total"] == 3
    assert "n_anomalies" in data
    assert "anomaly_rate" in data
