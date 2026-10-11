"""
A src/train.py modul tesztjei.

Ezek a tesztek ellenőrzik:
- A modell tanítását
- A predikciót
- A metrikák számítását
"""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from src.train import evaluate_model, train_isolation_forest


# --- Fixture-ök ---

@pytest.fixture
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Egyszerű szintetikus adat a tesztekhez."""
    rng = np.random.default_rng(42)
    # 200 normál + 50 anomaly pont, 5 feature
    X_normal = rng.normal(0.5, 0.1, size=(200, 5))
    X_anomaly = rng.normal(0.8, 0.1, size=(50, 5))
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * 200 + [1] * 50)
    return X, y


# --- Tesztek ---

def test_train_returns_isolation_forest(simple_data) -> None:
    """A tanítás IsolationForest-et ad vissza."""
    X, _ = simple_data
    model = train_isolation_forest(X, contamination=0.2)
    assert isinstance(model, IsolationForest)


def test_predict_returns_valid_labels(simple_data) -> None:
    """A predikció -1 vagy 1 értéket ad."""
    X, _ = simple_data
    model = train_isolation_forest(X, contamination=0.2)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({-1, 1})


def test_evaluate_returns_metrics(simple_data) -> None:
    """A kiértékelés metrikákat ad vissza."""
    X, y = simple_data
    model = train_isolation_forest(X, contamination=0.2)
    metrics, y_pred = evaluate_model(model, X, y)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_anomaly_detection_works(simple_data) -> None:
    """A modell képes megkülönböztetni a normál és anomaly pontokat."""
    X, y = simple_data
    model = train_isolation_forest(X, contamination=0.2)
    metrics, _ = evaluate_model(model, X, y)
    # Az F1 jobb kell legyen, mint a random (0.5 körül)
    assert metrics["f1"] > 0.3, f"F1 túl alacsony: {metrics['f1']}"
