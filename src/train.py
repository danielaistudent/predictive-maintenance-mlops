"""
Modell tanító modul.

Felelősség:
- Az adat betöltése és splitelése
- Isolation Forest modell tanítása
- Metrikák számítása (precision, recall, F1, ROC-AUC)
- MLflow experiment tracking: paraméterek, metrikák, modell

Ez a pipeline lelke. A kimenet egy tanított modell, amit az
app/main.py (FastAPI serving) fog használni.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_RANDOM_STATE,
    load_raw_data,
    prepare_xy,
    split_by_unit,
)


# --- Konfiguráció ---

DEFAULT_MODEL_PATH = Path("models/isolation_forest.pkl")
DEFAULT_METRICS_PATH = Path("models/metrics.json")
MLFLOW_EXPERIMENT_NAME = "predictive-maintenance"


def train_isolation_forest(
    X_train: np.ndarray | "pd.DataFrame",
    contamination: float = 0.25,
    n_estimators: int = 100,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> IsolationForest:
    """Isolation Forest modell tanítása."""
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def evaluate_model(
    model: IsolationForest, X: np.ndarray, y_true: np.ndarray
) -> dict:
    """
    A modell kiértékelése.

    Az IsolationForest.predict() -1-et ad anomaly-ra, 1-et normálra.
    Ezt átalakítjuk 1/0-ra, hogy a metrikák értelmezhetők legyenek.
    """
    y_pred_raw = model.predict(X)  # -1 vagy 1
    y_pred = (y_pred_raw == -1).astype(int)  # 1 = anomaly

    # Anomaly score: minél negatívabb, annál anomalybb
    # A ROC-AUC-hoz pozitív irányba fordítjuk
    y_score = -model.decision_function(X)

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC-AUC csak akkor értelmezhető, ha mindkét osztály jelen van
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)

    return metrics, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolation Forest tanítás")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--contamination", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    # --- MLflow experiment beállítása ---
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        # --- Adat betöltése és split ---
        print("\n[1/5] Adat betöltése...")
        df = load_raw_data(args.data_path)
        train_df, val_df, test_df = split_by_unit(df, random_state=args.random_state)

        X_train, y_train = prepare_xy(train_df)
        X_val, y_val = prepare_xy(val_df)
        X_test, y_test = prepare_xy(test_df)

        print(f"  Train: {len(X_train):>6,} sor")
        print(f"  Val:   {len(X_val):>6,} sor")
        print(f"  Test:  {len(X_test):>6,} sor")

        # --- Paraméterek logolása ---
        mlflow.log_params({
            "model_type": "IsolationForest",
            "contamination": args.contamination,
            "n_estimators": args.n_estimators,
            "random_state": args.random_state,
            "n_features": X_train.shape[1],
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        })

        # --- Modell tanítása ---
        print("\n[2/5] Modell tanítása...")
        model = train_isolation_forest(
            X_train,
            contamination=args.contamination,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
        )

        # --- Kiértékelés ---
        print("\n[3/5] Kiértékelés...")
        val_metrics, _ = evaluate_model(model, X_val, y_val)
        test_metrics, y_test_pred = evaluate_model(model, X_test, y_test)

        print("\n  Validation:")
        for k, v in val_metrics.items():
            print(f"    {k:10s}: {v:.4f}")

        print("\n  Test:")
        for k, v in test_metrics.items():
            print(f"    {k:10s}: {v:.4f}")

        # --- Metrikák logolása ---
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # --- Modell mentése ---
        print("\n[4/5] Modell mentése...")
        args.model_path.parent.mkdir(parents=True, exist_ok=True)

        import joblib
        joblib.dump(model, args.model_path)
        print(f"  Modell: {args.model_path}")

        # Metrikák JSON-ba
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_path, "w") as f:
            json.dump(
                {"val": val_metrics, "test": test_metrics},
                f,
                indent=2,
            )
        print(f"  Metrikák: {args.metrics_path}")

        # --- MLflow model logging ---
        print("\n[5/5] MLflow model logging...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5),
            registered_model_name="predictive-maintenance-isolation-forest",
        )

        print("\n✓ Kész!")
        print(f"  MLflow run ID: {run.info.run_id}")
        print(f"  MLflow UI:     mlflow ui")


if __name__ == "__main__":
    main()
