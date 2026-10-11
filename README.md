# Predictive Maintenance MLOps Demo

[![CI](https://github.com/danielaistudent/predictive-maintenance-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/danielaistudent/predictive-maintenance-mlops/actions/workflows/ci.yml)

> End-to-end MLOps pipeline for industrial sensor anomaly detection.

An anomaly detection system built on industrial sensor data, demonstrating the full ML lifecycle: from data generation through model training to containerized serving, with CI/CD and monitoring.

## What it does

- Synthetic industrial sensor data generation (NASA CMAPSS structure)
- Unit-level train/val/test split — preventing data leakage
- Isolation Forest anomaly detection model
- MLflow experiment tracking and model registry
- FastAPI serving (in Docker container)
- GitHub Actions CI/CD — testing + Docker build
- Monitoring — drift detection

## Architecture

Data Generator (src/generate_data.py)
|
v
Data Module (src/data.py) -- unit-level split (GroupShuffleSplit)
|
v
Training (src/train.py) -- IsolationForest --> MLflow Tracking + Registry
|
v
FastAPI Serving (app/main.py) -- Docker container -- /predict
|
v
Monitoring (src/monitor.py) -- drift detection

CI/CD: GitHub Actions -> test + docker build

## Quick start

### 1. Clone and environment

    git clone https://github.com/danielaistudent/predictive-maintenance-mlops.git
    cd predictive-maintenance-mlops

    python -m venv venv
    source venv/Scripts/activate   # Windows
    # source venv/bin/activate     # Linux/Mac

    pip install -r requirements.txt

### 2. Generate data and train

    python -m src.generate_data
    python -m src.train

### 3. Serving

    uvicorn app.main:app --reload --port 8000

Open: http://localhost:8000/docs

### 4. Docker

    docker build -t predictive-maintenance-api:0.1.0 .
    docker run -p 8000:8000 predictive-maintenance-api:0.1.0

### 5. Tests

    pytest tests/ -v

### 6. MLflow UI

    mlflow ui

Open: http://localhost:5000

## Project structure

    predictive-maintenance-mlops/
    |-- .github/workflows/ci.yml   # GitHub Actions CI/CD
    |-- app/main.py                # FastAPI serving
    |-- src/
    |   |-- data.py                # Data loading + split
    |   |-- generate_data.py       # Synthetic data generator
    |   |-- train.py               # Model training + MLflow
    |   |-- monitor.py             # Drift detection
    |-- tests/                     # pytest tests
    |-- notebooks/01_eda.ipynb     # Exploratory Data Analysis
    |-- Dockerfile                 # Containerization
    |-- requirements.txt           # Pinned dependencies
    |-- README.md

## Metrics

| Metric    | Validation | Test |
| --------- | ---------- | ---- |
| Precision | 0.39       | 0.39 |
| Recall    | 0.51       | 0.45 |
| F1        | 0.44       | 0.41 |
| ROC-AUC   | 0.66       | 0.64 |

Note: Isolation Forest is an unsupervised model — it does not use labels for training. The moderate metrics are expected; a supervised approach (e.g. Logistic Regression) would yield better results, but the MVP goal is to demonstrate the pipeline.

## Key decisions and trade-offs

### Data

- Synthetic generation instead of NASA CMAPSS — controlled anomalies and drift, reproducible, no external dependency
- Unit-level split — an entire unit's time series goes into only one set, avoiding data leakage

### Model

- Isolation Forest — unsupervised, fast, interpretable, designed for anomaly detection
- contamination=0.25 — from EDA (25.1% anomaly rate in data)

### MLOps

- MLflow — experiment tracking + model registry in one
- Docker — reproducibility, environment isolation
- GitHub Actions — CI/CD, automated testing + Docker build
- .gitattributes — LF line endings, binary file markers

### Data and model versioning

- Code -> Git
- Data -> gitignored, reproducible from generate_data.py
- Model -> MLflow Model Registry

## Next steps (in production)

- Feature engineering — rolling statistics, time-series transformations
- Supervised model — Logistic Regression / XGBoost for better F1
- DVC — data versioning
- Kubernetes — scalable deployment
- Prometheus + Grafana — monitoring
- Automatic retraining — on drift trigger
- LLM assistant — RAG for operator support

## Technologies

Python 3.12 · scikit-learn · MLflow · FastAPI · Docker · GitHub Actions · pytest · pandas · NumPy

## License

MIT
