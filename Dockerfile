# --- Base image ---
FROM python:3.12-slim

# --- Metadata ---
LABEL maintainer="OmenDaniel"
LABEL description="Predictive Maintenance MLOps API (Isolation Forest)"
LABEL version="0.1.0"

# --- Environment ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- Working directory ---
WORKDIR /app

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (separate layer for caching) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code ---
COPY src/ ./src/
COPY app/ ./app/

# --- Trained model ---
COPY models/ ./models/

# --- Port ---
EXPOSE 8000

# --- Health check ---
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# --- Run ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

